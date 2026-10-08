"""Empty agent answers after successful tools (fix_empty_answer_summary.md).

Testers had correct data (e.g. 4932 penicillin-resistant genomes) and saw
"... unable to produce a textual summary" with the tool's raw arguments.
Covered here:
  - the shared agent loop nudges once when the final reply is empty
  - the tool trace carries a small result summary (counts, links, errors)
  - synthesis is shown those summaries when the answer is empty
  - the fallback reports what was found, not the query arguments
"""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest

# Repo root on sys.path so ``shared`` is importable (as in test_shared_tools.py).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import shared.agent_loop as agent_loop
from shared.agent_dispatch import _build_tool_trace, _summarize_tool_result
from shared.agent_messages import EMPTY_ANSWER_NUDGE
from shared.models import BaseAgentConfig, BaseAgentState, ToolCall, ToolExecution
from orchestrator.synthesizer.prompts import build_synthesis_prompt
from orchestrator.synthesizer.synthesizer import _build_fallback

SEARCH_RESULT = {
    "numFound": 4932,
    "docs": [{"genome_id": "1.1"}] * 25,
    "viewer_url": "https://www.bv-brc.org/view/GenomeList/?eq(antibiotic,penicillin)",
    "download_tsv_url": "https://www.bv-brc.org/api/genome_amr/?eq(antibiotic,penicillin)&http_accept=text/tsv",
}


def _response(content=None, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=tool_calls or None),
    )])


def _tool_call(name="search_data", args=None):
    return SimpleNamespace(id="call_1", function=SimpleNamespace(
        name=name, arguments=json.dumps(args or {"collection": "genome_amr"})))


def _config():
    return BaseAgentConfig(llm_base_url="http://x/v1", llm_api_key="k", llm_model="m",
                           max_iterations=6)


async def _run(replies, monkeypatch):
    """Drive run_agent_loop with scripted LLM replies; returns (state, prompts)."""
    seen_messages = []
    scripted = list(replies)

    async def fake_completion(client, messages, tools, config):
        seen_messages.append([dict(m) for m in messages])
        return scripted.pop(0)

    async def fake_execute_tool(**kwargs):
        return SEARCH_RESULT

    monkeypatch.setattr(agent_loop, "execute_tool", fake_execute_tool)
    state = BaseAgentState()
    state.add_user_message("How many genomes are penicillin resistant?")
    await agent_loop.run_agent_loop(
        state=state, config=_config(), client=None, system_prompt="sys",
        chat_completion_fn=fake_completion,
    )
    return state, seen_messages


@pytest.mark.asyncio
async def test_empty_reply_after_tools_is_nudged_once(monkeypatch):
    state, seen = await _run([
        _response(tool_calls=[_tool_call()]),
        _response(content=None),                      # thinking ate the budget
        _response(content="4,932 genomes are resistant."),
    ], monkeypatch)
    assert state.status == "completed"
    assert state.final_answer == "4,932 genomes are resistant."
    assert seen[-1][-1] == {"role": "user", "content": EMPTY_ANSWER_NUDGE}


@pytest.mark.asyncio
async def test_nudge_fires_only_once(monkeypatch):
    state, seen = await _run([
        _response(tool_calls=[_tool_call()]),
        _response(content=""),
        _response(content=""),
    ], monkeypatch)
    assert state.status == "completed"
    assert state.final_answer == ""
    nudges = [m for m in seen[-1] if m.get("content") == EMPTY_ANSWER_NUDGE]
    assert len(nudges) == 1


@pytest.mark.asyncio
async def test_empty_reply_without_tools_is_not_nudged(monkeypatch):
    state, seen = await _run([_response(content="")], monkeypatch)
    assert state.final_answer == ""
    assert len(seen) == 1


def test_result_summary_keeps_counts_and_links_not_docs():
    s = _summarize_tool_result(SEARCH_RESULT)
    assert s["numFound"] == 4932
    assert s["docs_returned"] == 25
    assert s["viewer_url"].startswith("https://www.bv-brc.org/view/")
    assert "docs" not in s
    assert _summarize_tool_result({"data": "x" * 50_000}) is None
    assert _summarize_tool_result("not a dict") is None


def test_large_distinct_counts_are_dropped_to_fit():
    big = {"numFound": 10, "distinct_counts": {f"f{i}": i for i in range(200)}}
    s = _summarize_tool_result(big)
    assert s == {"numFound": 10}


def test_tool_trace_carries_result_summary():
    result = SimpleNamespace(tool_trace=[ToolExecution(
        tool_call=ToolCall(id="1", name="search_data", arguments={"q": "x"}),
        result=SEARCH_RESULT, duration_ms=5.0,
    )])
    (entry,) = _build_tool_trace(result)
    assert entry["result_summary"]["numFound"] == 4932
    assert entry["arguments"] == {"q": "x"}


def _empty_result():
    return {"agent": "data", "status": "completed", "answer": "", "tool_trace": [
        {"tool": "search_data", "arguments": {"collection": "genome_amr", "q": "antibiotic:penicillin"},
         "result_summary": _summarize_tool_result(SEARCH_RESULT)},
    ]}


def test_synthesis_prompt_includes_tool_results_when_answer_empty():
    _, user = build_synthesis_prompt("how many?", [_empty_result()])
    assert "What the agent's tools returned" in user
    assert "4932" in user


def test_synthesis_prompt_omits_them_when_answer_present():
    r = _empty_result() | {"answer": "4,932 genomes."}
    _, user = build_synthesis_prompt("how many?", [r])
    assert "What the agent's tools returned" not in user


def test_fallback_reports_findings_not_arguments():
    text = _build_fallback([_empty_result()])
    assert "4,932 records" in text
    assert "[view results](<https://www.bv-brc.org/view/GenomeList/" in text
    assert "antibiotic:penicillin" not in text          # no raw query args
    assert "unable to produce a textual summary" not in text
