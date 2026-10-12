"""When the 300 s agent limit fires, say how far the agent got.

Before, the agent's result (tool trace included) was discarded and the user
got only a generic "took longer than the limit" message (14% of tester
replies on Oct 9). A job submitted before the cut-off also lost its
submission_id, so it got no workflow watch and no completion message.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared import agent_dispatch  # noqa: E402
from shared.agent_dispatch import timeout_partial_result  # noqa: E402
from shared.models import BaseAgentState, ToolCall  # noqa: E402

VIEW = "https://www.bv-brc.org/view/GenomeList/?eq(genus,Streptococcus)"


def _step(tool, result=None, error=None, **args):
    return {"tool": tool, "arguments": args, "result": result, "error": error}


def test_nothing_done_returns_none():
    assert timeout_partial_result("data", [], 300) is None


def test_data_steps_are_listed_with_counts_and_links():
    progress = [
        _step("search_data", {"numFound": 1234, "viewer_url": VIEW}, collection="genome"),
        _step("facet_query", {"count": 2}, collection="genome_amr"),
        _step("search_data", None, error="400 Bad Request", collection="genome_amr"),
    ]
    r = timeout_partial_result("data", progress, 300)
    assert r["status"] == "error" and r["timed_out"]
    assert "the 5-minute limit" in r["answer"]
    assert "`search_data` (genome): 1,234 records, [view](<" + VIEW + ">)" in r["answer"]
    assert "`search_data` (genome_amr): failed — 400 Bad Request" in r["answer"]
    assert len(r["tool_trace"]) == 3


def test_job_submitted_before_the_cutoff_keeps_its_submission_id():
    progress = [
        _step("get_workflow_inputs", {"inputs": []}, workflow_id="GenomeAssembly"),
        _step("submit_gowe_job", {"submission_id": "sub_9", "output_path": "/u/home/chats/s1/asm"},
              workflow_id="GenomeAssembly"),
    ]
    r = timeout_partial_result("service", progress, 300)
    assert r["status"] == "completed"
    assert r["submission_ids"] == ["sub_9"]
    assert "job submitted, results in `/u/home/chats/s1/asm`" in r["answer"]
    assert "completion message will appear" in r["answer"]


def test_record_execution_feeds_the_progress_list():
    from shared.models import TOOL_PROGRESS
    sink: list = []
    tok = TOOL_PROGRESS.set(sink)
    try:
        BaseAgentState().record_execution(ToolCall(id="c0", name="search_data", arguments={"collection": "genome"}),
                                          result={"numFound": 5})
    finally:
        TOOL_PROGRESS.reset(tok)
    assert sink == [{"tool": "search_data", "arguments": {"collection": "genome"}, "result": {"numFound": 5}, "error": None}]
    # No list set: nothing collected, nothing breaks.
    BaseAgentState().record_execution(ToolCall(id="c1", name="search_data", arguments={}), result={})


def _handle(timeout):
    from orchestrator.registry.agent_handle import InProcessAgentHandle
    cfg = SimpleNamespace(name="Data", description="d", capabilities=[], endpoint=None,
                          chat_tool_params={"agent_type": "data"}, timeout_seconds=timeout)
    return InProcessAgentHandle("data", cfg)


@pytest.mark.asyncio
async def test_handle_returns_partial_result_on_timeout():
    async def slow_dispatch(**kwargs):
        state = BaseAgentState()
        state.record_execution(ToolCall(id="c0", name="search_data", arguments={"collection": "genome"}),
                               result={"numFound": 42})
        await asyncio.sleep(5)

    with patch.object(agent_dispatch, "dispatch_agent", slow_dispatch):
        r = await _handle(0.2).call_tool("agent_chat", {"query": "q"})
    assert r["timed_out"] and "42 records" in r["answer"]


@pytest.mark.asyncio
async def test_handle_reraises_when_nothing_was_done():
    async def slow_dispatch(**kwargs):
        await asyncio.sleep(5)

    with patch.object(agent_dispatch, "dispatch_agent", slow_dispatch):
        with pytest.raises(asyncio.TimeoutError):
            await _handle(0.2).call_tool("agent_chat", {"query": "q"})


@pytest.mark.asyncio
async def test_handle_passes_normal_results_through():
    async def fast_dispatch(**kwargs):
        return {"answer": "done", "status": "completed"}

    with patch.object(agent_dispatch, "dispatch_agent", fast_dispatch):
        r = await _handle(5).call_tool("agent_chat", {"query": "q"})
    assert r == {"answer": "done", "status": "completed"}
