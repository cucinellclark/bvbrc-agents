"""The populate loop's mid-loop corrections must not leak into replies.

The corrections go in as user messages (vLLM/Qwen reject a mid-conversation
system message), and the model answered them as if the user had said them:
"You're absolutely right – I haven't submitted any job yet…" on a first turn,
20+ times across the Sep codeathon and Oct 9 (Chris, dovie, cmao, kcoakley,
parrello).  Most of those fired on a reply that was legitimately asking the
user a question.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import shared.agent_dispatch  # noqa: E402, F401  -- sets up agent package paths
from service_agent.models import AgentConfig, AgentState  # noqa: E402
from service_agent.phases import populate  # noqa: E402

WF = "wf_df8b0db2-0898-4d1c-82a4-f4a9260a26b8"


def _reply(content="", tool_calls=None):
    calls = [SimpleNamespace(id=f"c{i}", type="function",
                             function=SimpleNamespace(name=n, arguments=json.dumps(a)))
             for i, (n, a) in enumerate(tool_calls or [])]
    msg = SimpleNamespace(content=content, tool_calls=calls or None, reasoning=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])


GET_INPUTS = _reply(tool_calls=[("get_workflow_inputs", {"workflow_id": WF})])


async def _run(replies, tool_results, mode="execute"):
    cfg = AgentConfig(llm_base_url="http://llm", llm_api_key="k", llm_model="m",
                      execution_mode=mode)
    state = AgentState(query="run variation on SRR123", context={"execution_mode": mode})
    llm = AsyncMock(side_effect=list(replies))
    with patch.object(populate, "create_client", lambda c: object()), \
         patch.object(populate, "chat_completion", llm), \
         patch.object(populate, "execute_tool", AsyncMock(side_effect=list(tool_results))):
        out = await populate.populate_and_submit("run variation on SRR123", cfg, state)
    return out, llm


def _user_messages(state):
    return [m["content"] for m in state.messages if m["role"] == "user"]


@pytest.mark.asyncio
async def test_question_after_get_inputs_is_not_nudged():
    question = "To run Variation I need a reference genome. Which genome ID should I use?"
    out, llm = await _run([GET_INPUTS, _reply(question)], [{"inputs": []}])
    assert llm.await_count == 2                      # no extra nudge round-trip
    assert out.status == "needs_input"
    assert out.question == question
    assert not any("SYSTEM CHECK" in m for m in _user_messages(out))


@pytest.mark.asyncio
async def test_please_confirm_counts_as_a_question():
    text = "I've prepared Variation with SRR123 against 83332.12. Please confirm and I'll submit."
    out, llm = await _run([GET_INPUTS, _reply(text)], [{"inputs": []}])
    assert llm.await_count == 2
    assert out.status == "needs_input"


@pytest.mark.asyncio
async def test_fake_success_with_a_question_is_still_nudged():
    """The guard exists for 'submitted successfully!' with no submit call."""
    fake = "Your Variation job has been submitted successfully! Anything else?"
    honest = "I could not submit yet: the reference genome is missing. Which genome should I use?"
    out, llm = await _run([GET_INPUTS, _reply(fake), _reply(honest)], [{"inputs": []}])
    assert llm.await_count == 3
    nudges = [m for m in _user_messages(out) if "SYSTEM CHECK" in m]
    assert len(nudges) == 1 and "NOT submitted any job" in nudges[0]
    assert out.question == honest


@pytest.mark.asyncio
async def test_echoed_opener_after_a_nudge_is_stripped():
    claims_ready = "Everything is set for the Variation job."
    echo = ("You're absolutely right – I haven't submitted any job yet. "
            "The reference genome is missing, so I can't submit. Which genome should I use?")
    out, _ = await _run([GET_INPUTS, _reply(claims_ready), _reply(echo)], [{"inputs": []}])
    assert out.question.startswith("The reference genome is missing")
    assert "absolutely right" not in out.question


@pytest.mark.asyncio
async def test_opener_without_a_nudge_is_left_alone():
    text = "You're right to ask. Which genome ID should I use as the reference?"
    out, _ = await _run([GET_INPUTS, _reply(text)], [{"inputs": []}])
    assert out.question == text


@pytest.mark.asyncio
async def test_every_nudge_carries_the_system_check_prefix():
    """Plan-mode success claim → PLAN_MODE_NUDGE, which must be marked too."""
    blocked = {"error": "PLAN mode", "blocked_by_mode": True, "tool": "submit_gowe_job",
               "arguments": {"workflow_id": WF, "inputs": {}}, "summary": "Variation"}
    replies = [
        GET_INPUTS,
        _reply(tool_calls=[("submit_gowe_job", {"workflow_id": WF, "inputs": {"srr_ids": ["SRR123"]}})]),
        _reply("Your job has been submitted successfully."),
        _reply("You're correct. Ready to submit: Variation on SRR123. Switch to Execute mode to run it."),
    ]
    out, _ = await _run(replies, [{"inputs": []}, blocked], mode="plan")
    nudges = [m for m in _user_messages(out) if "PLAN mode" in m]
    assert nudges and all(m.startswith(populate.SYSTEM_CHECK_PREFIX) for m in nudges)
    assert out.operation_message.startswith("Ready to submit")


def test_strip_keeps_text_that_is_only_an_apology():
    assert populate._strip_nudge_echo("You're absolutely right.") == "You're absolutely right."


def test_strip_handles_curly_apostrophes_and_paragraphs():
    text = "You’re correct, I have NOT submitted the job.\n\nHere is what I prepared:\n- reads: SRR123"
    assert populate._strip_nudge_echo(text) == "Here is what I prepared:\n- reads: SRR123"
