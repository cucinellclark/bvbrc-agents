"""Jobs skipped by the one-job-per-message cap are never reported as submitted.

semenleyn 1f2987c1: "Both annotation jobs have been submitted" with one
submission id.  cmao: "Proteome Comparison SUBMITTED" when it was skipped.
mshukla f2757228: five skipped jobs called "flagged as duplicates".
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

ANNOT = {"workflow_id": "GenomeAnnotation", "inputs": {"contigs": "/a/bin1.fa", "output_path": "annot_bin1"}}
ANNOT2 = {"workflow_id": "GenomeAnnotation", "inputs": {"contigs": "/a/bin2.fa", "output_path": "annot_bin2"}}


def _reply(content="", tool_calls=None):
    calls = [SimpleNamespace(id=f"c{i}", type="function",
                             function=SimpleNamespace(name=n, arguments=json.dumps(a)))
             for i, (n, a) in enumerate(tool_calls or [])]
    msg = SimpleNamespace(content=content, tool_calls=calls or None, reasoning=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])


def _ok(n):
    return {"workflow_id": "wf_annot", "submission_id": f"sub_{n}", "status": "PENDING",
            "output_path": f"/u/home/chats/s1/annot_bin{n}"}


async def _run(replies, tool_results, batch=False):
    cfg = AgentConfig(llm_base_url="http://llm", llm_api_key="k", llm_model="m", execution_mode="execute")
    ctx = {"execution_mode": "execute", **({"batch_mode": True} if batch else {})}
    state = AgentState(query="annotate both bins", context=ctx)
    tool = AsyncMock(side_effect=list(tool_results))
    with patch.object(populate, "create_client", lambda c: object()), \
         patch.object(populate, "chat_completion", AsyncMock(side_effect=list(replies))), \
         patch.object(populate, "execute_tool", tool):
        out = await populate.populate_and_submit("annotate both bins", cfg, state)
    return out, tool


@pytest.mark.asyncio
async def test_second_job_in_one_response_is_named_as_not_submitted():
    replies = [
        _reply(tool_calls=[("submit_gowe_job", ANNOT), ("submit_gowe_job", ANNOT2)]),
        _reply("Both annotation jobs have been submitted successfully."),
    ]
    out, tool = await _run(replies, [_ok(1)])
    assert tool.await_count == 1
    assert out.submission_ids == ["sub_1"]
    assert "**Not submitted (one job per message):** GenomeAnnotation → annot_bin2" in out.operation_message
    skipped_result = [m for m in out.messages if m["role"] == "tool" and "_skipped" in m["content"]]
    assert skipped_result and "NOT submitted" in skipped_result[0]["content"]


@pytest.mark.asyncio
async def test_identical_repeat_is_not_reported_as_a_second_job():
    replies = [
        _reply(tool_calls=[("submit_gowe_job", ANNOT), ("submit_gowe_job", ANNOT)]),
        _reply("Your annotation job has been submitted."),
    ]
    out, tool = await _run(replies, [_ok(1)])
    assert tool.await_count == 1
    assert "Not submitted (one job per message)" not in out.operation_message


@pytest.mark.asyncio
async def test_batch_mode_submits_both_and_adds_no_note():
    replies = [
        _reply(tool_calls=[("submit_gowe_job", ANNOT), ("submit_gowe_job", ANNOT2)]),
        _reply("Both annotation jobs have been submitted."),
    ]
    out, tool = await _run(replies, [_ok(1), _ok(2)], batch=True)
    assert tool.await_count == 2
    assert out.submission_ids == ["sub_1", "sub_2"]
    assert "Not submitted" not in out.operation_message


@pytest.mark.asyncio
async def test_single_job_has_no_note():
    replies = [_reply(tool_calls=[("submit_gowe_job", ANNOT)]), _reply("Submitted.")]
    out, _ = await _run(replies, [_ok(1)])
    assert out.operation_message == "Submitted."
