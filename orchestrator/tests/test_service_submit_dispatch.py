"""Service agent dispatch: ``submit`` with no workflow id (release plan risk R5), ``status``.

A job prepared through GoWe has no saved workflow id: its inputs exist only in
the conversation. The "Submit this job" button sends an execute-once follow-up,
which a classifier that returns valid JSON (Qwen3.8) labels ``action=submit,
workflow_id=None``. That must run the populate loop, not ask for an id.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import shared.agent_dispatch  # noqa: E402, F401  -- sets up agent package paths
from service_agent import agent as service_agent  # noqa: E402
from service_agent.models import AgentConfig, Intent  # noqa: E402

CFG = AgentConfig(llm_base_url="http://llm", llm_api_key="k", llm_model="m")
QUERY = "Execute mode is on for this message. Submit the job exactly as you prepared it above."


async def _run(intent: Intent):
    flow = AsyncMock(return_value="ran-populate")
    with patch.object(service_agent, "classify_intent", AsyncMock(return_value=intent)), \
         patch.object(service_agent, "_run_gowe_flow", flow):
        result = await service_agent.run_agent(QUERY, config=CFG, context={"execution_mode": "execute"})
    return result, flow


async def test_submit_without_workflow_id_runs_populate_loop():
    result, flow = await _run(Intent(action="submit", workflow_id=None, confidence=0.98))
    assert result == "ran-populate"
    flow.assert_awaited_once()
    assert flow.await_args.args[0] == QUERY


@pytest.mark.parametrize("workflow_id", [None, "wf_abc"])
async def test_status_runs_populate_loop(workflow_id):
    # The old direct GoWe lookup used a wf_ workflow id as a submission id
    # (always "not found") and, with no id, asked for one users never see.
    result, flow = await _run(Intent(action="status", workflow_id=workflow_id))
    assert result == "ran-populate"
    flow.assert_awaited_once()


async def test_cancel_without_workflow_id_still_asks():
    result, flow = await _run(Intent(action="cancel", workflow_id=None))
    flow.assert_not_awaited()
    assert result.status == "needs_input"


def test_populate_prompt_has_job_status_section():
    from service_agent.prompts.populate import build_populate_prompt
    prompt = build_populate_prompt()
    assert "== JOB STATUS ==" in prompt and "list_jobs" in prompt


async def test_submit_with_workflow_id_uses_handler():
    handler = AsyncMock(return_value="handled")
    with patch.object(service_agent, "classify_intent",
                      AsyncMock(return_value=Intent(action="submit", workflow_id="wf_abc"))), \
         patch.object(service_agent, "handle_submit", handler), \
         patch.object(service_agent, "_run_gowe_flow", AsyncMock()) as flow:
        result = await service_agent.run_agent("submit wf_abc", config=CFG, context={})
    assert result == "handled"
    flow.assert_not_awaited()
