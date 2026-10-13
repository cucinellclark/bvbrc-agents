"""F1 agent handoff: a call that belongs to another agent hands the turn to it.

Oct 13 verification, case 9: "Is this protein in BV-BRC? MKKLL…" went to the
data agent.  It prepared a BLAST job, got the allowlist refusal (only the
service agent submits), then called create_plan, which nothing turned into a
card, and told the user to "switch to Execute mode", which could never work.
Now submit_gowe_job -> service, create_group -> data, create_plan -> planning,
one hop per turn.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import shared.agent_dispatch as agent_dispatch  # noqa: E402
from shared.agent_utils import format_handoff  # noqa: E402
from shared.models import HANDOFF, BaseAgentState, ToolCall  # noqa: E402
from shared.tools import execute_tool, is_handoff_result, is_wrong_agent_result  # noqa: E402

from orchestrator.events.events import EventType  # noqa: E402
from orchestrator.executor import executor as plan_executor  # noqa: E402
from orchestrator.models import OrchestratorRequest  # noqa: E402
from orchestrator.router.models import Step  # noqa: E402

BLAST = {"workflow_id": "BLAST", "inputs": {"input_fasta_data": ">q\nMKKLL", "output_path": "b"}}


def _cfg(agent, mode="plan", handoff_from=None):
    return SimpleNamespace(agent_key=agent, execution_mode=mode, handoff_from=handoff_from)


def _table():
    return {name: AsyncMock(return_value={"ok": True})
            for name in ("submit_gowe_job", "create_group", "create_plan")}


# --- Trigger: execute_tool -------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("agent,tool,target", [
    ("data", "submit_gowe_job", "service"),
    ("helpdesk", "submit_gowe_job", "service"),
    ("planning", "submit_gowe_job", "service"),
    ("helpdesk", "create_group", "data"),
    ("workspace", "create_group", "data"),
    ("data", "create_plan", "planning"),
    ("service", "create_plan", "planning"),
])
async def test_call_owned_by_another_agent_hands_off(agent, tool, target):
    table = _table()
    result = await execute_tool(tool, dict(BLAST), table, config=_cfg(agent))
    assert is_handoff_result(result) and result["handoff_to"] == target
    assert result["arguments"]["workflow_id"] == "BLAST"
    assert "error" not in result                 # not a failure: nothing ran
    table[tool].assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("agent,tool", [
    ("data", "create_group"), ("planning", "create_plan"),
])
async def test_owner_runs_its_own_tool(agent, tool):
    table = _table()
    result = await execute_tool(tool, {"x": 1}, table, config=_cfg(agent, mode="execute"))
    assert not is_handoff_result(result)
    table[tool].assert_awaited_once()


@pytest.mark.asyncio
async def test_service_submit_still_goes_through_the_plan_gate():
    result = await execute_tool("submit_gowe_job", dict(BLAST), _table(), config=_cfg("service"))
    assert result.get("blocked_by_mode") and not is_handoff_result(result)


@pytest.mark.asyncio
async def test_recipient_cannot_hand_off_again():
    """One hop: the agent that received a handoff gets the plain refusal."""
    result = await execute_tool("submit_gowe_job", dict(BLAST), _table(),
                                config=_cfg("data", handoff_from="helpdesk"))
    assert is_wrong_agent_result(result) and not is_handoff_result(result)


@pytest.mark.asyncio
async def test_recipient_create_plan_just_runs():
    table = _table()
    result = await execute_tool("create_plan", {"x": 1}, table, config=_cfg("service", handoff_from="data"))
    assert not is_handoff_result(result)
    table["create_plan"].assert_awaited_once()


@pytest.mark.asyncio
async def test_config_without_agent_key_is_unrestricted():
    table = _table()
    result = await execute_tool("create_plan", {"x": 1}, table, config=SimpleNamespace(execution_mode="plan"))
    assert not is_handoff_result(result)


# --- Slot: record_execution + dispatch_agent -------------------------------

def _record_handoff(state):
    tc = ToolCall(id="c1", name="submit_gowe_job", arguments=dict(BLAST))
    result = {"handoff_to": "service", "tool": "submit_gowe_job", "arguments": dict(BLAST)}
    state.record_execution(tc=tc, result=result)


def test_record_execution_fills_the_slot_once():
    slot = {}
    token = HANDOFF.set(slot)
    try:
        state = BaseAgentState(query="q")
        _record_handoff(state)
        assert slot == {"to": "service", "tool": "submit_gowe_job", "arguments": BLAST}
        state.record_execution(
            tc=ToolCall(id="c2", name="create_group", arguments={}),
            result={"handoff_to": "data", "arguments": {}},
        )
        assert slot["to"] == "service"
    finally:
        HANDOFF.reset(token)


@pytest.mark.asyncio
async def test_dispatch_returns_status_handoff():
    async def fake_data_runner(query, config_kwargs, ctx, progress_callback):
        assert config_kwargs["agent_key"] == "data"
        _record_handoff(BaseAgentState(query=query))
        return {"answer": "Switch to Execute mode and I'll run BLAST.", "status": "completed",
                "tool_trace": [{"tool": "get_workflow_inputs", "arguments": {"workflow_id": "BLAST"}},
                               {"tool": "submit_gowe_job", "arguments": BLAST}],
                "elapsed_seconds": 12.0}

    with patch.dict(agent_dispatch._AGENT_RUNNERS, {"data": fake_data_runner}):
        out = await agent_dispatch.dispatch_agent("data", "Is this protein in BV-BRC?", {}, "tok")
    assert out["status"] == "handoff" and out["answer"] == ""
    h = out["handoff"]
    assert (h["from"], h["to"], h["tool"]) == ("data", "service", "submit_gowe_job")
    assert [s["tool"] for s in h["findings"]] == ["get_workflow_inputs"]
    assert HANDOFF.get() is None


def test_recipient_config_gets_handoff_from():
    kwargs = agent_dispatch._build_config_kwargs("tok", {"handoff": {"from": "data", "to": "service"}})
    assert kwargs["handoff_from"] == "data"
    assert "handoff_from" not in agent_dispatch._build_config_kwargs("tok", {})


# --- Shared loop stops at once ---------------------------------------------

@pytest.mark.asyncio
async def test_shared_loop_stops_without_a_reply():
    from shared.agent_loop import run_agent_loop

    call = SimpleNamespace(id="c1", type="function",
                           function=SimpleNamespace(name="submit_gowe_job", arguments=json.dumps(BLAST)))
    msg = SimpleNamespace(content="", tool_calls=[call], reasoning=None)
    llm = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")]))
    cfg = SimpleNamespace(agent_key="data", execution_mode="plan", handoff_from=None,
                          bvbrc_auth_token=None, max_iterations=5, tool_timeout_seconds=30,
                          max_context_tokens=100000)
    state = BaseAgentState(query="Is this protein in BV-BRC?")
    slot = {}
    token = HANDOFF.set(slot)
    try:
        await run_agent_loop(state=state, config=cfg, client=object(), system_prompt="sys",
                             chat_completion_fn=llm)
    finally:
        HANDOFF.reset(token)
    assert llm.await_count == 1
    assert state.status == "handoff" and state.final_answer == ""
    assert slot["to"] == "service"


# --- Executor runs the receiving agent -------------------------------------

class _Agent:
    def __init__(self, key, response):
        self.key = key
        self.config = SimpleNamespace(chat_tool_params=None, mcp_server_name=None, timeout_seconds=300)
        self.tool_names = ["agent_chat"]
        self.is_healthy = True
        self.response = response
        self.calls = []

    async def call_tool(self, name, arguments, progress_handler=None):
        self.calls.append(arguments)
        return dict(self.response)


class _Registry:
    def __init__(self, *agents):
        self.agents = {a.key: a for a in agents}
        self.agent_keys = list(self.agents)

    def get(self, key):
        return self.agents[key]


HANDOFF_FROM_DATA = {
    "answer": "", "status": "handoff",
    "handoff": {"from": "data", "to": "service", "tool": "submit_gowe_job",
                "arguments": BLAST, "findings": []},
    "tool_trace": [],
}
BLOCKED = [{"tool": "submit_gowe_job", "summary": "BLAST → b", "arguments": BLAST}]


async def _events(registry, query="Is this protein in BV-BRC?"):
    req = OrchestratorRequest(query=query, auth_token="tok")
    return [e async for e in plan_executor._run_single_step(Step(agent_key="data", task=query), 0, registry, req)]


@pytest.mark.asyncio
async def test_executor_runs_the_target_with_the_note():
    data = _Agent("data", HANDOFF_FROM_DATA)
    service = _Agent("service", {"answer": "Ready to submit: BLAST", "status": "completed",
                                 "blocked_actions": BLOCKED, "tool_trace": []})
    events = await _events(_Registry(data, service))

    results = [e for e in events if e.type == EventType.AGENT_RESULT]
    assert len(results) == 1                       # the data agent's empty result is dropped
    ui = results[0].data["result_for_ui"]
    assert ui["agent"] == "service" and ui["blocked_actions"] == BLOCKED
    assert ui["handoff"] == {"from": "data", "to": "service", "trigger": "submit_gowe_job"}

    sent = service.calls[0]
    assert sent["query"] == "Is this protein in BV-BRC?"
    assert sent["context"]["handoff"]["arguments"] == BLAST
    progress = [e.data.get("message") for e in events if e.type == EventType.AGENT_PROGRESS]
    assert "Passing this to the job-submission agent…" in progress


@pytest.mark.asyncio
async def test_missing_target_is_a_plain_error():
    events = await _events(_Registry(_Agent("data", HANDOFF_FROM_DATA)))
    errors = [e for e in events if e.type == EventType.ORCHESTRATOR_ERROR]
    assert errors and "can't run" in errors[0].data["error"]
    assert not [e for e in events if e.type == EventType.AGENT_RESULT]


@pytest.mark.asyncio
async def test_target_error_carries_handoff_metadata():
    data = _Agent("data", HANDOFF_FROM_DATA)
    service = _Agent("service", {"answer": "BLAST submission failed.", "status": "error", "tool_trace": []})
    events = await _events(_Registry(data, service))
    errors = [e for e in events if e.type == EventType.ORCHESTRATOR_ERROR]
    assert errors[0].data["handoff"]["to"] == "service"


# --- Prompt section --------------------------------------------------------

def test_format_handoff_section():
    text = format_handoff({"handoff": {
        "from": "data", "to": "service", "tool": "submit_gowe_job", "arguments": BLAST,
        "findings": [{"tool": "get_workflow_inputs", "arguments": {"workflow_id": "BLAST"},
                      "result_summary": {"count": 14}}],
    }})
    assert text.startswith("=== HANDOFF ===")
    assert "get_workflow_inputs" in text and '"workflow_id": "BLAST"' in text
    assert "draft" in text
    assert format_handoff({}) == "" and format_handoff(None) == ""
