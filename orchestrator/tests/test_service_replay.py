""""Submit this job" replays the card's blocked_actions verbatim (no LLM).

Before this, the execute-once turn re-ran the populate loop and the model
rebuilt the inputs from a truncated conversation, submitting different
parameters than the card showed (2026-10-09: SILVA / no_host became
Standard / homo_sapiens).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import shared.agent_dispatch  # noqa: E402, F401  -- sets up agent package paths
from service_agent import agent as service_agent  # noqa: E402
from service_agent.models import AgentConfig  # noqa: E402
from service_agent.phases import replay as replay_mod  # noqa: E402
from shared.tools import blocked_by_mode_result  # noqa: E402

QUERY = "Execute mode is on for this message. Submit the job exactly as you prepared it above."

TC_ARGS = {
    "workflow_id": "wf_tc",
    "inputs": {
        "srr_ids": ["ERR7916233"],
        "database": "SILVA",
        "host_genome": "no_host",
        "analysis_type": "pathogen",
        "output_path": "tc_ERR7916233",
    },
}
TC_ACTION = {"tool": "submit_gowe_job", "arguments": TC_ARGS, "summary": "Taxonomic Classification → tc_ERR7916233"}
GROUP_ACTION = {
    "tool": "create_group",
    "arguments": {"group_name": "strep_resist", "group_type": "genome", "collection": "genome", "query": "genome_id:(1.1 OR 2.2)"},
    "summary": "strep_resist (genome)",
}


def _cfg(mode: str = "execute") -> AgentConfig:
    return AgentConfig(
        llm_base_url="http://llm", llm_api_key="k", llm_model="m",
        bvbrc_auth_token="un=alice@patricbrc.org|tok", execution_mode=mode,
    )


def _submit_ok(n: int = 1):
    return {"workflow_id": "wf_tc", "submission_id": f"sub_{n}", "status": "PENDING",
            "output_path": "/alice@patricbrc.org/home/chats/s1/tc_ERR7916233"}


async def _run(actions, results, mode="execute", classifier=None):
    """Run the service agent with execute_tool mocked to return *results* in order."""
    tool = AsyncMock(side_effect=list(results))
    classify = classifier or AsyncMock(side_effect=AssertionError("classifier must not run"))
    flow = AsyncMock(side_effect=AssertionError("populate loop must not run"))
    with patch.object(replay_mod, "execute_tool", tool), \
         patch.object(service_agent, "classify_intent", classify), \
         patch.object(service_agent, "_run_gowe_flow", flow):
        result = await service_agent.run_agent(
            QUERY, config=_cfg(mode),
            context={"execution_mode": mode, "replay_actions": actions},
        )
    return result, tool


async def test_replays_exact_arguments_without_llm():
    result, tool = await _run([TC_ACTION], [_submit_ok()])
    tool.assert_awaited_once()
    kwargs = tool.await_args.kwargs
    assert kwargs["tool_name"] == "submit_gowe_job"
    assert kwargs["arguments"] == TC_ARGS
    assert result.status == "completed"
    assert result.submission_ids == ["sub_1"]
    assert "Submitted **Taxonomic Classification → tc_ERR7916233**" in result.operation_message
    assert "/chats/s1/tc_ERR7916233" in result.operation_message
    assert "sub_1" not in result.operation_message


async def test_two_actions_run_in_order():
    group_ok = {"name": "strep_resist", "path": "/alice@patricbrc.org/home/Genome Groups/strep_resist",
                "count": 2, "action": "created", "added": 2}
    result, tool = await _run([GROUP_ACTION, TC_ACTION], [group_ok, _submit_ok()])
    assert [c.kwargs["tool_name"] for c in tool.await_args_list] == ["create_group", "submit_gowe_job"]
    assert result.status == "completed"
    assert "Created group **strep_resist** (2 items)" in result.operation_message
    assert result.submission_ids == ["sub_1"]


async def test_first_failure_stops_the_rest():
    result, tool = await _run([TC_ACTION, GROUP_ACTION], [{"error": "GoWe returned 400: bad srr_ids"}])
    tool.assert_awaited_once()
    assert result.status == "error"
    assert "Could not run **Taxonomic Classification" in result.error_message
    assert "bad srr_ids" in result.error_message
    assert "Not attempted after that: **strep_resist (genome)**" in result.error_message
    assert result.submission_ids == []


async def test_partial_success_is_completed_and_names_the_failure():
    result, _ = await _run([GROUP_ACTION, TC_ACTION],
                           [{"name": "strep_resist", "count": 2, "action": "created"}, {"error": "boom"}])
    assert result.status == "completed"
    assert "Created group" in result.operation_message
    assert "Could not run **Taxonomic Classification" in result.operation_message


async def test_plan_mode_is_still_gated():
    result, _ = await _run([TC_ACTION], [blocked_by_mode_result("submit_gowe_job", TC_ARGS)], mode="plan")
    assert result.status == "error"
    assert "Plan mode" in result.error_message
    assert result.submission_ids == []


async def test_gate_runs_for_real_in_plan_mode():
    """Without mocking execute_tool, the execution-mode gate refuses the call."""
    submit = AsyncMock(side_effect=AssertionError("submit_gowe_job must not run in plan mode"))
    with patch.dict(replay_mod.TOOL_DISPATCH, {"submit_gowe_job": submit}), \
         patch.object(service_agent, "classify_intent", AsyncMock(side_effect=AssertionError)):
        result = await service_agent.run_agent(
            QUERY, config=_cfg("plan"),
            context={"execution_mode": "plan", "replay_actions": [TC_ACTION]},
        )
    submit.assert_not_awaited()
    assert result.status == "error"


async def test_config_and_headers_keys_are_dropped():
    sneaky = {"tool": "submit_gowe_job", "summary": "x",
              "arguments": {**TC_ARGS, "config": {"bvbrc_auth_token": "someone-else"}, "headers": {"Authorization": "x"}}}
    _, tool = await _run([sneaky], [_submit_ok()])
    sent = tool.await_args.kwargs["arguments"]
    assert "config" not in sent and "headers" not in sent
    assert tool.await_args.kwargs["config"].bvbrc_auth_token == "un=alice@patricbrc.org|tok"


async def test_non_replayable_tool_is_refused():
    result, tool = await _run([{"tool": "search_data", "arguments": {"collection": "genome"}}], [])
    tool.assert_not_awaited()
    assert result.status == "error"
    assert "Please ask me to prepare it again" in result.error_message


async def test_empty_or_oversized_list_is_refused():
    for actions in ([], [TC_ACTION] * 11, "not a list"):
        result, tool = await _run(actions, [])
        tool.assert_not_awaited()
        assert result.status == "error"


async def test_without_replay_actions_classifier_runs():
    flow = AsyncMock(return_value="ran-populate")
    from service_agent.models import Intent
    with patch.object(service_agent, "classify_intent",
                      AsyncMock(return_value=Intent(action="submit", workflow_id=None))), \
         patch.object(service_agent, "_run_gowe_flow", flow):
        result = await service_agent.run_agent(QUERY, config=_cfg(), context={"execution_mode": "execute"})
    assert result == "ran-populate"


async def test_replay_actions_not_left_in_context():
    """The actions must not ride along into any prompt dump."""
    captured = {}

    async def fake_replay(actions, cfg, state):
        captured["ctx"] = state.context
        state.status = "completed"
        state.operation_message = "ok"
        return state

    with patch.object(service_agent, "replay_blocked_actions", fake_replay):
        await service_agent.run_agent(QUERY, config=_cfg(),
                                      context={"execution_mode": "execute", "replay_actions": [TC_ACTION]})
    assert "replay_actions" not in captured["ctx"]


async def test_executor_forwards_replay_actions_in_context():
    from types import SimpleNamespace

    from orchestrator.events.stream import collect_events
    from orchestrator.executor.agent_executor import execute_agent_step
    from orchestrator.models import OrchestratorRequest
    from orchestrator.router.models import Step

    seen = {}

    class _Agent:
        key = "service"
        config = SimpleNamespace(chat_tool="agent_chat", chat_tool_params={},
                                 mcp_server_name=None, timeout_seconds=300)
        tool_names = ["agent_chat"]

        async def call_tool(self, name, arguments, progress_handler=None):
            seen.update(arguments)
            return {"answer": "ok", "status": "completed"}

    request = OrchestratorRequest(query=QUERY, execution_mode="execute",
                                  target_agent="service", replay_actions=[TC_ACTION])
    await collect_events(execute_agent_step(Step(agent_key="service", task=QUERY), _Agent(), request))
    context = seen.get("context") or {}
    assert context.get("replay_actions") == [TC_ACTION]
    assert context.get("execution_mode") == "execute"


async def test_repeat_click_reports_already_submitted():
    dup = {"error": "This exact job ... NOT submitted again. Tell the user ...",
           "already_submitted": True, "duplicate_of": "sub_1", "workflow_id": "wf_tc"}
    result, _ = await _run([TC_ACTION], [dup])
    assert result.status == "completed"
    assert "was already submitted from this chat" in result.operation_message
    assert "Tell the user" not in result.operation_message
    assert result.submission_ids == []
