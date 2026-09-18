"""Tests for the plan / execute execution-mode gate.

Covers:
  - shared.tools.execute_tool refusing EXECUTE_ONLY_TOOLS in plan mode
  - describe_blocked_action / blocked_by_mode_result shape
  - BaseAgentState.record_execution -> blocked_actions
  - shared.agent_dispatch._build_config_kwargs mode mapping
  - dispatch response carries blocked_actions
  - orchestrate() emits EXECUTION_BLOCKED
  - format_execution_mode / build_populate_prompt sections
  - service populate loop: a blocked submit is not a failure
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure the bvbrc-agents repo root is importable
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import shared.agent_dispatch as agent_dispatch  # noqa: E402  -- sets up agent package paths
from shared.agent_utils import format_execution_mode  # noqa: E402
from shared.models import BaseAgentConfig, BaseAgentState, ToolCall  # noqa: E402
from shared.tools import (  # noqa: E402
    EXECUTE_ONLY_TOOLS,
    PLAN_MODE_BLOCKED_MESSAGE,
    blocked_by_mode_result,
    describe_blocked_action,
    execute_tool,
    execution_mode_of,
    is_blocked_result,
    normalize_execution_mode,
)


def _config(mode: str | None = None, **extra) -> BaseAgentConfig:
    kwargs = dict(llm_base_url="http://x", llm_api_key="k", llm_model="m")
    if mode is not None:
        kwargs["execution_mode"] = mode
    kwargs.update(extra)
    return BaseAgentConfig(**kwargs)


# -----------------------------------------------------------------------
# execute_tool gate
# -----------------------------------------------------------------------


class TestExecuteToolGate:
    def _table(self):
        return {
            "submit_gowe_job": AsyncMock(return_value={"submission_id": "sub_1"}),
            "create_group": AsyncMock(return_value={"path": "/u/home/Genome Groups/g"}),
            "search_data": AsyncMock(return_value={"items": []}),
        }

    async def test_gated_tools_are_the_two_side_effecting_ones(self):
        assert EXECUTE_ONLY_TOOLS == frozenset({"submit_gowe_job", "create_group"})

    @pytest.mark.parametrize("tool", sorted(EXECUTE_ONLY_TOOLS))
    async def test_plan_mode_blocks_gated_tool(self, tool):
        table = self._table()
        result = await execute_tool(tool, {"workflow_id": "wf", "inputs": {}}, table, config=_config("plan"))
        assert is_blocked_result(result)
        assert result["tool"] == tool
        assert result["execution_mode"] == "plan"
        assert "PLAN mode" in result["error"]
        table[tool].assert_not_awaited()

    @pytest.mark.parametrize("tool", sorted(EXECUTE_ONLY_TOOLS))
    async def test_execute_mode_allows_gated_tool(self, tool):
        table = self._table()
        result = await execute_tool(tool, {"workflow_id": "wf", "inputs": {}}, table, config=_config("execute"))
        assert not is_blocked_result(result)
        table[tool].assert_awaited_once()

    async def test_missing_config_is_plan(self):
        table = self._table()
        result = await execute_tool("submit_gowe_job", {"workflow_id": "wf", "inputs": {}}, table, config=None)
        assert is_blocked_result(result)
        table["submit_gowe_job"].assert_not_awaited()

    async def test_config_without_attribute_is_plan(self):
        table = self._table()
        result = await execute_tool(
            "create_group", {"group_name": "g"}, table, config=SimpleNamespace(bvbrc_auth_token=None)
        )
        assert is_blocked_result(result)

    async def test_default_config_is_plan(self):
        assert _config().execution_mode == "plan"
        assert execution_mode_of(_config()) == "plan"

    async def test_read_tools_never_blocked(self):
        table = self._table()
        result = await execute_tool("search_data", {"q": "x"}, table, config=_config("plan"))
        assert not is_blocked_result(result)
        table["search_data"].assert_awaited_once()

    async def test_blocked_result_strips_config_and_headers(self):
        table = self._table()
        result = await execute_tool(
            "submit_gowe_job",
            {"workflow_id": "wf", "inputs": {}, "config": _config(), "headers": {"Authorization": "t"}},
            table,
            config=_config("plan"),
        )
        assert "config" not in result["arguments"]
        assert "headers" not in result["arguments"]

    def test_normalize(self):
        assert normalize_execution_mode("execute") == "execute"
        for bad in ("plan", "EXECUTE", None, "", 1, "auto_all"):
            assert normalize_execution_mode(bad) == "plan"

    def test_message_names_tool(self):
        assert "submit_gowe_job" in PLAN_MODE_BLOCKED_MESSAGE.format(tool="submit_gowe_job")


class TestDescribeBlockedAction:
    def test_submit_with_output_path_in_inputs(self):
        s = describe_blocked_action(
            "submit_gowe_job",
            {"workflow_id": "GenomeAssembly", "inputs": {"output_path": "/u/home/SRR1_asm"}},
        )
        assert s == "GenomeAssembly → SRR1_asm"

    def test_submit_without_output_path(self):
        assert describe_blocked_action("submit_gowe_job", {"workflow_id": "wf"}) == "wf"

    def test_create_group(self):
        s = describe_blocked_action(
            "create_group", {"group_type": "genome_group", "group_name": "Kleb", "limit": 500}
        )
        assert s == "genome_group 'Kleb' (up to 500 ids)"

    def test_fallback(self):
        assert describe_blocked_action("other", {}) == "other"

    def test_result_shape(self):
        r = blocked_by_mode_result("create_group", {"group_name": "g", "config": 1})
        assert r["blocked_by_mode"] is True
        assert r["summary"].endswith("'g'")
        assert "config" not in r["arguments"]


# -----------------------------------------------------------------------
# State / result propagation
# -----------------------------------------------------------------------


class TestRecordExecution:
    def test_blocked_result_recorded(self):
        state = BaseAgentState()
        tc = ToolCall(id="1", name="submit_gowe_job", arguments={"workflow_id": "wf"})
        result = blocked_by_mode_result("submit_gowe_job", {"workflow_id": "wf"})
        state.record_execution(tc=tc, result=result, error=result["error"])
        assert len(state.tool_executions) == 1
        assert state.blocked_actions == [
            {"tool": "submit_gowe_job", "summary": "wf", "arguments": {"workflow_id": "wf"}}
        ]

    def test_ordinary_error_not_recorded(self):
        state = BaseAgentState()
        tc = ToolCall(id="1", name="submit_gowe_job", arguments={})
        state.record_execution(tc=tc, result={"error": "boom"}, error="boom")
        assert state.blocked_actions == []

    def test_success_not_recorded(self):
        state = BaseAgentState()
        tc = ToolCall(id="1", name="search_data", arguments={})
        state.record_execution(tc=tc, result={"items": []})
        assert state.blocked_actions == []


class TestDispatchWiring:
    def test_build_config_kwargs_maps_mode(self):
        assert agent_dispatch._build_config_kwargs(None, {"execution_mode": "execute"})["execution_mode"] == "execute"
        assert agent_dispatch._build_config_kwargs(None, {"execution_mode": "plan"})["execution_mode"] == "plan"
        assert agent_dispatch._build_config_kwargs(None, {})["execution_mode"] == "plan"
        assert agent_dispatch._build_config_kwargs(None, {"execution_mode": "auto_all"})["execution_mode"] == "plan"

    def test_auto_submit_preference_gone(self):
        kw = agent_dispatch._build_config_kwargs(None, {"auto_submit_preference": "auto_all"})
        assert "auto_submit_preference" not in kw

    def test_with_blocked_actions(self):
        result = SimpleNamespace(blocked_actions=[{"tool": "create_group", "summary": "g", "arguments": {}}])
        resp = agent_dispatch._with_blocked_actions({"answer": "x"}, result)
        assert resp["blocked_actions"] == result.blocked_actions

    def test_with_blocked_actions_empty(self):
        resp = agent_dispatch._with_blocked_actions({"answer": "x"}, SimpleNamespace(blocked_actions=[]))
        assert "blocked_actions" not in resp
        resp = agent_dispatch._with_blocked_actions({"answer": "x"}, SimpleNamespace())
        assert "blocked_actions" not in resp

    def test_every_agent_result_has_blocked_actions_field(self):
        from analysis_agent.models import AgentResult as A
        from data_agent.models import AgentResult as D
        from helpdesk_agent.models import AgentResult as H
        from planning_agent.models import AgentResult as P
        from service_agent.models import AgentResult as S
        from workspace_agent.models import AgentResult as W

        for cls in (A, D, H, P, S, W):
            assert "blocked_actions" in cls.model_fields, cls


# -----------------------------------------------------------------------
# Orchestrator request / executor / events
# -----------------------------------------------------------------------


class TestOrchestratorPlumbing:
    def test_request_default_and_validation(self):
        from pydantic import ValidationError

        from orchestrator.models import OrchestratorRequest

        assert OrchestratorRequest(query="x").execution_mode == "plan"
        assert OrchestratorRequest(query="x", execution_mode="execute").execution_mode == "execute"
        with pytest.raises(ValidationError):
            OrchestratorRequest(query="x", execution_mode="auto_all")
        assert "auto_submit_preference" not in OrchestratorRequest.model_fields

    def test_event_type_exists(self):
        from orchestrator.events.events import EventType

        assert EventType.EXECUTION_BLOCKED.value == "execution_blocked"

    def test_orchestrator_config_has_no_auto_submit(self):
        from orchestrator.config import OrchestratorConfig

        assert "auto_submit" not in OrchestratorConfig.model_fields

    async def test_orchestrate_emits_execution_blocked(self):
        from mcp.types import Tool as McpTool

        from orchestrator.config import AgentConfig, OrchestratorConfig
        from orchestrator.events.events import EventType
        from orchestrator.events.stream import collect_events
        from orchestrator.llm.client import LLMClient
        from orchestrator.models import OrchestratorRequest
        from orchestrator.orchestrate import orchestrate
        from orchestrator.registry.agent_handle import AgentHandle
        from orchestrator.registry.agent_registry import AgentRegistry

        config = OrchestratorConfig(
            agents={
                "service": AgentConfig(
                    name="Service Agent",
                    description="Plans service workflows.",
                    endpoint="http://localhost:8053",
                    capabilities=["workflow_planning"],
                ),
            },
            health_check_interval=0,
            routing_model="m",
            routing_base_url="http://x",
            routing_api_key="k",
        )
        registry = AgentRegistry(config)
        handle = AgentHandle("service", config.agents["service"])
        handle._healthy = True
        handle._tools = [McpTool(name="agent_chat", description="Chat", inputSchema={})]
        registry._agents["service"] = handle

        blocked = [{"tool": "submit_gowe_job", "summary": "GenomeAssembly → SRR1", "arguments": {}}]
        handle.call_tool = AsyncMock(
            return_value={
                "answer": "Ready to submit.",
                "status": "completed",
                "sources": [],
                "iterations_used": 3,
                "elapsed_seconds": 1.0,
                "tool_trace": [],
                "blocked_actions": blocked,
            }
        )
        llm = MagicMock(spec=LLMClient)
        llm.complete = AsyncMock(return_value="unused")

        request = OrchestratorRequest(query="assemble SRR1", target_agent="service")
        events = await collect_events(orchestrate(request, registry, llm))

        blocked_events = [e for e in events if e.type == EventType.EXECUTION_BLOCKED]
        assert len(blocked_events) == 1
        data = blocked_events[0].data
        assert data["agent"] == "service"
        assert data["execution_mode"] == "plan"
        assert data["blocked_actions"] == blocked
        assert data["original_query"] == "assemble SRR1"
        # The turn still completes normally
        assert any(e.type == EventType.ORCHESTRATOR_DONE for e in events)
        # The executor forwarded the request's mode to the agent
        args = handle.call_tool.await_args.args[1]
        assert args["context"]["execution_mode"] == "plan"


# -----------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------


class TestPrompts:
    def test_format_execution_mode_from_config(self):
        assert "PLAN mode" in format_execution_mode(_config("plan"))
        assert "EXECUTE mode" in format_execution_mode(_config("execute"))

    def test_format_execution_mode_from_context(self):
        assert "PLAN mode" in format_execution_mode({})
        assert "PLAN mode" in format_execution_mode(None)
        assert "EXECUTE mode" in format_execution_mode({"execution_mode": "execute"})

    def test_format_execution_mode_names_gated_tools(self):
        text = format_execution_mode(_config("plan"))
        assert "submit_gowe_job" in text and "create_group" in text

    def test_populate_prompt_sections(self):
        from service_agent.prompts.populate import build_populate_prompt

        plan = build_populate_prompt(execution_mode="plan")
        assert "This session is in PLAN mode" in plan
        assert "WHEN TO SKIP CONFIRMATION" not in plan
        assert "blocked_by_mode" in plan
        execute = build_populate_prompt(execution_mode="execute")
        assert "This session is in EXECUTE mode" in execute
        assert "This session is in PLAN mode" not in execute
        assert "This session is in PLAN mode" in build_populate_prompt()


# -----------------------------------------------------------------------
# Service populate loop
# -----------------------------------------------------------------------


def _llm_response(content: str | None = None, tool_calls: list[tuple[str, str, dict]] | None = None):
    """Build a minimal OpenAI-shaped ChatCompletion response."""
    tcs = None
    if tool_calls:
        tcs = [
            SimpleNamespace(id=tid, function=SimpleNamespace(name=name, arguments=json.dumps(args)))
            for tid, name, args in tool_calls
        ]
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tcs))])


class TestPopulateLoopPlanMode:
    async def _run(self, mode: str, llm_side_effect: list, dispatch: dict):
        from service_agent.models import AgentConfig, AgentState
        from service_agent.phases import populate as populate_mod

        config = AgentConfig(
            llm_base_url="http://x", llm_api_key="k", llm_model="m", execution_mode=mode, max_iterations=10
        )
        state = AgentState(query="assemble SRR1")
        chat = AsyncMock(side_effect=llm_side_effect)
        with patch.object(populate_mod, "chat_completion", chat), patch.object(
            populate_mod, "create_client", lambda cfg: MagicMock()
        ), patch.object(populate_mod, "TOOL_DISPATCH", dispatch):
            state = await populate_mod.populate_and_submit("assemble SRR1", config, state)
        return state, chat

    async def test_blocked_submit_is_not_a_failure(self):
        submit = AsyncMock(return_value={"submission_id": "sub_1", "workflow_id": "wf"})
        get_inputs = AsyncMock(return_value={"inputs": []})
        state, chat = await self._run(
            "plan",
            [
                _llm_response(tool_calls=[("1", "get_workflow_inputs", {"workflow_id": "wf"})]),
                _llm_response(
                    tool_calls=[("2", "submit_gowe_job", {"workflow_id": "wf", "inputs": {"output_path": "/u/home/A"}})]
                ),
                _llm_response(content="Ready to submit: GenomeAssembly on SRR1. Switch to Execute mode."),
            ],
            {"get_workflow_inputs": get_inputs, "submit_gowe_job": submit},
        )
        submit.assert_not_awaited()
        assert state.status == "completed"
        assert state.submission_ids == []
        assert state.blocked_actions == [
            {"tool": "submit_gowe_job", "summary": "wf → A", "arguments": {"workflow_id": "wf", "inputs": {"output_path": "/u/home/A"}}}
        ]
        assert "Ready to submit" in state.operation_message
        # The call after the block is forced text-only
        assert chat.await_args_list[-1].kwargs["tool_choice"] == "none"
        # The refusal reached the LLM as a tool result
        tool_msgs = [m for m in state.messages if m.get("role") == "tool"]
        assert any("blocked_by_mode" in m["content"] for m in tool_msgs)

    async def test_success_claim_after_block_is_nudged_then_overwritten(self):
        submit = AsyncMock()
        state, chat = await self._run(
            "plan",
            [
                _llm_response(tool_calls=[("2", "submit_gowe_job", {"workflow_id": "wf", "inputs": {}})]),
                _llm_response(content="Your job has been submitted successfully!"),
                _llm_response(content="The job was submitted and is now running."),
            ],
            {"submit_gowe_job": submit},
        )
        submit.assert_not_awaited()
        assert state.status == "completed"
        assert "submitted successfully" not in state.operation_message.lower()
        assert "Ready to submit" in state.operation_message
        assert "Execute mode" in state.operation_message
        # One nudge was sent as a user message
        assert any("PLAN mode" in (m.get("content") or "") for m in state.messages if m.get("role") == "user")

    async def test_execute_mode_submits(self):
        submit = AsyncMock(return_value={"submission_id": "sub_1", "workflow_id": "wf", "output_path": "/u/home/A"})
        state, chat = await self._run(
            "execute",
            [
                _llm_response(tool_calls=[("2", "submit_gowe_job", {"workflow_id": "wf", "inputs": {}})]),
                _llm_response(content="Submitted. Results will be in /u/home/A."),
            ],
            {"submit_gowe_job": submit},
        )
        submit.assert_awaited_once()
        assert state.status == "completed"
        assert state.submission_ids == ["sub_1"]
        assert state.blocked_actions == []

    async def test_real_failure_still_hits_circuit_breaker(self):
        submit = AsyncMock(return_value={"error": "GoWe down"})
        state, chat = await self._run(
            "execute",
            [
                _llm_response(tool_calls=[("2", "submit_gowe_job", {"workflow_id": "wf", "inputs": {}})]),
                _llm_response(tool_calls=[("3", "submit_gowe_job", {"workflow_id": "wf", "inputs": {"x": 1}})]),
                _llm_response(content="Submission failed: GoWe down."),
            ],
            {"submit_gowe_job": submit},
        )
        assert submit.await_count == 2
        assert state.status == "error"
