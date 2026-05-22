"""Tests for the executor (plan execution and agent step execution)."""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from mcp.types import Tool as McpTool, CallToolResult, TextContent

from orchestrator.config import AgentConfig, OrchestratorConfig
from orchestrator.events.events import EventType
from orchestrator.events.stream import collect_events
from orchestrator.executor.agent_executor import execute_agent_step, _parse_mcp_result
from orchestrator.executor.executor import execute_plan
from orchestrator.models import OrchestratorRequest
from orchestrator.registry.agent_handle import AgentHandle
from orchestrator.registry.agent_registry import AgentRegistry
from orchestrator.router.models import Plan, Step


# --- Fixtures ---


def _make_agent_handle(
    key: str = "data",
    tool_names: list[str] | None = None,
    healthy: bool = True,
    chat_tool: str = "agent_chat",
) -> AgentHandle:
    """Create a mock AgentHandle."""
    config = AgentConfig(
        name=f"{key.title()} Agent",
        description=f"Test {key} agent",
        endpoint=f"http://localhost:9999",
        capabilities=[f"{key}_capability"],
        chat_tool=chat_tool,
    )
    handle = AgentHandle(key, config)
    handle._healthy = healthy

    if tool_names is None:
        tool_names = ["agent_chat", "some_tool"]

    handle._tools = [
        McpTool(name=name, description=f"Tool: {name}", inputSchema={})
        for name in tool_names
    ]

    return handle


def _make_mcp_result(
    answer: str = "Test answer",
    status: str = "completed",
    sources: list[str] | None = None,
    is_error: bool = False,
) -> CallToolResult:
    """Create a mock MCP CallToolResult."""
    data = {
        "answer": answer,
        "status": status,
        "sources": sources or [],
        "iterations_used": 2,
        "elapsed_seconds": 1.5,
        "tool_trace": [],
    }
    result = MagicMock(spec=CallToolResult)
    result.content = [TextContent(type="text", text=json.dumps(data))]
    result.isError = is_error
    return result


def _make_request(query: str = "test query") -> OrchestratorRequest:
    return OrchestratorRequest(query=query)


# --- Tests: _parse_mcp_result ---


class TestParseMCPResult:
    def test_parse_json_result(self):
        result = _make_mcp_result("Found 10 genomes", "completed")
        parsed = _parse_mcp_result(result)
        assert parsed["answer"] == "Found 10 genomes"
        assert parsed["status"] == "completed"

    def test_parse_plain_text_result(self):
        result = MagicMock(spec=CallToolResult)
        result.content = [TextContent(type="text", text="plain text answer")]
        parsed = _parse_mcp_result(result)
        assert parsed["answer"] == "plain text answer"
        assert parsed["status"] == "completed"

    def test_parse_empty_result(self):
        result = MagicMock(spec=CallToolResult)
        result.content = []
        parsed = _parse_mcp_result(result)
        assert "empty" in parsed["answer"].lower()
        assert parsed["status"] == "error"

    def test_parse_none_content(self):
        result = MagicMock(spec=CallToolResult)
        result.content = None
        parsed = _parse_mcp_result(result)
        assert parsed["status"] == "error"


# --- Tests: execute_agent_step ---


class TestExecuteAgentStep:
    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """Test successful agent step execution."""
        agent = _make_agent_handle()
        agent.call_tool = AsyncMock(
            return_value=_make_mcp_result("Found 10 genomes", "completed")
        )

        step = Step(agent_key="data", task="find E. coli genomes")
        request = _make_request("find E. coli genomes")

        events = await collect_events(
            execute_agent_step(step, agent, request, step_index=0)
        )

        event_types = [e.type for e in events]
        assert EventType.AGENT_START in event_types
        assert EventType.AGENT_TOOL_CALL in event_types
        assert EventType.AGENT_TOOL_RESULT in event_types
        assert EventType.AGENT_RESULT in event_types

        # Check the result event
        result_event = next(e for e in events if e.type == EventType.AGENT_RESULT)
        assert result_event.data["result_for_llm"] == "Found 10 genomes"
        assert result_event.data["result_for_ui"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_missing_chat_tool(self):
        """Test error when agent doesn't have the chat tool."""
        agent = _make_agent_handle(tool_names=["other_tool"])

        step = Step(agent_key="data", task="find genomes")
        request = _make_request()

        events = await collect_events(
            execute_agent_step(step, agent, request)
        )

        event_types = [e.type for e in events]
        assert EventType.ORCHESTRATOR_ERROR in event_types

        error_event = next(
            e for e in events if e.type == EventType.ORCHESTRATOR_ERROR
        )
        assert "agent_chat" in error_event.data["error"]

    @pytest.mark.asyncio
    async def test_mcp_call_failure(self):
        """Test error handling when MCP tool call fails."""
        agent = _make_agent_handle()
        agent.call_tool = AsyncMock(side_effect=Exception("Connection refused"))

        step = Step(agent_key="data", task="find genomes")
        request = _make_request()

        events = await collect_events(
            execute_agent_step(step, agent, request)
        )

        event_types = [e.type for e in events]
        assert EventType.ORCHESTRATOR_ERROR in event_types

    @pytest.mark.asyncio
    async def test_agent_error_result(self):
        """Test handling of agent error results."""
        agent = _make_agent_handle()
        agent.call_tool = AsyncMock(
            return_value=_make_mcp_result("Something went wrong", "error")
        )

        step = Step(agent_key="data", task="find genomes")
        request = _make_request()

        events = await collect_events(
            execute_agent_step(step, agent, request)
        )

        event_types = [e.type for e in events]
        assert EventType.ORCHESTRATOR_ERROR in event_types

    @pytest.mark.asyncio
    async def test_context_passed_to_agent(self):
        """Test that conversation context is passed in the call."""
        agent = _make_agent_handle()
        agent.call_tool = AsyncMock(
            return_value=_make_mcp_result("Found data", "completed")
        )

        step = Step(agent_key="data", task="find genomes")
        request = OrchestratorRequest(
            query="find genomes",
            conversation_summary="Previously discussed E. coli",
            auth_token="test_token",
        )

        events = await collect_events(
            execute_agent_step(step, agent, request)
        )

        # Verify call_tool was called with context and token
        call_args = agent.call_tool.call_args
        arguments = call_args[1].get("arguments") or call_args[0][1]
        assert "context" in arguments
        assert "token" in arguments
        assert arguments["token"] == "test_token"

    @pytest.mark.asyncio
    async def test_custom_chat_tool_name(self):
        """Test using a custom chat tool name from config."""
        agent = _make_agent_handle(
            tool_names=["custom_chat", "other"],
            chat_tool="custom_chat",
        )
        agent.call_tool = AsyncMock(
            return_value=_make_mcp_result("result", "completed")
        )

        step = Step(agent_key="data", task="test")
        request = _make_request()

        events = await collect_events(
            execute_agent_step(step, agent, request)
        )

        # Should have called custom_chat, not agent_chat
        agent.call_tool.assert_called_once()
        call_args = agent.call_tool.call_args
        assert call_args[0][0] == "custom_chat"


# --- Tests: execute_plan ---


class TestExecutePlan:
    @pytest.mark.asyncio
    async def test_single_step_plan(self):
        """Test executing a single-step plan."""
        config = OrchestratorConfig(
            agents={
                "data": AgentConfig(
                    name="Data",
                    description="test",
                    endpoint="http://localhost:9999",
                ),
            },
            health_check_interval=0,
        )
        registry = AgentRegistry(config)

        # Add a mock agent
        agent = _make_agent_handle()
        agent.call_tool = AsyncMock(
            return_value=_make_mcp_result("Found genomes", "completed")
        )
        registry._agents["data"] = agent

        plan = Plan(
            reasoning="data query",
            steps=[Step(agent_key="data", task="find genomes")],
        )
        request = _make_request()

        events = await collect_events(execute_plan(plan, registry, request))

        event_types = [e.type for e in events]
        assert EventType.AGENT_START in event_types
        assert EventType.AGENT_RESULT in event_types

    @pytest.mark.asyncio
    async def test_unknown_agent_in_plan(self):
        """Test plan with unknown agent key."""
        config = OrchestratorConfig(agents={}, health_check_interval=0)
        registry = AgentRegistry(config)

        plan = Plan(
            reasoning="test",
            steps=[Step(agent_key="nonexistent", task="do something")],
        )
        request = _make_request()

        events = await collect_events(execute_plan(plan, registry, request))

        event_types = [e.type for e in events]
        assert EventType.ORCHESTRATOR_ERROR in event_types

    @pytest.mark.asyncio
    async def test_unhealthy_agent_warning(self):
        """Test that unhealthy agents get a warning but still execute."""
        config = OrchestratorConfig(
            agents={
                "data": AgentConfig(
                    name="Data",
                    description="test",
                    endpoint="http://localhost:9999",
                ),
            },
            health_check_interval=0,
        )
        registry = AgentRegistry(config)

        agent = _make_agent_handle(healthy=False)
        agent.call_tool = AsyncMock(
            return_value=_make_mcp_result("result", "completed")
        )
        registry._agents["data"] = agent

        plan = Plan(
            reasoning="test",
            steps=[Step(agent_key="data", task="find genomes")],
        )
        request = _make_request()

        events = await collect_events(execute_plan(plan, registry, request))

        # Should have a warning event about unhealthy agent
        progress_events = [
            e for e in events if e.type == EventType.AGENT_PROGRESS
        ]
        assert len(progress_events) >= 1
        assert "unhealthy" in progress_events[0].data.get("warning", "")
