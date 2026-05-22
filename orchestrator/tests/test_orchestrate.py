"""Integration tests for the full orchestration loop."""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp.types import Tool as McpTool, CallToolResult, TextContent

from orchestrator.config import AgentConfig, OrchestratorConfig
from orchestrator.events.events import EventType
from orchestrator.events.stream import collect_events
from orchestrator.llm.client import LLMClient
from orchestrator.models import OrchestratorRequest, OrchestratorResponse
from orchestrator.orchestrate import orchestrate, orchestrate_to_response
from orchestrator.registry.agent_handle import AgentHandle
from orchestrator.registry.agent_registry import AgentRegistry


# --- Fixtures ---


def _make_registry() -> AgentRegistry:
    """Create a registry with mock healthy agents."""
    config = OrchestratorConfig(
        agents={
            "data": AgentConfig(
                name="Data Agent",
                description="Retrieves biological data.",
                endpoint="http://localhost:12009",
                capabilities=["data_retrieval"],
            ),
            "service2": AgentConfig(
                name="Service Agent",
                description="Plans service workflows.",
                endpoint="http://localhost:8053",
                capabilities=["workflow_planning"],
            ),
        },
        health_check_interval=0,
    )
    registry = AgentRegistry(config)

    for key in ["data", "service2"]:
        handle = AgentHandle(key, config.agents[key])
        handle._healthy = True
        handle._tools = [
            McpTool(name="agent_chat", description="Chat", inputSchema={}),
            McpTool(name="other_tool", description="Other", inputSchema={}),
        ]
        registry._agents[key] = handle

    return registry


def _make_agent_result(
    answer: str = "Agent answer",
    status: str = "completed",
) -> CallToolResult:
    """Create a mock MCP CallToolResult for agent_chat."""
    data = {
        "answer": answer,
        "status": status,
        "sources": ["genome"],
        "iterations_used": 2,
        "elapsed_seconds": 1.5,
        "tool_trace": [],
    }
    result = MagicMock(spec=CallToolResult)
    result.content = [TextContent(type="text", text=json.dumps(data))]
    result.isError = False
    return result


def _make_llm_client(routing_response: str, synthesis_response: str = "") -> LLMClient:
    """Create a mock LLM client that returns different responses for routing vs synthesis."""
    client = MagicMock(spec=LLMClient)
    # The first call is for routing, subsequent calls are for synthesis
    client.complete = AsyncMock(side_effect=[routing_response, synthesis_response])
    return client


# --- Tests ---


class TestOrchestrate:
    @pytest.mark.asyncio
    async def test_direct_response_flow(self):
        """Test full flow when router decides to respond directly."""
        registry = _make_registry()
        llm_response = json.dumps({
            "decision": "direct",
            "reasoning": "greeting",
            "direct_response": "Hello! I can help you with BV-BRC data and services.",
        })
        llm = MagicMock(spec=LLMClient)
        llm.complete = AsyncMock(return_value=llm_response)

        request = OrchestratorRequest(query="Hello!")

        events = await collect_events(orchestrate(request, registry, llm))

        event_types = [e.type for e in events]
        assert EventType.ORCHESTRATOR_START in event_types
        assert EventType.ROUTING_START in event_types
        assert EventType.ROUTING_DECISION in event_types
        assert EventType.ORCHESTRATOR_DONE in event_types

        # Should NOT have agent execution or synthesis
        assert EventType.AGENT_START not in event_types
        assert EventType.SYNTHESIS_START not in event_types

        done_event = next(e for e in events if e.type == EventType.ORCHESTRATOR_DONE)
        assert "Hello" in done_event.data["response_text"]
        assert done_event.data["decision"] == "direct"

    @pytest.mark.asyncio
    async def test_agent_routing_flow(self):
        """Test full flow: route to data agent -> execute -> synthesize."""
        registry = _make_registry()

        # Mock the agent's call_tool to return a result
        data_agent = registry.get("data")
        data_agent.call_tool = AsyncMock(
            return_value=_make_agent_result("Found 50 E. coli genomes.")
        )

        routing_response = json.dumps({
            "decision": "agent",
            "reasoning": "data query",
            "agent_key": "data",
            "task": "Find E. coli genomes",
        })
        llm = MagicMock(spec=LLMClient)
        llm.complete = AsyncMock(return_value=routing_response)

        request = OrchestratorRequest(query="Find E. coli genomes")

        events = await collect_events(orchestrate(request, registry, llm))

        event_types = [e.type for e in events]
        assert EventType.ORCHESTRATOR_START in event_types
        assert EventType.ROUTING_DECISION in event_types
        assert EventType.AGENT_START in event_types
        assert EventType.AGENT_TOOL_CALL in event_types
        assert EventType.AGENT_RESULT in event_types
        assert EventType.SYNTHESIS_START in event_types
        assert EventType.SYNTHESIS_DONE in event_types
        assert EventType.ORCHESTRATOR_DONE in event_types

        done_event = next(e for e in events if e.type == EventType.ORCHESTRATOR_DONE)
        assert "50 E. coli genomes" in done_event.data["response_text"]
        assert "data" in done_event.data["agents_used"]

    @pytest.mark.asyncio
    async def test_agent_execution_error_flow(self):
        """Test flow when agent execution fails."""
        registry = _make_registry()

        data_agent = registry.get("data")
        data_agent.call_tool = AsyncMock(
            side_effect=Exception("MCP connection failed")
        )

        routing_response = json.dumps({
            "decision": "agent",
            "reasoning": "data query",
            "agent_key": "data",
            "task": "Find genomes",
        })
        # Need synthesis response too since error triggers LLM synthesis
        synthesis_response = "I encountered an error while searching."
        llm = MagicMock(spec=LLMClient)
        llm.complete = AsyncMock(
            side_effect=[routing_response, synthesis_response]
        )

        request = OrchestratorRequest(query="Find genomes")

        events = await collect_events(orchestrate(request, registry, llm))

        event_types = [e.type for e in events]
        assert EventType.ORCHESTRATOR_ERROR in event_types
        assert EventType.ORCHESTRATOR_DONE in event_types

    @pytest.mark.asyncio
    async def test_forced_routing_flow(self):
        """Test flow with target_agent override."""
        registry = _make_registry()

        service_agent = registry.get("service2")
        service_agent.call_tool = AsyncMock(
            return_value=_make_agent_result("Workflow planned.", "completed")
        )

        llm = MagicMock(spec=LLMClient)
        llm.complete = AsyncMock(return_value="unused")

        request = OrchestratorRequest(
            query="Assemble this genome",
            target_agent="service2",
        )

        events = await collect_events(orchestrate(request, registry, llm))

        # Routing LLM should NOT have been called (forced routing)
        # Only synthesis might call it, but single result passes through
        routing_decision = next(
            e for e in events if e.type == EventType.ROUTING_DECISION
        )
        assert routing_decision.data["agent_key"] == "service2"

        done_event = next(e for e in events if e.type == EventType.ORCHESTRATOR_DONE)
        assert "Workflow planned" in done_event.data["response_text"]


class TestOrchestrateToResponse:
    @pytest.mark.asyncio
    async def test_returns_orchestrator_response(self):
        """Test the convenience wrapper returns OrchestratorResponse."""
        registry = _make_registry()

        data_agent = registry.get("data")
        data_agent.call_tool = AsyncMock(
            return_value=_make_agent_result("Found genomes.")
        )

        routing_response = json.dumps({
            "decision": "agent",
            "reasoning": "data",
            "agent_key": "data",
            "task": "find genomes",
        })
        llm = MagicMock(spec=LLMClient)
        llm.complete = AsyncMock(return_value=routing_response)

        request = OrchestratorRequest(query="Find genomes")

        response = await orchestrate_to_response(request, registry, llm)

        assert isinstance(response, OrchestratorResponse)
        assert response.response_text == "Found genomes."
        assert response.agent_used == "data"
        assert "data" in response.agents_used
        assert response.status == "completed"
        assert len(response.execution_trace) > 0

    @pytest.mark.asyncio
    async def test_direct_response(self):
        """Test convenience wrapper with direct response."""
        registry = _make_registry()

        llm_response = json.dumps({
            "decision": "direct",
            "reasoning": "greeting",
            "direct_response": "Hello!",
        })
        llm = MagicMock(spec=LLMClient)
        llm.complete = AsyncMock(return_value=llm_response)

        request = OrchestratorRequest(query="Hi")

        response = await orchestrate_to_response(request, registry, llm)

        assert response.response_text == "Hello!"
        assert response.agent_used is None
        assert response.agents_used == []
        assert response.status == "completed"
