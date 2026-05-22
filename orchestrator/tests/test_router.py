"""Tests for the LLM-powered router."""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp.types import Tool as McpTool

from orchestrator.config import AgentConfig, OrchestratorConfig
from orchestrator.llm.client import LLMClient
from orchestrator.models import OrchestratorRequest
from orchestrator.registry.agent_handle import AgentHandle
from orchestrator.registry.agent_registry import AgentRegistry
from orchestrator.router.models import Plan, RoutingDecision, Step
from orchestrator.router.prompts import build_routing_prompt
from orchestrator.router.router import route, _parse_routing_response, _fallback_routing


# --- Fixtures ---


def _make_registry_with_agents() -> AgentRegistry:
    """Create a registry with mock healthy agents."""
    config = OrchestratorConfig(
        agents={
            "data": AgentConfig(
                name="Data Agent",
                description="Retrieves biological data from BV-BRC.",
                endpoint="http://localhost:12009",
                capabilities=["data_retrieval", "solr_query"],
            ),
            "service2": AgentConfig(
                name="Service Agent",
                description="Constructs BV-BRC service workflows.",
                endpoint="http://localhost:8053",
                capabilities=["workflow_planning", "service_configuration"],
            ),
        },
        health_check_interval=0,
    )
    registry = AgentRegistry(config)

    # Manually register mock agents
    for key, agent_config in config.agents.items():
        handle = AgentHandle(key, agent_config)
        handle._healthy = True
        handle._tools = [
            McpTool(
                name=f"agent_chat",
                description=f"Chat with {key} agent",
                inputSchema={},
            )
        ]
        registry._agents[key] = handle

    return registry


def _make_llm_client(response: str = "") -> LLMClient:
    """Create a mock LLM client."""
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock(return_value=response)
    return client


# --- Tests: RoutingDecision / Plan models ---


class TestRoutingModels:
    def test_step_creation(self):
        step = Step(agent_key="data", task="find genomes")
        assert step.agent_key == "data"
        assert step.task == "find genomes"
        assert step.depends_on == []

    def test_plan_creation(self):
        plan = Plan(
            reasoning="Data query",
            steps=[Step(agent_key="data", task="search genomes")],
        )
        assert len(plan.steps) == 1
        assert plan.reasoning == "Data query"

    def test_routing_decision_direct(self):
        decision = RoutingDecision(
            decision="direct",
            direct_response="Hello!",
        )
        assert decision.decision == "direct"
        assert decision.direct_response == "Hello!"
        assert decision.plan is None

    def test_routing_decision_agent(self):
        decision = RoutingDecision(
            decision="agent",
            plan=Plan(
                reasoning="test",
                steps=[Step(agent_key="data", task="search")],
            ),
        )
        assert decision.decision == "agent"
        assert decision.plan is not None
        assert decision.plan.steps[0].agent_key == "data"


# --- Tests: Prompts ---


class TestPrompts:
    def test_build_routing_prompt_basic(self):
        system, user = build_routing_prompt(
            query="Find E. coli genomes",
            agent_catalog="Agent: Data Agent\n  Description: ...",
        )
        assert "routing" in system.lower()
        assert "E. coli" in user
        assert "Data Agent" in system

    def test_build_routing_prompt_with_context(self):
        system, user = build_routing_prompt(
            query="What about AMR?",
            agent_catalog="catalog text",
            conversation_context="Previously searched for genomes",
        )
        assert "Previously searched" in user
        assert "AMR" in user


# --- Tests: _parse_routing_response ---


class TestParseRoutingResponse:
    def test_parse_direct_response(self):
        registry = _make_registry_with_agents()
        raw = json.dumps({
            "decision": "direct",
            "reasoning": "greeting",
            "direct_response": "Hello! How can I help?",
        })
        result = _parse_routing_response(raw, "hi", registry)
        assert result.decision == "direct"
        assert result.direct_response == "Hello! How can I help?"

    def test_parse_agent_routing(self):
        registry = _make_registry_with_agents()
        raw = json.dumps({
            "decision": "agent",
            "reasoning": "data query",
            "agent_key": "data",
            "task": "Find E. coli genomes",
        })
        result = _parse_routing_response(raw, "find ecoli", registry)
        assert result.decision == "agent"
        assert result.plan is not None
        assert result.plan.steps[0].agent_key == "data"
        assert result.plan.steps[0].task == "Find E. coli genomes"

    def test_parse_with_markdown_fences(self):
        registry = _make_registry_with_agents()
        raw = '```json\n{"decision": "direct", "reasoning": "test", "direct_response": "hi"}\n```'
        result = _parse_routing_response(raw, "hi", registry)
        assert result.decision == "direct"

    def test_parse_with_surrounding_text(self):
        registry = _make_registry_with_agents()
        raw = 'Here is my decision: {"decision": "agent", "reasoning": "data", "agent_key": "data", "task": "search"} done.'
        result = _parse_routing_response(raw, "search", registry)
        assert result.decision == "agent"

    def test_parse_invalid_json_fallback(self):
        registry = _make_registry_with_agents()
        raw = "This is not valid JSON at all"
        result = _parse_routing_response(raw, "find genomes", registry)
        # Should fall back to keyword routing
        assert result.decision == "agent"
        assert result.plan is not None
        assert result.plan.steps[0].agent_key == "data"

    def test_parse_unknown_agent_fallback(self):
        registry = _make_registry_with_agents()
        raw = json.dumps({
            "decision": "agent",
            "reasoning": "unknown",
            "agent_key": "nonexistent_agent",
            "task": "do something",
        })
        result = _parse_routing_response(raw, "find genomes", registry)
        # Should fall back to keyword routing
        assert result.decision == "agent"


# --- Tests: _fallback_routing ---


class TestFallbackRouting:
    def test_data_keywords(self):
        registry = _make_registry_with_agents()
        result = _fallback_routing("find all E. coli genomes", registry)
        assert result.decision == "agent"
        assert result.plan.steps[0].agent_key == "data"
        assert result.confidence < 1.0

    def test_service_keywords(self):
        registry = _make_registry_with_agents()
        result = _fallback_routing("run blast alignment and build a phylogenetic tree", registry)
        assert result.decision == "agent"
        assert result.plan.steps[0].agent_key == "service2"

    def test_ambiguous_defaults_to_data(self):
        registry = _make_registry_with_agents()
        result = _fallback_routing("help me with something", registry)
        # Should default to data agent as fallback
        assert result.decision == "agent"
        assert result.plan.steps[0].agent_key == "data"

    def test_no_healthy_agents(self):
        config = OrchestratorConfig(agents={}, health_check_interval=0)
        registry = AgentRegistry(config)
        result = _fallback_routing("find genomes", registry)
        assert result.decision == "direct"


# --- Tests: route() ---


class TestRoute:
    @pytest.mark.asyncio
    async def test_forced_routing(self):
        """Test target_agent override skips LLM."""
        registry = _make_registry_with_agents()
        llm = _make_llm_client()
        request = OrchestratorRequest(
            query="find genomes",
            target_agent="data",
        )

        result = await route(request, registry, llm)
        assert result.decision == "agent"
        assert result.plan.steps[0].agent_key == "data"
        assert result.confidence == 1.0
        # LLM should NOT have been called
        llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_forced_routing_unknown_agent(self):
        """Test target_agent with unknown agent returns direct."""
        registry = _make_registry_with_agents()
        llm = _make_llm_client()
        request = OrchestratorRequest(
            query="find genomes",
            target_agent="nonexistent",
        )

        result = await route(request, registry, llm)
        assert result.decision == "direct"
        assert "not available" in result.direct_response

    @pytest.mark.asyncio
    async def test_no_healthy_agents(self):
        """Test routing with no healthy agents."""
        config = OrchestratorConfig(agents={}, health_check_interval=0)
        registry = AgentRegistry(config)
        llm = _make_llm_client()
        request = OrchestratorRequest(query="hello")

        result = await route(request, registry, llm)
        assert result.decision == "direct"
        assert "available" in result.direct_response.lower()

    @pytest.mark.asyncio
    async def test_llm_routing_data_query(self):
        """Test LLM routes a data query to the data agent."""
        registry = _make_registry_with_agents()
        llm_response = json.dumps({
            "decision": "agent",
            "reasoning": "data retrieval query",
            "agent_key": "data",
            "task": "Find E. coli genomes in BV-BRC",
        })
        llm = _make_llm_client(llm_response)
        request = OrchestratorRequest(query="Find E. coli genomes")

        result = await route(request, registry, llm)
        assert result.decision == "agent"
        assert result.plan.steps[0].agent_key == "data"
        llm.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_routing_direct_response(self):
        """Test LLM handles a greeting directly."""
        registry = _make_registry_with_agents()
        llm_response = json.dumps({
            "decision": "direct",
            "reasoning": "greeting",
            "direct_response": "Hello! How can I help with BV-BRC?",
        })
        llm = _make_llm_client(llm_response)
        request = OrchestratorRequest(query="Hello")

        result = await route(request, registry, llm)
        assert result.decision == "direct"
        assert "Hello" in result.direct_response

    @pytest.mark.asyncio
    async def test_llm_failure_fallback(self):
        """Test fallback when LLM call fails."""
        registry = _make_registry_with_agents()
        llm = _make_llm_client()
        llm.complete = AsyncMock(side_effect=Exception("LLM unavailable"))
        request = OrchestratorRequest(query="find genomes")

        result = await route(request, registry, llm)
        # Should fall back to keyword routing
        assert result.decision == "agent"
        assert result.plan.steps[0].agent_key == "data"

    @pytest.mark.asyncio
    async def test_routing_with_conversation_context(self):
        """Test that conversation context is passed to LLM."""
        registry = _make_registry_with_agents()
        llm_response = json.dumps({
            "decision": "agent",
            "reasoning": "following up on data",
            "agent_key": "data",
            "task": "find AMR data",
        })
        llm = _make_llm_client(llm_response)
        request = OrchestratorRequest(
            query="What about AMR?",
            conversation_summary="User previously searched for E. coli genomes",
            recent_messages=[
                {"role": "user", "content": "Find E. coli genomes"},
                {"role": "assistant", "content": "Found 100 genomes"},
            ],
        )

        result = await route(request, registry, llm)
        assert result.decision == "agent"
        # Verify context was included in the prompt
        call_args = llm.complete.call_args
        prompt = call_args.kwargs.get("prompt", "")
        assert "previously searched" in prompt.lower()
