"""Tests for analysis agent routing and dispatch integration.

Tests cover:
  1. Routing: analysis-related queries route to the analysis agent
  2. Dispatch: agent_type="analysis" dispatches correctly in agent_chat_tool
  3. OrchestratorRequest: workflow_context field is properly threaded
"""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp.types import Tool as McpTool, CallToolResult, TextContent

from orchestrator.config import AgentConfig, OrchestratorConfig
from orchestrator.events.events import EventType
from orchestrator.events.stream import collect_events
from orchestrator.executor.agent_executor import execute_agent_step
from orchestrator.models import OrchestratorRequest
from orchestrator.registry.agent_handle import AgentHandle
from orchestrator.registry.agent_registry import AgentRegistry
from orchestrator.router.models import Plan, Step
from orchestrator.router.prompts import build_routing_prompt
from orchestrator.router.router import _fallback_routing, _parse_routing_response


# --- Fixtures ---


def _make_registry_with_all_agents() -> AgentRegistry:
    """Create a registry with mock healthy agents including analysis."""
    config = OrchestratorConfig(
        agents={
            "data": AgentConfig(
                name="Data Agent",
                description="Retrieves biological data from BV-BRC.",
                endpoint="http://localhost:8053",
                capabilities=["data_retrieval", "solr_query"],
            ),
            "service2": AgentConfig(
                name="Service Agent",
                description="Constructs BV-BRC service workflows.",
                endpoint="http://localhost:8053",
                capabilities=["workflow_planning", "service_configuration"],
            ),
            "workspace": AgentConfig(
                name="Workspace Agent",
                description="Explores the user's BV-BRC cloud workspace.",
                endpoint="http://localhost:8053",
                capabilities=["workspace_browsing", "file_search"],
            ),
            "helpdesk": AgentConfig(
                name="Helpdesk Agent",
                description="Answers questions about how to use BV-BRC.",
                endpoint="http://localhost:8053",
                capabilities=["helpdesk_guidance"],
            ),
            "analysis": AgentConfig(
                name="Analysis Agent",
                description=(
                    "Analyzes output files from completed BV-BRC service jobs. "
                    "Browses job output directories, reads key result files, "
                    "and extracts service-specific metrics."
                ),
                endpoint="http://localhost:8053",
                capabilities=[
                    "output_analysis",
                    "metric_extraction",
                    "result_summarization",
                ],
                chat_tool_params={"agent_type": "analysis"},
            ),
        },
        health_check_interval=0,
    )
    registry = AgentRegistry(config)

    for key, agent_config in config.agents.items():
        handle = AgentHandle(key, agent_config)
        handle._healthy = True
        handle._tools = [
            McpTool(
                name="agent_chat",
                description=f"Chat with {key} agent",
                inputSchema={},
            )
        ]
        registry._agents[key] = handle

    return registry


def _make_agent_handle(
    key: str = "analysis",
    tool_names: list[str] | None = None,
    healthy: bool = True,
) -> AgentHandle:
    """Create a mock AgentHandle for the analysis agent."""
    config = AgentConfig(
        name="Analysis Agent",
        description="Analyzes output files from completed BV-BRC service jobs.",
        endpoint="http://localhost:8053",
        capabilities=["output_analysis", "metric_extraction"],
        chat_tool_params={"agent_type": "analysis"},
    )
    handle = AgentHandle(key, config)
    handle._healthy = healthy

    if tool_names is None:
        tool_names = ["agent_chat"]

    handle._tools = [
        McpTool(name=name, description=f"Tool: {name}", inputSchema={})
        for name in tool_names
    ]

    return handle


def _make_analysis_mcp_result(
    answer: str = "Assembly produced 47 contigs with N50 of 234,891 bp.",
    status: str = "completed",
    sources: list[str] | None = None,
    output_files: list[dict] | None = None,
    metrics: list[dict] | None = None,
    report_links: list[dict] | None = None,
) -> CallToolResult:
    """Create a mock MCP CallToolResult for the analysis agent."""
    data = {
        "answer": answer,
        "status": status,
        "sources": sources or ["GenomeAssembly2"],
        "iterations_used": 3,
        "elapsed_seconds": 5.2,
        "tool_trace": [],
        "output_files": output_files or [
            {"path": "/user/home/.assembly/contigs.fasta", "name": "contigs.fasta", "type": "contigs", "size": 4800000},
        ],
        "metrics": metrics or [
            {"service": "GenomeAssembly2", "metric_name": "N50", "value": 234891, "unit": "bp"},
            {"service": "GenomeAssembly2", "metric_name": "Total contigs", "value": 47, "unit": "count"},
        ],
        "previews": [],
        "report_links": report_links or [
            {"path": "/user/home/.assembly/AssemblyReport.html", "label": "AssemblyReport.html"},
        ],
        "step_summaries": [],
    }
    result = MagicMock(spec=CallToolResult)
    result.content = [TextContent(type="text", text=json.dumps(data))]
    result.isError = False
    return result


# --- Tests: Routing ---


class TestAnalysisRouting:
    """Test that analysis-related queries route to the analysis agent."""

    def test_fallback_routing_analyze_results(self):
        """'analyze my assembly results' routes to analysis agent."""
        registry = _make_registry_with_all_agents()
        result = _fallback_routing("analyze my assembly results", registry)
        assert result.decision == "agent"
        assert result.plan.steps[0].agent_key == "analysis"

    def test_fallback_routing_summarize_outputs(self):
        """'summarize the output files' routes to analysis agent."""
        registry = _make_registry_with_all_agents()
        result = _fallback_routing("summarize the output from my job", registry)
        assert result.decision == "agent"
        assert result.plan.steps[0].agent_key == "analysis"

    def test_fallback_routing_what_did_job_find(self):
        """'what did my BLAST job find' routes to analysis agent."""
        registry = _make_registry_with_all_agents()
        result = _fallback_routing("what did my BLAST job results show", registry)
        assert result.decision == "agent"
        assert result.plan.steps[0].agent_key == "analysis"

    def test_fallback_routing_job_output(self):
        """'look at the job output' routes to analysis agent."""
        registry = _make_registry_with_all_agents()
        result = _fallback_routing("look at the job output from my analysis", registry)
        assert result.decision == "agent"
        assert result.plan.steps[0].agent_key == "analysis"

    def test_llm_routing_analysis_query(self):
        """Test LLM JSON response routes to analysis agent."""
        registry = _make_registry_with_all_agents()
        raw = json.dumps({
            "decision": "agent",
            "reasoning": "User wants to analyze job results",
            "agent_key": "analysis",
            "task": "Analyze the assembly results and extract metrics",
        })
        result = _parse_routing_response(raw, "analyze my assembly results", registry)
        assert result.decision == "agent"
        assert result.plan is not None
        assert result.plan.steps[0].agent_key == "analysis"

    def test_service_routing_not_confused_with_analysis(self):
        """Action-oriented queries should route to service, not analysis."""
        registry = _make_registry_with_all_agents()
        result = _fallback_routing("run blast alignment and build a phylogenetic tree", registry)
        assert result.decision == "agent"
        assert result.plan.steps[0].agent_key == "service2"


# --- Tests: Routing prompt includes analysis agent ---


class TestAnalysisPrompts:
    def test_routing_prompt_includes_analysis(self):
        """System prompt mentions analysis agent routing rules."""
        system, user = build_routing_prompt(
            query="analyze my results",
            agent_catalog="analysis: Analyzes output files from completed jobs.",
        )
        assert "analysis" in system.lower()
        assert "analyze" in user.lower()


# --- Tests: Dispatch (execute_agent_step with analysis agent) ---


class TestAnalysisDispatch:
    @pytest.mark.asyncio
    async def test_successful_analysis_execution(self):
        """Test successful analysis agent step execution."""
        agent = _make_agent_handle()
        agent.call_tool = AsyncMock(
            return_value=_make_analysis_mcp_result()
        )

        step = Step(agent_key="analysis", task="analyze assembly results")
        request = OrchestratorRequest(query="analyze assembly results")

        events = await collect_events(
            execute_agent_step(step, agent, request, step_index=0)
        )

        event_types = [e.type for e in events]
        assert EventType.AGENT_START in event_types
        assert EventType.AGENT_TOOL_CALL in event_types
        assert EventType.AGENT_TOOL_RESULT in event_types
        assert EventType.AGENT_RESULT in event_types

        # Check the result event carries analysis-specific fields
        result_event = next(e for e in events if e.type == EventType.AGENT_RESULT)
        result_for_ui = result_event.data["result_for_ui"]
        assert result_for_ui["agent"] == "analysis"
        assert result_for_ui["status"] == "completed"
        assert "N50" in result_event.data["result_for_llm"]
        # Analysis-specific structured fields pass through
        assert "output_files" in result_for_ui
        assert "metrics" in result_for_ui
        assert "report_links" in result_for_ui

    @pytest.mark.asyncio
    async def test_workflow_context_threaded(self):
        """Test that workflow_context is threaded from request to agent call."""
        agent = _make_agent_handle()
        agent.call_tool = AsyncMock(
            return_value=_make_analysis_mcp_result()
        )

        workflow_context = {
            "workflow_id": "wf_test123",
            "workflow_name": "test-assembly",
            "status": "succeeded",
            "steps": [
                {
                    "step_name": "assembly",
                    "app_name": "GenomeAssembly2",
                    "status": "succeeded",
                    "output_path": "/user/home/CopilotWorkflows",
                    "output_file": "assembly_output",
                }
            ],
            "output_paths": ["/user/home/CopilotWorkflows"],
        }

        step = Step(agent_key="analysis", task="analyze workflow results")
        request = OrchestratorRequest(
            query="analyze workflow results",
            auth_token="test_token",
            workflow_context=workflow_context,
        )

        events = await collect_events(
            execute_agent_step(step, agent, request, step_index=0)
        )

        # Verify call_tool was called with workflow_context in the context JSON
        call_args = agent.call_tool.call_args
        arguments = call_args[1].get("arguments") or call_args[0][1]
        assert "context" in arguments

        context_data = json.loads(arguments["context"])
        assert "workflow_context" in context_data
        assert context_data["workflow_context"]["workflow_id"] == "wf_test123"
        assert context_data["workflow_context"]["status"] == "succeeded"

    @pytest.mark.asyncio
    async def test_agent_type_param_passed(self):
        """Test that agent_type='analysis' is passed in chat_tool_params."""
        agent = _make_agent_handle()
        agent.call_tool = AsyncMock(
            return_value=_make_analysis_mcp_result()
        )

        step = Step(agent_key="analysis", task="analyze results")
        request = OrchestratorRequest(query="analyze results")

        events = await collect_events(
            execute_agent_step(step, agent, request, step_index=0)
        )

        # Verify agent_type was included in call arguments
        call_args = agent.call_tool.call_args
        arguments = call_args[1].get("arguments") or call_args[0][1]
        assert arguments.get("agent_type") == "analysis"


# --- Tests: OrchestratorRequest model ---


class TestOrchestratorRequestModel:
    def test_workflow_context_optional(self):
        """workflow_context is optional and defaults to None."""
        request = OrchestratorRequest(query="hello")
        assert request.workflow_context is None

    def test_workflow_context_set(self):
        """workflow_context can be set with structured data."""
        wf = {
            "workflow_id": "wf_123",
            "workflow_name": "test",
            "status": "succeeded",
            "steps": [{"step_name": "s1", "app_name": "GenomeAssembly2"}],
            "output_paths": ["/user/home/output"],
        }
        request = OrchestratorRequest(query="analyze", workflow_context=wf)
        assert request.workflow_context is not None
        assert request.workflow_context["workflow_id"] == "wf_123"
        assert len(request.workflow_context["steps"]) == 1

    def test_workflow_context_serialization(self):
        """workflow_context round-trips through JSON serialization."""
        wf = {
            "workflow_id": "wf_abc",
            "status": "succeeded",
            "steps": [],
        }
        request = OrchestratorRequest(query="test", workflow_context=wf)
        data = request.model_dump()
        restored = OrchestratorRequest(**data)
        assert restored.workflow_context == wf
