"""Shared Pydantic models for the orchestrator.

These models define the data contracts between orchestrator components
and between the orchestrator and the gateway.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Tool representation
# ---------------------------------------------------------------------------


class ToolDef(BaseModel):
    """A tool definition as discovered from an agent's MCP server."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    agent_key: str = ""  # Which agent owns this tool
    read_only: bool = False
    annotations: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent summary (serializable view of an AgentHandle)
# ---------------------------------------------------------------------------


class AgentSummary(BaseModel):
    """Serializable summary of a registered agent."""

    key: str
    name: str
    description: str
    endpoint: str
    connected: bool
    healthy: bool
    latency_ms: float
    tool_count: int
    tools: list[str]
    capabilities: list[str]


# ---------------------------------------------------------------------------
# Orchestrator request/response (contract with the gateway)
# ---------------------------------------------------------------------------


class LLMOverride(BaseModel):
    """Per-request LLM configuration override.

    When provided, the orchestrator and agents will use this LLM endpoint
    instead of the default from llm.yaml.  Sent by the gateway after
    looking up the user-selected model from MongoDB.
    """

    base_url: str | None = None   # e.g. "http://mango.cels.anl.gov:8004/v1"
    api_key: str | None = None    # e.g. "EMPTY"
    model: str | None = None      # e.g. "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic"


class OrchestratorRequest(BaseModel):
    """Inbound request from the gateway to the orchestrator."""

    query: str
    model: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    auth_token: str | None = None

    # Conversation context (provided by gateway)
    conversation_summary: str | None = None
    recent_messages: list[dict[str, Any]] = Field(default_factory=list)

    # Session context
    workspace_path: str | None = None
    selected_items: list[dict[str, Any]] = Field(default_factory=list)

    # Orchestrator-level overrides
    target_agent: str | None = None  # Force routing to a specific agent
    max_steps: int = 5

    # Per-request LLM override (from gateway model lookup)
    llm_override: LLMOverride | None = None


class OrchestratorResponse(BaseModel):
    """Final response from the orchestrator to the gateway."""

    response_text: str
    agent_used: str | None = None
    agents_used: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    result_for_ui: dict[str, Any] = Field(default_factory=dict)
    execution_trace: list[dict[str, Any]] = Field(default_factory=list)
    status: Literal["completed", "needs_input", "error"] = "completed"
