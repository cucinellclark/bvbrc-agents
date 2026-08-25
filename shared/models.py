"""Shared base models for all BV-BRC agents.

Eliminates the copy-pasted ``ToolCall``, ``ToolExecution``, ``AgentConfig``,
``AgentState``, and ``AgentResult`` boilerplate from each agent's models.py.
Agents subclass these and add domain-specific fields.

Usage::

    from shared.models import (
        ToolCall, ToolExecution,
        BaseAgentConfig, BaseAgentState, BaseAgentResult,
    )

    class AgentConfig(BaseAgentConfig):
        # Add agent-specific fields here
        max_results_per_query: int = 100
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from shared.config import LLM_DEFAULTS


# ---------------------------------------------------------------------------
# Tool call / execution records (identical across all agents)
# ---------------------------------------------------------------------------


class ToolCall(BaseModel):
    """A single tool call as requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


class ToolExecution(BaseModel):
    """Record of a tool call and its result."""

    tool_call: ToolCall
    result: Any = None
    error: str | None = None
    duration_ms: float | None = None
    iteration: int = 0


# ---------------------------------------------------------------------------
# Base agent configuration
# ---------------------------------------------------------------------------


class BaseAgentConfig(BaseModel):
    """Common configuration fields shared by all agents.

    LLM defaults are loaded once from ``shared.config.LLM_DEFAULTS``
    (sourced from ``config/llm.yaml`` + environment variable overrides).
    """

    # LLM settings
    llm_base_url: str = LLM_DEFAULTS["base_url"]
    llm_api_key: str = LLM_DEFAULTS["api_key"]
    llm_model: str = LLM_DEFAULTS["model"]
    temperature: float = LLM_DEFAULTS["temperature"]
    max_tokens: int = LLM_DEFAULTS["max_tokens"]

    # Agent behavior
    max_iterations: int = 1000
    tool_timeout_seconds: int = 30

    # Context window management (opt-in per agent)
    max_context_tokens: int = 90000
    max_tool_result_chars: int = 8000

    # BV-BRC API
    bvbrc_api_url: str = "https://www.bv-brc.org/api-bulk"
    bvbrc_workspace_url: str = "https://p3.theseed.org/services/Workspace"
    bvbrc_auth_token: str | None = None

    # GoWe workflow engine
    gowe_url: str = "http://140.221.78.67:12009"

    # Literature RAG retrieval gateway
    literature_rag_url: str = "http://ash.cels.anl.gov:12006"
    literature_rag_timeout_seconds: int = 45

    # Similar Genome Finder (MinHash service)
    similar_genome_finder_url: str = "https://p3.theseed.org/services/minhash_service"

    # SRA tools
    singularity_container_path: str = (
        "/vol/patric3/production/containers/ubuntu-176-build12-2.sif"
    )

    # MCP server path (for importing data_functions, workspace_functions, etc.)
    mcp_server_path: str = str(
        Path(__file__).resolve().parent.parent / "mcp_server"
    )

    # Session context (injected by orchestrator via agent_chat_tool).
    # Used by submit_gowe_job to rewrite output_path under the chat
    # session's workspace folder.  Defined on BaseAgentConfig so that
    # ALL agents (not just the service agent) can pass session context
    # through to GoWe submissions.
    session_id: str | None = None
    workspace_path: str | None = None


# ---------------------------------------------------------------------------
# Base agent state
# ---------------------------------------------------------------------------

# Union of all status values used by any agent.
AgentStatus = Literal[
    "running",
    "completed",
    "error",
    "max_iterations",
    "in_progress",
    "needs_input",
    "needs_approval",
    "step_ready",
]


class BaseAgentState(BaseModel):
    """Common state tracked during an agent execution.

    Provides the message-management helpers and tool-execution recording
    that were previously copy-pasted across all 6 agents.
    """

    query: str = ""
    context: dict[str, Any] = Field(default_factory=dict)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    tool_executions: list[ToolExecution] = Field(default_factory=list)
    iteration: int = 0
    final_answer: str | None = None
    question: str | None = None
    status: AgentStatus = "running"
    start_time: float = Field(default_factory=time.time)

    # ------------------------------------------------------------------
    # Message helpers (previously copy-pasted in every agent)
    # ------------------------------------------------------------------

    def add_system_message(self, content: str) -> None:
        self.messages.append({"role": "system", "content": content})

    def add_user_message(self, content: str | list) -> None:
        """Add a user message.

        Args:
            content: Plain string for text-only messages, or a list of
                OpenAI content blocks for multimodal (text + image) messages.
        """
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(
        self,
        content: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        msg: dict[str, Any] = {"role": "assistant"}
        if content is not None:
            msg["content"] = content
        if tool_calls is not None:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self.messages.append(
            {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        )

    def reset_messages(self) -> None:
        """Clear messages for starting a new sub-loop."""
        self.messages = []

    # ------------------------------------------------------------------
    # Tool execution recording
    # ------------------------------------------------------------------

    def record_execution(
        self,
        tc: ToolCall,
        result: Any = None,
        error: str | None = None,
        duration_ms: float = 0.0,
        iteration: int | None = None,
    ) -> None:
        """Record a completed tool execution.

        Args:
            tc: The tool call that was executed.
            result: The tool's return value.
            error: Error message if the tool failed.
            duration_ms: Wall-clock time for the execution.
            iteration: Override iteration number (defaults to ``self.iteration``).
        """
        self.tool_executions.append(
            ToolExecution(
                tool_call=tc,
                result=result,
                error=error,
                duration_ms=duration_ms,
                iteration=iteration if iteration is not None else self.iteration,
            )
        )


# ---------------------------------------------------------------------------
# Base agent result
# ---------------------------------------------------------------------------


class BaseAgentResult(BaseModel):
    """Common fields returned by every agent's ``to_result()``."""

    answer: str = ""
    status: str = "completed"
    question: str | None = None
    sources: list[str] = Field(default_factory=list)
    tool_trace: list[ToolExecution] = Field(default_factory=list)
    iterations_used: int = 0
    elapsed_seconds: float = 0.0
