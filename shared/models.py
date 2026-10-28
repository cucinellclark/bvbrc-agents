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
from contextvars import ContextVar
from typing import Any, Literal

from pydantic import BaseModel, Field

from shared.config import LLM_DEFAULTS

# Tool executions of the current agent run, visible to whoever set the list.
# InProcessAgentHandle sets it before dispatching so that when the 300 s
# agent limit cancels the run it can still report what was done (and any job
# already submitted).  Before, the result -- tool trace included -- was lost
# and the user got only a generic "took too long" message.
TOOL_PROGRESS: ContextVar[list | None] = ContextVar("tool_progress", default=None)
MAX_TOOL_PROGRESS = 50

# F1 agent handoff.  dispatch_agent sets an empty dict; record_execution
# fills it when a tool call came back as a handoff (execute_tool decided the
# call belongs to another agent), and dispatch_agent then returns
# status "handoff" for the executor to run that agent.
HANDOFF: ContextVar[dict | None] = ContextVar("agent_handoff", default=None)


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

    LLM endpoint details (base_url, api_key, model) are REQUIRED — they
    arrive per request via ``llm_override`` and are propagated by
    ``shared/agent_dispatch.py:_build_config_kwargs``. There is no default
    model. Structural defaults (temperature, max_tokens, timeout) come from
    ``shared.config.LLM_DEFAULTS`` (sourced from ``config/llm.yaml`` + env).
    """

    # LLM endpoint — must be supplied per request (no default).
    llm_base_url: str
    llm_api_key: str
    llm_model: str

    # Structural knobs — default from config/llm.yaml.
    temperature: float = LLM_DEFAULTS["temperature"]
    max_tokens: int = LLM_DEFAULTS["max_tokens"]
    llm_timeout_seconds: int = LLM_DEFAULTS.get("timeout_seconds", 180)

    # Agent behavior
    max_iterations: int = 1000
    tool_timeout_seconds: int = 30

    # Context window management (opt-in per agent)
    # Prompt budget for trim_messages_to_fit. Qwen3.6 serves 131072 tokens
    # with max_tokens=40000 reserved for output, leaving ~91k for the prompt;
    # 80k leaves room for estimate error (FASTA is underestimated ~7%).
    max_context_tokens: int = 80000
    max_tool_result_chars: int = 8000
    # The model's context window (shared/context_budget.py).  Set by
    # dispatch_agent from the model server (vLLM max_model_len); None means
    # DEFAULT_CONTEXT_WINDOW.  The budget is window - max_tokens.
    context_window_tokens: int | None = None

    # BV-BRC API
    bvbrc_api_url: str = "https://www.bv-brc.org/api-bulk"
    bvbrc_workspace_url: str = "https://p3.theseed.org/services/Workspace"
    bvbrc_auth_token: str | None = None

    # GoWe workflow engine. Set by dispatch_agent from the orchestrator's
    # gowe_url (agents.yaml / GOWE_URL); None means no GoWe, and GoWe tools
    # return an error instead of calling another deployment's server.
    gowe_url: str | None = None

    # Literature RAG retrieval (RAGStack API)
    literature_rag_url: str = "https://www.bv-brc.org/ragstack/asm-next/api"
    literature_rag_timeout_seconds: int = 45
    # "open-access" = PubMed Central open-access (no ASM journals, 2026-10-26).
    # Override with LITERATURE_RAG_COLLECTIONS (comma-separated), read by
    # agent_dispatch._build_config_kwargs.
    literature_rag_collections: list[str] = ["open-access"]

    # Similar Genome Finder (MinHash service)
    similar_genome_finder_url: str = "https://p3.theseed.org/services/minhash_service"

    # SRA tools
    singularity_container_path: str = (
        "/vol/patric3/production/containers/ubuntu-176-build12-2.sif"
    )

    # Session context (injected by orchestrator via agent_chat_tool).
    # Used by submit_gowe_job to rewrite output_path under the chat
    # session's workspace folder.  Defined on BaseAgentConfig so that
    # ALL agents (not just the service agent) can pass session context
    # through to GoWe submissions.
    session_id: str | None = None
    workspace_path: str | None = None

    # Execution mode of the chat session ("plan" | "execute").  Gates the
    # side-effecting tools in ``shared.tools.EXECUTE_ONLY_TOOLS``: in
    # "plan" mode ``execute_tool`` refuses them.  Defaults to "plan" —
    # the safe direction — and is set per request from the session by
    # ``shared/agent_dispatch.py:_build_config_kwargs``.
    execution_mode: Literal["plan", "execute"] = "plan"

    # Which agent this config belongs to ("data", "service", ...), set by
    # ``shared/agent_dispatch.py:dispatch_agent``.  ``execute_tool`` uses it
    # to keep job submission with the service agent (TOOL_AGENT_ALLOWLIST).
    # None (direct callers, tests) is not restricted.
    agent_key: str | None = None
    # The agent that handed this turn to us (F1), or None.  A recipient
    # cannot hand off again: execute_tool gives it the plain refusal.
    handoff_from: str | None = None
    # Set when this run is a delegated plan step: {plan_id, step_id,
    # prior_groups: [{name, path}]}.  create_group's guard allows one group per
    # step and refuses one identical to an earlier step's (bvbrc_mcp.tools.groups).
    plan_step: dict | None = None
    # D1: send chat_template_kwargs.enable_thinking=false on this agent's
    # LLM calls (Qwen models only; see shared.agent_utils.thinking_extra_body).
    disable_thinking: bool = False
    disable_thinking_classifier: bool = False


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
    "handoff",
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

    # Gated tool calls refused by the execution-mode gate this run.
    # Populated by ``record_execution``; each entry is
    # ``{"tool", "summary", "arguments"}``.
    blocked_actions: list[dict[str, Any]] = Field(default_factory=list)

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
        progress = TOOL_PROGRESS.get()
        if progress is not None and len(progress) < MAX_TOOL_PROGRESS:
            progress.append({"tool": tc.name, "arguments": tc.arguments, "result": result, "error": error})
        slot = HANDOFF.get()
        if slot is not None and not slot and isinstance(result, dict) and result.get("handoff_to"):
            slot.update(
                to=result["handoff_to"],
                tool=tc.name,
                arguments=result.get("arguments") or {},
            )
        # A refusal from the execution-mode gate is not a failure — the
        # tool never ran.  Track it so the result can surface what the
        # user needs to switch to Execute mode to run.
        if isinstance(result, dict) and result.get("blocked_by_mode"):
            self.blocked_actions.append(
                {
                    "tool": tc.name,
                    "summary": result.get("summary") or tc.name,
                    "arguments": result.get("arguments") or {},
                }
            )
        # A tool that prepares jobs without running them (rerun_jobs): each
        # one joins the set the user confirms, exactly like a refused submit.
        from shared.tools import prepared_actions_of

        self.blocked_actions.extend(prepared_actions_of(result))


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
    # Gated tool calls refused because the session is in plan mode.
    blocked_actions: list[dict[str, Any]] = Field(default_factory=list)
