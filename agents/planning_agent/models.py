"""Pydantic models for the BV-BRC Planning Agent."""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

# Make the shared config loader importable
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent.parent / "config")
if _CONFIG_DIR not in sys.path:
    sys.path.insert(0, _CONFIG_DIR)

from llm_config import load_llm_defaults  # noqa: E402

_LLM_DEFAULTS = load_llm_defaults()


class AgentConfig(BaseModel):
    """Configuration for the planning agent.

    LLM defaults are loaded from the shared Agents/config/llm.yaml.
    Override via constructor kwargs or environment variables.
    """

    # LLM settings (defaults from shared config)
    llm_base_url: str = _LLM_DEFAULTS["base_url"]
    llm_api_key: str = _LLM_DEFAULTS["api_key"]
    llm_model: str = _LLM_DEFAULTS["model"]
    temperature: float = _LLM_DEFAULTS["temperature"]
    max_tokens: int = _LLM_DEFAULTS["max_tokens"]

    # Agent behavior
    max_iterations: int = 1000

    # BV-BRC API
    bvbrc_auth_token: str | None = None

    # MCP server path (for importing shared functions)
    mcp_server_path: str = str(
        Path(__file__).resolve().parent.parent.parent / "mcp_server"
    )


# ---------------------------------------------------------------------------
# Planning-specific models
# ---------------------------------------------------------------------------


class ClarificationQuestion(BaseModel):
    """A single clarification question with suggested options."""

    id: str
    question: str
    options: list[str]
    required: bool = True


class ReviewConfig(BaseModel):
    """Configuration for a review/checkpoint step.

    Review steps pause plan execution to present intermediate results
    to the user for review, filtering, or decision-making before the
    plan continues.
    """

    data_source_step: str
    review_type: str  # "data_selection" | "workflow_choice" | "parameter_config" | "group_management"
    prompt: str
    suggested_workflows: list[str] = Field(default_factory=list)

    # Group management fields (used when review_type == "group_management")
    suggested_group_name: str | None = (
        None  # Pre-filled name, e.g. "Salmonella AMR Genomes"
    )
    group_type: str | None = None  # "genome_group" | "feature_group"
    group_action: str | None = None  # "create" | "add_to" | "use_existing"
    id_field: str | None = None  # "genome_id" | "feature_id"


class PlanStep(BaseModel):
    """A single step in a plan, assigned to an agent."""

    step_id: str
    description: str
    agent: str  # "data" | "service" | ... | "review" | "direct"
    reasoning: str
    depends_on: list[str] = Field(default_factory=list)
    status: Literal["pending", "running", "completed", "failed", "skipped"] = "pending"
    result_summary: str | None = None
    result_data: dict[str, Any] | None = None
    review_config: ReviewConfig | None = None


class Plan(BaseModel):
    """A structured, multi-step execution plan."""

    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    steps: list[PlanStep]
    status: Literal["draft", "approved", "executing", "completed", "failed"] = "draft"
    current_step_index: int = 0


# ---------------------------------------------------------------------------
# Standard agent models (shared pattern across all agents)
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


class AgentState(BaseModel):
    """Tracks the full state of a planning agent execution."""

    query: str
    context: dict[str, Any] = Field(default_factory=dict)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    tool_executions: list[ToolExecution] = Field(default_factory=list)
    iteration: int = 0
    final_answer: str = ""
    status: Literal[
        "running",
        "completed",
        "needs_input",
        "needs_approval",
        "step_ready",
        "error",
        "max_iterations",
    ] = "running"
    start_time: float = Field(default_factory=time.time)

    # Planning-specific state
    plan: Plan | None = None
    clarification_questions: list[ClarificationQuestion] = Field(default_factory=list)
    step_execution: dict[str, Any] | None = None

    def add_system_message(self, content: str) -> None:
        self.messages.append({"role": "system", "content": content})

    def add_user_message(self, content: str) -> None:
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

    def record_execution(
        self,
        tc: ToolCall,
        result: Any = None,
        error: str | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        """Record a completed tool execution."""
        self.tool_executions.append(
            ToolExecution(
                tool_call=tc,
                result=result,
                error=error,
                duration_ms=duration_ms,
                iteration=self.iteration,
            )
        )

    def to_result(self) -> AgentResult:
        """Convert current state to an AgentResult."""
        elapsed = time.time() - self.start_time

        return AgentResult(
            answer=self.final_answer,
            status=self.status,
            sources=[],
            tool_trace=self.tool_executions,
            iterations_used=self.iteration,
            elapsed_seconds=round(elapsed, 2),
            plan=self.plan.model_dump() if self.plan else None,
            clarification_questions=(
                [q.model_dump() for q in self.clarification_questions]
                if self.clarification_questions
                else None
            ),
            step_execution=self.step_execution,
        )


class AgentResult(BaseModel):
    """Returned by run_agent(). Clean interface for consumers."""

    answer: str
    status: str = "completed"
    sources: list[str] = Field(default_factory=list)
    tool_trace: list[ToolExecution] = Field(default_factory=list)
    iterations_used: int = 0
    elapsed_seconds: float = 0.0

    # Planning-specific fields
    plan: dict[str, Any] | None = None
    clarification_questions: list[dict[str, Any]] | None = None
    step_execution: dict[str, Any] | None = None

    def pretty(self) -> str:
        """Human-readable summary for CLI output."""
        lines = [
            f"Status: {self.status}",
            f"Iterations: {self.iterations_used}",
            f"Elapsed: {self.elapsed_seconds}s",
        ]
        if self.plan:
            lines.append(f"Plan: {self.plan.get('title', 'untitled')}")
            steps = self.plan.get("steps", [])
            for i, step in enumerate(steps, 1):
                lines.append(
                    f"  {i}. [{step.get('agent', '?')}] {step.get('description', '')}"
                )
        if self.clarification_questions:
            lines.append(f"Questions: {len(self.clarification_questions)}")
        if self.answer:
            lines.extend(["", "--- ANSWER ---", self.answer])
        return "\n".join(lines)
