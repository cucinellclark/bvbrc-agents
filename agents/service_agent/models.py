"""Pydantic models for the BV-BRC Service Agent.

Key models:
  - AgentConfig: LLM and API configuration
  - Intent: intent classifier output
  - AgentState: state of one service-agent run (populate loop, replay,
    lifecycle handlers)
  - AgentResult: Final output returned to the orchestrator
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field

from shared.models import (
    ToolCall,  # noqa: F401 -- re-export for backward compat
    ToolExecution,  # noqa: F401
    BaseAgentConfig,
    BaseAgentState,
    BaseAgentResult,
)
from shared.config import LLM_DEFAULTS


# ---------------------------------------------------------------------------
# Agent configuration
# ---------------------------------------------------------------------------


class AgentConfig(BaseAgentConfig):
    """Configuration for the service agent.

    Adds service-specific fields on top of BaseAgentConfig.
    """

    # Optional lightweight model for intent classification.
    classifier_model: str | None = LLM_DEFAULTS.get("classifier_model")

    # session_id and workspace_path are inherited from BaseAgentConfig.


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------


class Intent(BaseModel):
    """Result of the lightweight intent classifier."""

    action: Literal["plan", "submit", "status", "cancel", "modify", "unknown"] = "plan"
    workflow_id: str | None = None
    confidence: float = 1.0
    reasoning: str = ""
    submit_after_plan: bool = False
    # D4: "many" when the message asks for several jobs (several services on
    # one input, "all services", or several samples run separately).  The
    # service agent then prepares the set under the plan gate and asks once.
    job_count: Literal["one", "many"] = "one"
    services_named: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent state (serializable for pause/resume)
# ---------------------------------------------------------------------------


class AgentState(BaseAgentState):
    """Tracks the state of one service-agent run."""

    # Phase tracking
    current_phase: Literal["start", "populate", "ready", "done", "handoff"] = "start"

    # Workflow engine persistence
    workflow_id: str | None = None
    persisted: bool = False

    # GoWe-specific state
    submission_id: str | None = None
    submission_ids: list[str] = Field(default_factory=list)
    submission_inputs: dict | None = None

    # Agent result status (override base with service-specific literals)
    status: Literal["in_progress", "needs_input", "completed", "error"] = "in_progress"  # type: ignore[assignment]
    question: str | None = None
    error_message: str | None = None

    # Lifecycle operation output (non-planning paths)
    operation_message: str | None = None
    auto_submitted: bool = False

    # Intent from classifier
    classified_intent: Intent | None = None

    # D4 multi-job mode: the populate loop runs under the plan gate whatever
    # the session mode, keeps tools enabled after each blocked submit (up to
    # MAX_PREPARED_JOBS), and the user confirms the set with one question.
    multi_job: bool = False
    # Submits refused because the set was already at MAX_PREPARED_JOBS.
    capped_submits: list[str] = Field(default_factory=list)

    # -----------------------------------------------------------------------
    # Result conversion
    # -----------------------------------------------------------------------

    def to_result(self) -> "AgentResult":
        """Convert current state to an AgentResult for the caller."""
        elapsed = time.time() - self.start_time

        return AgentResult(
            status=self.status,
            question=self.question,
            error_message=self.error_message,
            tool_trace=self.tool_executions,
            elapsed_seconds=round(elapsed, 2),
            blocked_actions=list(self.blocked_actions),
            workflow_id=self.workflow_id,
            persisted=self.persisted,
            operation_message=self.operation_message,
            auto_submitted=self.auto_submitted,
            submission_id=self.submission_id,
            submission_ids=self.submission_ids,
        )


# ---------------------------------------------------------------------------
# Agent result (returned to caller / orchestrator)
# ---------------------------------------------------------------------------


class AgentResult(BaseAgentResult):
    """Returned by run_agent(). Clean interface for consumers.

    Status values:
      - "completed": jobs submitted or prepared, or a lifecycle operation done
      - "needs_input": Agent needs user input; question in `question`
      - "error": Unrecoverable error; details in `error_message`
    """

    question: str | None = None
    error_message: str | None = None
    workflow_id: str | None = None
    persisted: bool = False

    # GoWe-specific fields
    submission_id: str | None = None
    submission_ids: list[str] = Field(default_factory=list)

    # Lifecycle operation output
    operation_message: str | None = None
    auto_submitted: bool = False

    def pretty(self) -> str:
        """Plain summary, the answer when no message, question or error was set."""
        import json

        if self.operation_message:
            return self.operation_message

        lines = [
            f"Status: {self.status}",
            f"Elapsed: {self.elapsed_seconds}s",
        ]

        if self.question:
            lines.extend(["", "--- QUESTION FOR USER ---", self.question])

        if self.error_message:
            lines.extend(["", "--- ERROR ---", self.error_message])

        if self.workflow_id:
            lines.extend(["", f"Workflow ID: {self.workflow_id}"])

        if self.tool_trace:
            lines.extend(
                [
                    "",
                    f"--- TOOL EXECUTIONS ({len(self.tool_trace)}) ---",
                ]
            )
            for i, ex in enumerate(self.tool_trace, 1):
                tc = ex.tool_call
                duration = f" ({ex.duration_ms:.0f}ms)" if ex.duration_ms else ""
                lines.append(f"\n  {i}. {tc.name}{duration}")
                args_str = json.dumps(tc.arguments, default=str)
                if len(args_str) > 120:
                    args_str = json.dumps(tc.arguments, indent=4, default=str)
                lines.append(f"     Args: {args_str}")
                if ex.error:
                    lines.append(f"     ERROR: {ex.error}")

        return "\n".join(lines)
