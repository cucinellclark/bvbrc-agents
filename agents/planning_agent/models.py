"""Pydantic models for the BV-BRC Planning Agent.

Subclasses the shared base models. Adds planning-specific domain models
(Plan, PlanStep, ClarificationQuestion).
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from shared.models import (
    ToolCall,  # noqa: F401 -- re-export for backward compat
    ToolExecution,  # noqa: F401
    BaseAgentConfig,
    BaseAgentState,
    BaseAgentResult,
)


class AgentConfig(BaseAgentConfig):
    """Configuration for the planning agent.

    Inherits all fields from BaseAgentConfig. No additional fields needed.
    """

    pass


# ---------------------------------------------------------------------------
# Planning-specific models
# ---------------------------------------------------------------------------


class ClarificationQuestion(BaseModel):
    """A single clarification question with suggested options."""

    id: str
    question: str
    options: list[str]
    required: bool = True


class PlanStep(BaseModel):
    """A single step in a plan, assigned to an agent."""

    step_id: str
    description: str
    agent: str  # "data" | "service" | "workspace" | "helpdesk" | "analysis" | "direct"
    reasoning: str
    depends_on: list[str] = Field(default_factory=list)
    status: Literal["pending", "running", "completed", "failed", "skipped"] = "pending"
    result_summary: str | None = None
    result_data: dict[str, Any] | None = None


class Plan(BaseModel):
    """A structured, multi-step execution plan."""

    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    steps: list[PlanStep]
    status: Literal["draft", "approved", "executing", "completed", "failed"] = "draft"
    current_step_index: int = 0


# ---------------------------------------------------------------------------
# Agent state and result
# ---------------------------------------------------------------------------


class AgentState(BaseAgentState):
    """Tracks the full state of a planning agent execution."""

    # Planning-specific state
    plan: Plan | None = None
    clarification_questions: list[ClarificationQuestion] = Field(default_factory=list)
    step_execution: dict[str, Any] | None = None

    def to_result(self) -> "AgentResult":
        """Convert current state to an AgentResult."""
        elapsed = time.time() - self.start_time

        return AgentResult(
            answer=self.final_answer or "",
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


class AgentResult(BaseAgentResult):
    """Returned by run_agent(). Clean interface for consumers."""

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
