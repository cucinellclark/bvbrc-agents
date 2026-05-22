"""Routing models — Plan, Step, and RoutingDecision.

These define what the router produces: a decision about how to handle
the user's request (direct response, single agent, or future pipeline).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Step(BaseModel):
    """A single step in an execution plan.

    For Phase 2 (single-agent routing), plans have exactly one step.
    The pipeline placeholder is here for Phase 4.
    """

    agent_key: str  # Registry key, e.g. "data", "service2"
    task: str  # Focused task description passed to the agent
    depends_on: list[int] = Field(default_factory=list)  # Step indices


class Plan(BaseModel):
    """An execution plan produced by the router."""

    reasoning: str  # Why the router chose this plan
    steps: list[Step]


class RoutingDecision(BaseModel):
    """The router's output: what to do with this request.

    decision types:
      - "direct": Respond directly without invoking any agent.
      - "agent": Route to a single agent (plan has exactly one step).
      - "pipeline": Multi-agent pipeline (Phase 4 — not yet implemented).
    """

    decision: Literal["direct", "agent", "pipeline"]
    plan: Plan | None = None
    direct_response: str | None = None  # Only set when decision == "direct"
    confidence: float = 1.0  # Router's confidence in this decision
