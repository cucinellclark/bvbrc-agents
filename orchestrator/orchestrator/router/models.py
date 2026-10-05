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

    agent_key: str  # Registry key, e.g. "data", "service"
    task: str  # Focused task description passed to the agent
    depends_on: list[int] = Field(default_factory=list)  # Step indices
    batch_mode: bool = False  # True only for planning-delegated steps


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


# ---------------------------------------------------------------------------
# Structured-output schema
# ---------------------------------------------------------------------------

# JSON schema handed to the routing model via ``response_format`` so the reply
# is guaranteed to carry the field names _parse_routing_response expects.
#
# Deliberately permissive: `decision` is an enum and every shape-specific field
# is optional, because the three decision shapes (direct / agent / pipeline)
# use different subsets and a oneOf is poorly supported by guided-decoding
# backends. This still eliminates the real failure mode, which is the model
# inventing key names.
ROUTING_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["direct", "agent", "pipeline"]},
        "reasoning": {"type": "string"},
        # decision == "agent"
        "agent_key": {"type": "string"},
        "task": {"type": "string"},
        # decision == "direct"
        "direct_response": {"type": "string"},
        # decision == "pipeline"
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "agent_key": {"type": "string"},
                    "task": {"type": "string"},
                    "depends_on": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["agent_key", "task"],
            },
        },
    },
    "required": ["decision", "reasoning"],
}


def build_routing_extra_body(
    disable_thinking: bool = True,
    structured_output: bool = True,
) -> dict:
    """Build the provider-specific request additions for the routing client.

    Returns an empty dict when both features are off, so the routing call is
    byte-identical to the previous behaviour.
    """
    extra: dict = {}
    if disable_thinking:
        extra["chat_template_kwargs"] = {"enable_thinking": False}
    if structured_output:
        extra["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "routing_decision", "schema": ROUTING_JSON_SCHEMA},
        }
    return extra
