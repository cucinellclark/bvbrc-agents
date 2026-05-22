"""Typed event system for orchestrator streaming output.

Every operation in the orchestrator produces a stream of Event objects.
The gateway (Copilot API) maps these to its SSE protocol for the UI.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """All event types the orchestrator can emit."""

    # --- Lifecycle ---
    ORCHESTRATOR_START = "orchestrator_start"
    ORCHESTRATOR_DONE = "orchestrator_done"
    ORCHESTRATOR_ERROR = "orchestrator_error"

    # --- Routing ---
    ROUTING_START = "routing_start"
    ROUTING_DECISION = "routing_decision"

    # --- Agent execution ---
    AGENT_START = "agent_start"
    AGENT_PROGRESS = "agent_progress"
    AGENT_TOOL_CALL = "agent_tool_call"
    AGENT_TOOL_RESULT = "agent_tool_result"
    AGENT_RESULT = "agent_result"
    AGENT_ERROR = "agent_error"

    # --- Response synthesis ---
    SYNTHESIS_START = "synthesis_start"
    SYNTHESIS_CHUNK = "synthesis_chunk"
    SYNTHESIS_DONE = "synthesis_done"

    # --- Agent interaction ---
    NEEDS_INPUT = "needs_input"

    # --- Registry / discovery ---
    DISCOVERY_START = "discovery_start"
    DISCOVERY_AGENT = "discovery_agent"
    DISCOVERY_DONE = "discovery_done"

    # --- Health ---
    HEALTH_CHECK = "health_check"


class Event(BaseModel):
    """A single orchestrator event.

    Events are the universal output unit. Every async generator in the
    orchestrator yields Event objects, making the entire pipeline composable
    and streamable.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    type: EventType
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)

    # Optional context fields for tracing
    agent_name: str | None = None
    step_index: int | None = None

    def __str__(self) -> str:
        agent = f" [{self.agent_name}]" if self.agent_name else ""
        return f"Event({self.type.value}{agent})"


# --- Convenience constructors ---


def discovery_event(agent_name: str, tool_count: int) -> Event:
    return Event(
        type=EventType.DISCOVERY_AGENT,
        agent_name=agent_name,
        data={"agent": agent_name, "tool_count": tool_count},
    )


def agent_start_event(agent_name: str, task: str) -> Event:
    return Event(
        type=EventType.AGENT_START,
        agent_name=agent_name,
        data={"agent": agent_name, "task": task},
    )


def agent_result_event(
    agent_name: str,
    result_for_llm: str,
    result_for_ui: dict[str, Any] | None = None,
) -> Event:
    return Event(
        type=EventType.AGENT_RESULT,
        agent_name=agent_name,
        data={
            "agent": agent_name,
            "result_for_llm": result_for_llm,
            "result_for_ui": result_for_ui or {},
        },
    )


def error_event(
    message: str,
    agent_name: str | None = None,
    details: dict[str, Any] | None = None,
) -> Event:
    return Event(
        type=EventType.ORCHESTRATOR_ERROR,
        agent_name=agent_name,
        data={"error": message, **(details or {})},
    )


def health_event(agent_name: str, healthy: bool, latency_ms: float) -> Event:
    return Event(
        type=EventType.HEALTH_CHECK,
        agent_name=agent_name,
        data={
            "agent": agent_name,
            "healthy": healthy,
            "latency_ms": round(latency_ms, 1),
        },
    )
