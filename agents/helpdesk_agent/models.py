"""Pydantic models for the Helpdesk Agent.

Subclasses the shared base models. The helpdesk agent is the simplest --
it has no domain-specific state or result fields beyond the base classes.
"""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import Field

from shared.models import (
    ToolCall,  # noqa: F401 -- re-export for backward compat
    ToolExecution,  # noqa: F401
    BaseAgentConfig,
    BaseAgentState,
    BaseAgentResult,
)


class AgentConfig(BaseAgentConfig):
    """Configuration for the helpdesk agent.

    Inherits all fields from BaseAgentConfig. No additional fields needed.
    """

    pass


class AgentState(BaseAgentState):
    """Tracks the full state of a helpdesk agent execution."""

    def to_result(self) -> "AgentResult":
        elapsed = time.time() - self.start_time

        # Collect source references from tool calls
        sources: list[str] = []
        for ex in self.tool_executions:
            tc = ex.tool_call
            if tc.name == "query_helpdesk":
                if "helpdesk_rag" not in sources:
                    sources.append("helpdesk_rag")
            elif tc.name == "get_service_schema":
                svc = tc.arguments.get("service_name", "")
                label = f"service_schema:{svc}"
                if label not in sources:
                    sources.append(label)
            elif tc.name == "list_services":
                if "service_catalog" not in sources:
                    sources.append("service_catalog")

        return AgentResult(
            answer=self.final_answer or "",
            sources=sources,
            tool_trace=self.tool_executions,
            iterations_used=self.iteration,
            status=self.status,
            elapsed_seconds=round(elapsed, 2),
        )


class AgentResult(BaseAgentResult):
    """Returned by run_agent(). Clean interface for consumers."""

    def pretty(self) -> str:
        """Human-readable summary for CLI output."""
        lines = [
            f"Status: {self.status}",
            f"Iterations: {self.iterations_used}",
            f"Elapsed: {self.elapsed_seconds}s",
            f"Sources: {', '.join(self.sources) if self.sources else 'none'}",
        ]

        if self.tool_trace:
            lines.extend(["", "--- TOOL EXECUTIONS ---"])
            for i, ex in enumerate(self.tool_trace, 1):
                tc = ex.tool_call
                lines.append(f"\n  Step {i}: {tc.name}")
                lines.append(f"    Arguments: {json.dumps(tc.arguments, indent=6)}")
                if ex.duration_ms is not None:
                    lines.append(f"    Duration: {ex.duration_ms:.0f}ms")
                if ex.error:
                    lines.append(f"    ERROR: {ex.error}")
                elif ex.result is not None:
                    result_str = json.dumps(ex.result, indent=6, default=str)
                    if len(result_str) > 500:
                        result_str = result_str[:500] + "\n      ... [truncated]"
                    lines.append(f"    Result: {result_str}")

        if self.answer:
            lines.extend(["", "--- ANSWER ---", self.answer])
        return "\n".join(lines)
