"""Pydantic models for the Helpdesk Agent."""

from __future__ import annotations

import json
import sys
import time
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
    """Configuration for the helpdesk agent.

    LLM defaults are loaded from the shared Agents/config/llm.yaml.
    Override via constructor kwargs, CLI args, or environment variables
    (LLM_BASE_URL, LLM_API_KEY, LLM_MODEL).
    """

    # LLM settings (defaults from shared config)
    llm_base_url: str = _LLM_DEFAULTS["base_url"]
    llm_api_key: str = _LLM_DEFAULTS["api_key"]
    llm_model: str = _LLM_DEFAULTS["model"]
    temperature: float = _LLM_DEFAULTS["temperature"]
    max_tokens: int = _LLM_DEFAULTS["max_tokens"]

    # Agent behavior
    max_iterations: int = 8
    tool_timeout_seconds: int = 30

    # BV-BRC API (for service schema lookups)
    bvbrc_api_url: str = "https://www.bv-brc.org/api-bulk"
    bvbrc_auth_token: str | None = None

    # MCP server path (for importing functions)
    mcp_server_path: str = str(
        Path(__file__).resolve().parent.parent.parent / "mcp_server"
    )


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
    """Tracks the full state of an agent execution."""

    query: str
    context: dict[str, Any] = Field(default_factory=dict)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls_executed: list[ToolExecution] = Field(default_factory=list)
    iteration: int = 0
    final_answer: str | None = None
    status: Literal["running", "completed", "error", "max_iterations"] = "running"
    start_time: float = Field(default_factory=time.time)

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
        self.tool_calls_executed.append(
            ToolExecution(
                tool_call=tc,
                result=result,
                error=error,
                duration_ms=duration_ms,
                iteration=self.iteration,
            )
        )

    def to_result(self) -> AgentResult:
        elapsed = time.time() - self.start_time

        # Collect source references from tool calls
        sources: list[str] = []
        for ex in self.tool_calls_executed:
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
            tool_trace=self.tool_calls_executed,
            iterations_used=self.iteration,
            status=self.status,
            elapsed_seconds=round(elapsed, 2),
        )


class AgentResult(BaseModel):
    """Returned by run_agent(). Clean interface for consumers."""

    answer: str
    sources: list[str] = Field(default_factory=list)
    tool_trace: list[ToolExecution] = Field(default_factory=list)
    iterations_used: int = 0
    status: str = "completed"
    elapsed_seconds: float = 0.0

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
