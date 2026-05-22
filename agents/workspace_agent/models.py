"""Pydantic models for the BV-BRC Workspace Exploration Agent.

Key design: AgentResult carries BOTH a natural language answer AND structured
data (file listings, metadata, ui_grids) so the orchestrator/UI can render
rich file browser views alongside the LLM's summary.
"""

from __future__ import annotations

import json
import os
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

# Default config file location: agent_config.yaml next to the Workspace/ directory
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "agent_config.yaml"


class AgentConfig(BaseModel):
    """Configuration for the workspace agent. Supports any OpenAI-compatible endpoint.

    Parameters can be set via:
      1. Shared Agents/config/llm.yaml (LLM defaults for the entire system)
      2. agent_config.yaml (agent-specific overrides, or WORKSPACE_AGENT_CONFIG env var)
      3. Constructor keyword arguments (override file values)
      4. Environment variables: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
    """

    # LLM settings (defaults from shared config)
    llm_base_url: str = _LLM_DEFAULTS["base_url"]
    llm_api_key: str = _LLM_DEFAULTS["api_key"]
    llm_model: str = _LLM_DEFAULTS["model"]
    temperature: float = _LLM_DEFAULTS["temperature"]
    max_tokens: int = _LLM_DEFAULTS["max_tokens"]

    # Context window management
    max_context_tokens: int = 90000  # leave headroom below the model's context limit
    max_tool_result_chars: int = 8000  # per-tool-result truncation budget

    # Agent behavior
    max_iterations: int = 8
    tool_timeout_seconds: int = 30

    # BV-BRC workspace API
    bvbrc_workspace_url: str = "https://p3.theseed.org/services/Workspace"
    bvbrc_auth_token: str | None = None

    # MCP server path (for importing workspace_functions via sys.path)
    mcp_server_path: str = str(
        Path(__file__).resolve().parent.parent.parent / "mcp_server"
    )

    @classmethod
    def from_yaml(cls, config_path: str | Path | None = None, **overrides: Any) -> "AgentConfig":
        """Load configuration from a YAML file, with optional overrides.

        Resolution order:
          1. Explicit ``config_path`` argument
          2. ``WORKSPACE_AGENT_CONFIG`` environment variable
          3. ``agent_config.yaml`` next to the Workspace/ directory

        Any keyword arguments override values loaded from the file.
        """
        import yaml

        path = config_path or os.environ.get("WORKSPACE_AGENT_CONFIG") or _DEFAULT_CONFIG_PATH
        path = Path(path)

        file_values: dict[str, Any] = {}
        if path.exists():
            with open(path) as f:
                raw = yaml.safe_load(f) or {}

            # Flatten the nested YAML structure into flat AgentConfig fields
            llm = raw.get("llm", {})
            if llm:
                if "base_url" in llm:
                    file_values["llm_base_url"] = llm["base_url"]
                if "api_key" in llm:
                    file_values["llm_api_key"] = llm["api_key"]
                if "model" in llm:
                    file_values["llm_model"] = llm["model"]
                if "temperature" in llm:
                    file_values["temperature"] = llm["temperature"]
                if "max_tokens" in llm:
                    file_values["max_tokens"] = llm["max_tokens"]

            ctx = raw.get("context", {})
            if ctx:
                if "max_context_tokens" in ctx:
                    file_values["max_context_tokens"] = ctx["max_context_tokens"]
                if "max_tool_result_chars" in ctx:
                    file_values["max_tool_result_chars"] = ctx["max_tool_result_chars"]

            agent = raw.get("agent", {})
            if agent:
                if "max_iterations" in agent:
                    file_values["max_iterations"] = agent["max_iterations"]
                if "tool_timeout_seconds" in agent:
                    file_values["tool_timeout_seconds"] = agent["tool_timeout_seconds"]

            ws = raw.get("workspace", {})
            if ws:
                if "api_url" in ws:
                    file_values["bvbrc_workspace_url"] = ws["api_url"]
                if "mcp_server_path" in ws:
                    file_values["mcp_server_path"] = ws["mcp_server_path"]

        # Overrides take precedence over file values
        file_values.update(overrides)
        return cls(**file_values)


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
    """Tracks the full state of a workspace agent execution."""

    query: str
    context: dict[str, Any] = Field(default_factory=dict)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls_executed: list[ToolExecution] = Field(default_factory=list)
    iteration: int = 0
    final_answer: str | None = None
    status: Literal["running", "completed", "error", "max_iterations"] = "running"
    start_time: float = Field(default_factory=time.time)

    # Structured data collected during exploration.
    # Each tool execution that returns file listings/metadata appends here.
    # This is passed through to AgentResult so the UI can render grids.
    collected_items: list[dict[str, Any]] = Field(default_factory=list)
    collected_metadata: list[dict[str, Any]] = Field(default_factory=list)
    collected_ui_grids: list[dict[str, Any]] = Field(default_factory=list)
    collected_previews: list[dict[str, Any]] = Field(default_factory=list)

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
        """Record a completed tool execution and extract structured data."""
        self.tool_calls_executed.append(
            ToolExecution(
                tool_call=tc,
                result=result,
                error=error,
                duration_ms=duration_ms,
                iteration=self.iteration,
            )
        )

        # Extract structured data from tool results for passthrough
        if isinstance(result, dict) and not result.get("error"):
            self._extract_structured_data(tc.name, result)

    def _extract_structured_data(self, tool_name: str, result: dict) -> None:
        """Extract file listings, metadata, and ui_grids from tool results."""
        # workspace_browse returns nested result envelope
        inner = result.get("result", result)

        if tool_name in ("workspace_browse", "workspace_search"):
            items = inner.get("items", [])
            if items:
                self.collected_items.extend(items)
            ui_grid = inner.get("ui_grid")
            if ui_grid:
                self.collected_ui_grids.append(ui_grid)

        elif tool_name == "get_file_metadata":
            metadata = inner.get("metadata", inner)
            if metadata:
                self.collected_metadata.append(metadata)

        elif tool_name == "read_file_preview":
            preview = {
                "path": result.get("workspace_path") or result.get("path", ""),
                "data": result.get("data", ""),
                "bytes_read": result.get("bytes_read", 0),
                "total_size": result.get("total_size"),
                "is_complete": result.get("is_complete", False),
            }
            self.collected_previews.append(preview)

    def to_result(self) -> AgentResult:
        elapsed = time.time() - self.start_time

        # Collect unique paths explored
        paths_explored: list[str] = []
        for ex in self.tool_calls_executed:
            tc = ex.tool_call
            path = tc.arguments.get("path")
            if path and path not in paths_explored:
                paths_explored.append(path)

        return AgentResult(
            answer=self.final_answer or "",
            status=self.status,
            # Structured data for UI rendering
            items=self.collected_items,
            metadata=self.collected_metadata,
            ui_grids=self.collected_ui_grids,
            previews=self.collected_previews,
            # Trace info
            paths_explored=paths_explored,
            tool_trace=self.tool_calls_executed,
            iterations_used=self.iteration,
            elapsed_seconds=round(elapsed, 2),
        )


class AgentResult(BaseModel):
    """Returned by run_agent(). Carries both answer text and structured data.

    The dual-output design lets consumers choose how to present results:
    - CLI/chat: use `answer` for a human-readable summary
    - Web UI: use `items`, `ui_grids`, `metadata` to render file browser views
    - API: use the full structured response
    """

    # Natural language summary
    answer: str

    # Agent status
    status: str = "completed"

    # Structured workspace data (passthrough from tool results)
    items: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Flat list of workspace items found (file metadata arrays).",
    )
    metadata: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Detailed metadata for individual files inspected.",
    )
    ui_grids: list[dict[str, Any]] = Field(
        default_factory=list,
        description="UI grid payloads for rendering file browser tables.",
    )
    previews: list[dict[str, Any]] = Field(
        default_factory=list,
        description="File content previews (first N bytes of files read).",
    )

    # Trace info
    paths_explored: list[str] = Field(default_factory=list)
    tool_trace: list[ToolExecution] = Field(default_factory=list)
    iterations_used: int = 0
    elapsed_seconds: float = 0.0

    def pretty(self) -> str:
        """Human-readable summary for CLI output."""
        lines = [
            f"Status: {self.status}",
            f"Iterations: {self.iterations_used}",
            f"Elapsed: {self.elapsed_seconds}s",
        ]

        if self.paths_explored:
            lines.append(f"Paths explored: {', '.join(self.paths_explored)}")

        if self.items:
            lines.append(f"Files found: {len(self.items)}")

        if self.metadata:
            lines.append(f"Files inspected: {len(self.metadata)}")

        if self.previews:
            lines.append(f"Files previewed: {len(self.previews)}")

        # Tool trace
        if self.tool_trace:
            lines.extend(["", "--- TOOL EXECUTIONS ---"])
            for i, ex in enumerate(self.tool_trace, 1):
                tc = ex.tool_call
                duration = f" ({ex.duration_ms:.0f}ms)" if ex.duration_ms else ""
                lines.append(f"\n  Step {i}: {tc.name}{duration}")

                args_str = json.dumps(tc.arguments, default=str)
                if len(args_str) > 120:
                    args_str = json.dumps(tc.arguments, indent=4, default=str)
                lines.append(f"    Args: {args_str}")

                if ex.error:
                    lines.append(f"    ERROR: {ex.error}")
                elif ex.result is not None:
                    result_str = json.dumps(ex.result, default=str)
                    if len(result_str) > 300:
                        r = ex.result
                        if isinstance(r, dict):
                            summary_parts = []
                            inner = r.get("result", r)
                            if "count" in inner:
                                summary_parts.append(f"count={inner['count']}")
                            if "items" in inner and isinstance(inner["items"], list):
                                summary_parts.append(
                                    f"items=[{len(inner['items'])} files]"
                                )
                            if "metadata" in inner:
                                summary_parts.append("metadata=<present>")
                            if summary_parts:
                                result_str = "{" + ", ".join(summary_parts) + "}"
                            else:
                                result_str = result_str[:300] + "..."
                    lines.append(f"    Result: {result_str}")

        if self.answer:
            lines.extend(["", "--- ANSWER ---", self.answer])

        return "\n".join(lines)
