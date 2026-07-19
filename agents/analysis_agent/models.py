"""Pydantic models for the BV-BRC Analysis Agent.

Key design: AgentResult carries BOTH a natural language answer AND structured
data (output_files, metrics, previews, report_links, step_summaries) so the
orchestrator/UI can render rich analysis views alongside the LLM's summary.

Follows the same pattern as workspace_agent/models.py:
  - sys.path manipulation for shared config
  - load_llm_defaults for LLM settings
  - AgentConfig, AgentState, ToolExecution, AgentResult
"""

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
    """Configuration for the analysis agent. Supports any OpenAI-compatible endpoint.

    LLM defaults are loaded from the shared config/llm.yaml.
    Override via constructor kwargs or environment variables
    (LLM_BASE_URL, LLM_API_KEY, LLM_MODEL).
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
    max_iterations: int = 1000
    tool_timeout_seconds: int = 30

    # Analysis-specific
    max_preview_bytes: int = 8192  # 8KB, same as workspace agent

    # BV-BRC workspace API
    bvbrc_workspace_url: str = "https://p3.theseed.org/services/Workspace"
    bvbrc_auth_token: str | None = None

    # MCP server path (for importing workspace_functions via sys.path)
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
    """Tracks the full state of an analysis agent execution."""

    query: str
    context: dict[str, Any] = Field(default_factory=dict)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls_executed: list[ToolExecution] = Field(default_factory=list)
    iteration: int = 0
    final_answer: str | None = None
    status: Literal["running", "completed", "error", "max_iterations"] = "running"
    start_time: float = Field(default_factory=time.time)

    # Structured data collected during analysis.
    # Each tool execution that returns relevant data appends here.
    # Passed through to AgentResult for UI rendering.
    collected_output_files: list[dict[str, Any]] = Field(default_factory=list)
    collected_metrics: list[dict[str, Any]] = Field(default_factory=list)
    collected_previews: list[dict[str, Any]] = Field(default_factory=list)
    collected_report_links: list[dict[str, Any]] = Field(default_factory=list)
    collected_step_summaries: list[dict[str, Any]] = Field(default_factory=list)

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
        """Extract output files, metrics, previews, and report links from tool results."""
        # workspace_browse returns nested result envelope
        inner = result.get("result", result)

        if tool_name in ("workspace_browse",):
            items = inner.get("items", [])
            for item in items:
                if isinstance(item, dict):
                    file_entry = {
                        "path": item.get("path", ""),
                        "name": item.get("name", ""),
                        "type": item.get("type", ""),
                        "size": item.get("size", 0),
                    }
                    self.collected_output_files.append(file_entry)

                    # Detect HTML reports for report_links
                    name = item.get("name", "")
                    if name.lower().endswith((".html", ".htm")):
                        self.collected_report_links.append({
                            "path": item.get("path", ""),
                            "label": name,
                        })

        elif tool_name == "get_file_metadata":
            metadata = inner.get("metadata", inner)
            if metadata and isinstance(metadata, dict):
                file_entry = {
                    "path": metadata.get("path", ""),
                    "name": metadata.get("name", ""),
                    "type": metadata.get("type", ""),
                    "size": metadata.get("size", 0),
                }
                self.collected_output_files.append(file_entry)

        elif tool_name == "read_file_preview":
            preview = {
                "path": result.get("workspace_path") or result.get("path", ""),
                "data": result.get("data", ""),
                "bytes_read": result.get("bytes_read", 0),
                "total_size": result.get("total_size"),
            }
            self.collected_previews.append(preview)

    def to_result(self) -> "AgentResult":
        elapsed = time.time() - self.start_time

        # Collect unique sources (service names) from context
        sources: list[str] = []
        wf_ctx = self.context.get("workflow_context")
        if wf_ctx and isinstance(wf_ctx, dict):
            for step in wf_ctx.get("steps", []):
                app_name = step.get("app_name", "")
                if app_name and app_name not in sources:
                    sources.append(app_name)

        return AgentResult(
            answer=self.final_answer or "",
            status=self.status,
            sources=sources,
            tool_trace=self.tool_calls_executed,
            iterations_used=self.iteration,
            elapsed_seconds=round(elapsed, 2),
            # Analysis-specific structured data
            output_files=self.collected_output_files,
            metrics=self.collected_metrics,
            previews=self.collected_previews,
            report_links=self.collected_report_links,
            step_summaries=self.collected_step_summaries,
        )


class AgentResult(BaseModel):
    """Returned by run_agent(). Carries both answer text and structured data.

    The dual-output design lets consumers choose how to present results:
    - CLI/chat: use `answer` for a human-readable summary
    - Web UI: use `output_files`, `metrics`, `report_links`, `step_summaries`
      to render rich analysis cards
    - API: use the full structured response
    """

    # Natural language summary
    answer: str

    # Agent status
    status: str = "completed"

    # Sources: service names analyzed (e.g., ["GenomeAssembly2"])
    sources: list[str] = Field(default_factory=list)

    # Trace info
    tool_trace: list[ToolExecution] = Field(default_factory=list)
    iterations_used: int = 0
    elapsed_seconds: float = 0.0

    # Analysis-specific structured output for UI rendering
    output_files: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Output files found: [{path, name, type, size, service}].",
    )
    metrics: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Extracted metrics: [{service, metric_name, value, unit}].",
    )
    previews: list[dict[str, Any]] = Field(
        default_factory=list,
        description="File content previews: [{path, data, bytes_read, total_size}].",
    )
    report_links: list[dict[str, Any]] = Field(
        default_factory=list,
        description="HTML reports to link: [{path, label, service}].",
    )
    step_summaries: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per-step summaries: [{step_name, app_name, status, summary, metrics}].",
    )

    def pretty(self) -> str:
        """Human-readable summary for CLI output."""
        lines = [
            f"Status: {self.status}",
            f"Iterations: {self.iterations_used}",
            f"Elapsed: {self.elapsed_seconds}s",
        ]

        if self.sources:
            lines.append(f"Services analyzed: {', '.join(self.sources)}")

        if self.output_files:
            lines.append(f"Output files found: {len(self.output_files)}")

        if self.metrics:
            lines.append(f"Metrics extracted: {len(self.metrics)}")

        if self.report_links:
            lines.append(f"Report links: {len(self.report_links)}")

        if self.step_summaries:
            lines.append(f"Steps summarized: {len(self.step_summaries)}")

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
                        result_str = result_str[:300] + "..."
                    lines.append(f"    Result: {result_str}")

        if self.answer:
            lines.extend(["", "--- ANSWER ---", self.answer])

        return "\n".join(lines)
