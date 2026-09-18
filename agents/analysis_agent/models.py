"""Pydantic models for the BV-BRC Analysis Agent.

Key design: AgentResult carries BOTH a natural language answer AND structured
data (output_files, metrics, previews, report_links, step_summaries) so the
orchestrator/UI can render rich analysis views alongside the LLM's summary.
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
    """Configuration for the analysis agent.

    Adds analysis-specific fields on top of BaseAgentConfig.
    """

    max_preview_bytes: int = 32768  # 32 KB, matches read_file_preview page size


class AgentState(BaseAgentState):
    """Tracks the full state of an analysis agent execution."""

    # Structured data collected during analysis.
    # Each tool execution that returns relevant data appends here.
    # Passed through to AgentResult for UI rendering.
    collected_output_files: list[dict[str, Any]] = Field(default_factory=list)
    collected_metrics: list[dict[str, Any]] = Field(default_factory=list)
    collected_previews: list[dict[str, Any]] = Field(default_factory=list)
    collected_report_links: list[dict[str, Any]] = Field(default_factory=list)
    collected_step_summaries: list[dict[str, Any]] = Field(default_factory=list)
    # Byte ranges for subsequent pages (first page data is in collected_previews)
    preview_ranges: list[dict[str, Any]] = Field(default_factory=list)

    def record_execution(
        self,
        tc: ToolCall,
        result: Any = None,
        error: str | None = None,
        duration_ms: float = 0.0,
        iteration: int | None = None,
    ) -> None:
        """Record a completed tool execution and extract structured data."""
        super().record_execution(
            tc=tc, result=result, error=error, duration_ms=duration_ms,
            iteration=iteration,
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
                        self.collected_report_links.append(
                            {
                                "path": item.get("path", ""),
                                "label": name,
                            }
                        )

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
            file_path = result.get("workspace_path") or result.get("path", "")
            sb = result.get("start_byte", 0)

            if sb == 0:
                # First page: store data for UI/result rendering
                preview = {
                    "path": file_path,
                    "data": result.get("data", ""),
                    "bytes_read": result.get("bytes_read", 0),
                    "total_size": result.get("total_size"),
                }
                self.collected_previews.append(preview)
            else:
                # Subsequent pages: store only byte range metadata
                # to avoid bloating AgentResult
                self.preview_ranges.append({
                    "path": file_path,
                    "start_byte": sb,
                    "next_start": result.get("next_start"),
                    "bytes_read": result.get("bytes_read", 0),
                    "is_complete": result.get("is_complete", False),
                })

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
            question=self.question,
            sources=sources,
            tool_trace=self.tool_executions,
            iterations_used=self.iteration,
            elapsed_seconds=round(elapsed, 2),
            # Analysis-specific structured data
            output_files=self.collected_output_files,
            metrics=self.collected_metrics,
            previews=self.collected_previews,
            report_links=self.collected_report_links,
            step_summaries=self.collected_step_summaries,
        )


class AgentResult(BaseAgentResult):
    """Returned by run_agent(). Carries both answer text and structured data.

    The dual-output design lets consumers choose how to present results:
    - CLI/chat: use `answer` for a human-readable summary
    - Web UI: use `output_files`, `metrics`, `report_links`, `step_summaries`
      to render rich analysis cards
    - API: use the full structured response
    """

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
