"""Pydantic models for the Data Retrieval Agent.

Subclasses the shared base models. Adds data-specific fields
(planned_calls, structured_data) and the Solr-to-RQL converter.
"""

from __future__ import annotations

import json
import re
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

# Solr-to-RQL converter lives in shared/tools/url_utils.py (used by
# search_data() for URL enrichment and by _extract_structured_data).
from shared.tools.url_utils import solr_to_rql  # noqa: F401 -- re-export


# ---------------------------------------------------------------------------
# Agent models
# ---------------------------------------------------------------------------


class AgentConfig(BaseAgentConfig):
    """Configuration for the data agent.

    Adds data-specific fields on top of BaseAgentConfig.
    """

    max_results_per_query: int = 100


class AgentState(BaseAgentState):
    """Tracks the full state of a data agent execution."""

    planned_calls: list[ToolCall] = Field(default_factory=list)

    def record_planned_call(self, tc: ToolCall) -> None:
        self.planned_calls.append(tc)

    def to_result(self) -> "AgentResult":
        elapsed = time.time() - self.start_time

        # Collect unique collections from both planned and executed calls
        sources: list[str] = []
        all_calls = list(self.planned_calls) + [
            ex.tool_call for ex in self.tool_executions
        ]
        for tc in all_calls:
            if (
                tc.name in ("search_data", "facet_query")
                and "collection" in tc.arguments
            ):
                col = tc.arguments["collection"]
                if col not in sources:
                    sources.append(col)

        structured_data = self._extract_structured_data()

        return AgentResult(
            answer=self.final_answer or "",
            status=self.status,
            question=self.question,
            plan=[
                {
                    "tool": tc.name,
                    "arguments": tc.arguments,
                }
                for tc in self.planned_calls
            ],
            sources=sources,
            tool_trace=self.tool_executions,
            planned_tool_calls=[
                {
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                }
                for tc in self.planned_calls
            ],
            iterations_used=self.iteration,
            elapsed_seconds=round(elapsed, 2),
            structured_data=structured_data,
        )

    def _extract_structured_data(self) -> dict[str, Any] | None:
        """Extract structured data from tool executions for downstream agents.

        Pulls record IDs, counts, facet distributions, and query metadata
        from search_data and facet_query tool results.
        """
        record_ids: list[str] = []
        record_count: int | None = None
        facets: dict[str, Any] = {}
        collection: str | None = None
        query_used: str | None = None

        for ex in self.tool_executions:
            if ex.error or not isinstance(ex.result, dict):
                continue
            tc = ex.tool_call

            if tc.name == "search_data":
                collection = tc.arguments.get("collection", collection)
                query_used = tc.arguments.get("rql_query") or tc.arguments.get(
                    "query", query_used
                )

                num_found = ex.result.get("numFound") or ex.result.get("count")
                if num_found is not None:
                    record_count = int(num_found)

                docs = ex.result.get("docs") or ex.result.get("items", [])
                for doc in docs:
                    if isinstance(doc, dict):
                        rid = (
                            doc.get("genome_id")
                            or doc.get("feature_id")
                            or doc.get("id")
                        )
                        if rid and rid not in record_ids:
                            record_ids.append(str(rid))

            elif tc.name == "facet_query":
                facet_field = tc.arguments.get("facet_field", "")
                facet_data = ex.result.get("facets") or ex.result.get(
                    "facet_counts", {}
                )
                if facet_field and facet_data:
                    facets[facet_field] = facet_data

        if not record_ids and record_count is None and not facets:
            return None

        # Convert Solr query to RQL so downstream consumers (frontend viewer
        # links, group creation POST requests) receive the format they expect.
        rql_query = None
        if query_used:
            try:
                rql_query = solr_to_rql(query_used)
            except Exception:
                # Fall back to raw Solr query if conversion fails
                rql_query = query_used

        result = {
            "record_ids": record_ids,
            "record_count": record_count,
            "facets": facets,
            "collection": collection,
            "query_used": rql_query,
        }

        # Attach actionable URLs so the LLM can include markdown links
        # in its response (viewer page, TSV download, FASTA download).
        if rql_query and collection:
            from shared.tools.url_utils import enrich_search_result

            enrich_search_result(result, collection, rql_query)

        return result


class AgentResult(BaseAgentResult):
    """Returned by run_agent() / plan_only(). Clean interface for consumers."""

    plan: list[dict[str, Any]] = Field(default_factory=list)
    planned_tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    structured_data: dict[str, Any] | None = None

    def pretty(self) -> str:
        """Human-readable summary for CLI output."""
        lines = [
            f"Status: {self.status}",
            f"Iterations: {self.iterations_used}",
            f"Elapsed: {self.elapsed_seconds}s",
            f"Collections referenced: {', '.join(self.sources) if self.sources else 'none'}",
        ]

        # Show execution trace if we have real executions
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
                    # Truncate long results for display
                    if len(result_str) > 500:
                        result_str = result_str[:500] + "\n      ... [truncated]"
                    lines.append(f"    Result: {result_str}")

        # Show planned calls if we only have those (plan_only mode)
        elif self.planned_tool_calls:
            lines.extend(["", "--- PLANNED TOOL CALLS ---"])
            for i, tc in enumerate(self.planned_tool_calls, 1):
                lines.append(f"\n  Step {i}: {tc['name']}")
                lines.append(f"    Arguments: {json.dumps(tc['arguments'], indent=6)}")

        if self.answer:
            lines.extend(["", "--- LLM ANSWER ---", self.answer])
        return "\n".join(lines)
