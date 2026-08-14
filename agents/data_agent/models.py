"""Pydantic models for the Data Retrieval Agent."""

from __future__ import annotations

import json
import re
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


# ---------------------------------------------------------------------------
# Solr-to-RQL converter
# ---------------------------------------------------------------------------


def _escape_rql_value(value: str) -> str:
    """Escape characters that are meaningful in RQL function argument lists."""
    return value.replace("\\", "\\\\").replace(",", "\\,").replace(")", "\\)")


def _solr_term_to_rql(term: str) -> str:
    """Convert a single Solr ``field:value`` term to an RQL expression.

    Handles:
      - ``field:value``              → ``eq(field,value)``
      - ``field:"multi word"``       → ``eq(field,multi word)``
      - ``field:(v1 OR v2 OR v3)``   → ``or(eq(field,v1),eq(field,v2),eq(field,v3))``
      - ``field:[min TO max]``       → ``between(field,min,max)``  (inclusive)
      - ``field:{min TO max}``       → ``between(field,min,max)``  (treated same)
      - ``field:val*``               → ``eq(field,val*)``          (wildcard passthrough)
      - ``*:*`` or ``*``             → ``eq(*,*)``
    """
    if term in ("*:*", "*"):
        return "eq(*,*)"

    # Split on first colon to get field and value
    colon_idx = term.find(":")
    if colon_idx <= 0:
        # No field:value structure — treat as a keyword
        return f"keyword({_escape_rql_value(term)})"

    field = term[:colon_idx].strip()
    raw_value = term[colon_idx + 1 :].strip()

    if not raw_value:
        return f"keyword({_escape_rql_value(field)})"

    # Range query: field:[min TO max] or field:{min TO max} (mixed brackets too)
    range_match = re.match(r"^[\[\{]\s*(.+?)\s+TO\s+(.+?)\s*[\]\}]$", raw_value)
    if range_match:
        low, high = range_match.group(1), range_match.group(2)
        low_rql = _escape_rql_value(low.strip('"'))
        high_rql = _escape_rql_value(high.strip('"'))
        return f"between({field},{low_rql},{high_rql})"

    # Grouped OR values: field:(val1 OR val2 OR val3)
    if raw_value.startswith("(") and raw_value.endswith(")"):
        inner = raw_value[1:-1].strip()
        # Split on " OR " (the boolean operator, not substring inside quotes)
        parts = re.split(r"\s+OR\s+", inner)
        if len(parts) > 1:
            eq_parts = []
            for p in parts:
                p = p.strip().strip('"')
                eq_parts.append(f"eq({field},{_escape_rql_value(p)})")
            return "or(" + ",".join(eq_parts) + ")"
        # Single value in parens
        val = inner.strip('"')
        return f"eq({field},{_escape_rql_value(val)})"

    # Quoted value: field:"multi word value"
    if raw_value.startswith('"') and raw_value.endswith('"'):
        val = raw_value[1:-1]
        return f"eq({field},{_escape_rql_value(val)})"

    # Plain value (may include wildcard)
    return f"eq({field},{_escape_rql_value(raw_value)})"


def _tokenize_solr_query(query: str) -> list[str]:
    """Split a Solr query into tokens preserving quoted strings and brackets.

    Returns a list of tokens: field:value terms, boolean operators (AND, OR,
    NOT), and grouping parentheses.
    """
    tokens: list[str] = []
    i = 0
    n = len(query)

    while i < n:
        # Skip whitespace
        if query[i].isspace():
            i += 1
            continue

        # Grouping parentheses (standalone, not part of field:(v1 OR v2))
        if query[i] == "(" and (i == 0 or query[i - 1] != ":"):
            tokens.append("(")
            i += 1
            continue
        if query[i] == ")":
            # Check if this closes a standalone group paren
            # vs. part of field:(v1 OR v2).  We detect by checking whether
            # the last field:value term we started is still open.
            tokens.append(")")
            i += 1
            continue

        # Boolean operators
        for kw in ("AND", "OR", "NOT"):
            if (
                query[i : i + len(kw)] == kw
                and (i + len(kw) >= n or not query[i + len(kw)].isalnum())
                and (i == 0 or not query[i - 1].isalnum())
            ):
                tokens.append(kw)
                i += len(kw)
                break
        else:
            # Accumulate a field:value term
            term_start = i
            while i < n and not query[i].isspace():
                if query[i] == '"':
                    # Skip quoted string
                    i += 1
                    while i < n and query[i] != '"':
                        i += 1
                    if i < n:
                        i += 1  # skip closing quote
                elif query[i] == "(" and i > term_start and query[i - 1] == ":":
                    # field:(v1 OR v2) — consume until matching ')'
                    depth = 1
                    i += 1
                    while i < n and depth > 0:
                        if query[i] == "(":
                            depth += 1
                        elif query[i] == ")":
                            depth -= 1
                        elif query[i] == '"':
                            i += 1
                            while i < n and query[i] != '"':
                                i += 1
                        i += 1
                elif query[i] == "[" or query[i] == "{":
                    # Range: field:[min TO max] — consume until matching bracket
                    close_char = "]" if query[i] == "[" else "}"
                    i += 1
                    while i < n and query[i] != close_char:
                        i += 1
                    if i < n:
                        i += 1
                elif query[i] == ")":
                    # End of a group paren — don't consume
                    break
                else:
                    i += 1

            token = query[term_start:i].strip()
            if token:
                tokens.append(token)

    return tokens


def _parse_solr_expr(tokens: list[str], pos: int = 0) -> tuple[str, int]:
    """Recursive-descent parser for Solr boolean expressions.

    Returns (rql_string, next_position).
    """
    left, pos = _parse_solr_unary(tokens, pos)

    while pos < len(tokens) and tokens[pos] in ("AND", "OR"):
        op = tokens[pos]
        pos += 1
        # Collect all terms at the same precedence level
        parts = [left]
        rql_op = "and" if op == "AND" else "or"
        right, pos = _parse_solr_unary(tokens, pos)
        parts.append(right)
        # Continue collecting same-level operators
        while pos < len(tokens) and tokens[pos] == op:
            pos += 1
            next_part, pos = _parse_solr_unary(tokens, pos)
            parts.append(next_part)
        left = f"{rql_op}(" + ",".join(parts) + ")"

    return left, pos


def _parse_solr_unary(tokens: list[str], pos: int) -> tuple[str, int]:
    """Parse NOT prefix and parenthesized groups."""
    if pos >= len(tokens):
        return "eq(*,*)", pos

    if tokens[pos] == "NOT":
        pos += 1
        inner, pos = _parse_solr_unary(tokens, pos)
        return f"not({inner})", pos

    if tokens[pos] == "(":
        pos += 1  # skip '('
        inner, pos = _parse_solr_expr(tokens, pos)
        if pos < len(tokens) and tokens[pos] == ")":
            pos += 1  # skip ')'
        return inner, pos

    # Leaf: a field:value term
    term = tokens[pos]
    pos += 1
    return _solr_term_to_rql(term), pos


def solr_to_rql(solr_query: str) -> str:
    """Convert a Solr/Lucene query string to BV-BRC RQL format.

    Handles the subset of Solr syntax produced by the data agent LLM:
      - ``field:value``, ``field:"multi word"``, ``field:(v1 OR v2)``
      - ``AND``, ``OR``, ``NOT`` boolean operators
      - Parenthesized grouping
      - Range queries ``field:[min TO max]``
      - Wildcards ``field:val*``

    Returns an RQL string suitable for BV-BRC viewer URLs and the data API's
    ``application/rqlquery+x-www-form-urlencoded`` content type.
    """
    if not solr_query or not solr_query.strip():
        return "eq(*,*)"

    solr_query = solr_query.strip()
    if solr_query in ("*:*", "*"):
        return "eq(*,*)"

    tokens = _tokenize_solr_query(solr_query)
    if not tokens:
        return "eq(*,*)"

    rql, _ = _parse_solr_expr(tokens, 0)
    return rql


class AgentConfig(BaseModel):
    """Configuration for the data agent. Supports any OpenAI-compatible endpoint.

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
    max_iterations: int = 1000
    max_results_per_query: int = 100
    tool_timeout_seconds: int = 30

    # BV-BRC API
    bvbrc_api_url: str = "https://www.bv-brc.org/api-bulk"
    bvbrc_auth_token: str | None = None

    # BV-BRC Workspace API (for group creation)
    bvbrc_workspace_url: str = "https://p3.theseed.org/services/Workspace"

    # Literature RAG retrieval gateway
    literature_rag_url: str = "http://ash.cels.anl.gov:12006"
    literature_rag_timeout_seconds: int = 45

    # Similar Genome Finder (MinHash service)
    similar_genome_finder_url: str = "https://p3.theseed.org/services/minhash_service"

    # SRA tools
    singularity_container_path: str = (
        "/vol/patric3/production/containers/ubuntu-176-build12-2.sif"
    )

    # MCP server path (for importing data_functions, group_functions, etc.)
    mcp_server_path: str = str(
        Path(__file__).resolve().parent.parent.parent / "mcp_server"
    )


class ToolCall(BaseModel):
    """A single tool call as requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


class ToolExecution(BaseModel):
    """Record of a tool call and its result (or simulated result in plan-only mode)."""

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
    planned_calls: list[ToolCall] = Field(default_factory=list)
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

    def record_planned_call(self, tc: ToolCall) -> None:
        self.planned_calls.append(tc)

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

        # Collect unique collections from both planned and executed calls
        sources: list[str] = []
        all_calls = list(self.planned_calls) + [
            ex.tool_call for ex in self.tool_calls_executed
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
            plan=[
                {
                    "tool": tc.name,
                    "arguments": tc.arguments,
                }
                for tc in self.planned_calls
            ],
            sources=sources,
            tool_trace=self.tool_calls_executed,
            planned_tool_calls=[
                {
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                }
                for tc in self.planned_calls
            ],
            iterations_used=self.iteration,
            status=self.status,
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

        for ex in self.tool_calls_executed:
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

        return {
            "record_ids": record_ids,
            "record_count": record_count,
            "facets": facets,
            "collection": collection,
            "query_used": rql_query,
        }


class AgentResult(BaseModel):
    """Returned by run_agent() / plan_only(). Clean interface for consumers."""

    answer: str
    plan: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    tool_trace: list[ToolExecution] = Field(default_factory=list)
    planned_tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    iterations_used: int = 0
    status: str = "completed"
    elapsed_seconds: float = 0.0
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
