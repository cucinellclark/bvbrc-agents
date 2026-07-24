"""
Tool dispatcher for the BV-BRC Data Retrieval Agent.

Maps tool names (from the LLM's tool_calls) to their async implementation
functions.  Core implementations come from the shared tool library.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from shared.tools import execute_tool as _shared_execute_tool
from shared.tools.data import search_data, facet_query, probe_data
from shared.tools.collections import list_collections, get_collection_fields
from shared.tools.similar_genome import find_similar_genomes
from shared.tools.literature import search_literature

# ---------------------------------------------------------------------------
# Dispatch table: tool name -> async callable
# ---------------------------------------------------------------------------
TOOL_DISPATCH: Dict[str, Any] = {
    "search_data": search_data,
    "list_collections": list_collections,
    "get_collection_fields": get_collection_fields,
    "facet_query": facet_query,
    "probe_data": probe_data,
    "find_similar_genomes": find_similar_genomes,
    "search_literature": search_literature,
}


async def execute_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    timeout_seconds: float = 30.0,
    base_url: str | None = None,
    headers: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Execute a tool by name with the given arguments.

    For API tools (search_data, facet_query, probe_data), injects
    base_url and headers into the arguments.  For search_literature,
    injects headers (auth) only.

    Delegates to the shared ``execute_tool`` for dispatch, timeout,
    and error handling.
    """
    # Inject base_url/headers for BV-BRC API tools. search_literature needs
    # headers (auth) but not base_url — it uses config.literature_rag_url.
    api_tools = {"search_data", "facet_query", "probe_data"}
    if tool_name in api_tools:
        if base_url and "base_url" not in arguments:
            arguments["base_url"] = base_url
        if headers and "headers" not in arguments:
            arguments["headers"] = headers
    elif tool_name == "search_literature":
        if headers and "headers" not in arguments:
            arguments["headers"] = headers

    return await _shared_execute_tool(
        tool_name=tool_name,
        arguments=arguments,
        dispatch_table=TOOL_DISPATCH,
        timeout_seconds=timeout_seconds,
        inject_config=False,
        inject_headers=False,
    )


def truncate_result(result: Dict[str, Any], max_chars: int = 8000) -> str:
    """Serialize a tool result to JSON, truncating if too large.

    Data-agent-specific: tries to preserve structure by reducing the
    ``results`` list progressively.
    """
    serialized = json.dumps(result, indent=2, default=str)

    if len(serialized) <= max_chars:
        return serialized

    if "results" in result and isinstance(result["results"], list):
        num_results = len(result["results"])
        truncated = dict(result)
        for n in range(num_results, 0, -1):
            truncated["results"] = result["results"][:n]
            truncated["_truncated"] = {
                "total_results": num_results,
                "shown": n,
                "note": f"Showing {n} of {num_results} results. Use more specific queries or select fewer fields to see all.",
            }
            serialized = json.dumps(truncated, indent=2, default=str)
            if len(serialized) <= max_chars:
                return serialized

    # Fallback: hard truncate
    return serialized[:max_chars] + f"\n... [TRUNCATED at {max_chars} chars]"
