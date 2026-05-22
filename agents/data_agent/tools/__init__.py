"""
Tool dispatcher for the BV-BRC Data Retrieval Agent.

Maps tool names (from the LLM's tool_calls) to their async implementation
functions. The `execute_tool` function handles argument unpacking, timeout
enforcement, and error wrapping.
"""

from __future__ import annotations

import asyncio
import json
import traceback
from typing import Any, Dict

from data_agent.tools.search import search_data, facet_query, probe_data
from data_agent.tools.collections import list_collections, get_collection_fields

# ---------------------------------------------------------------------------
# Dispatch table: tool name -> async callable
# ---------------------------------------------------------------------------
TOOL_DISPATCH: Dict[str, Any] = {
    "search_data": search_data,
    "list_collections": list_collections,
    "get_collection_fields": get_collection_fields,
    "facet_query": facet_query,
    "probe_data": probe_data,
    # Group tools will be added later:
    # "get_genome_group": ...,
    # "get_feature_group": ...,
}


async def execute_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    timeout_seconds: float = 30.0,
    base_url: str | None = None,
    headers: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """
    Execute a tool by name with the given arguments.

    Handles:
      - Looking up the tool function
      - Injecting base_url/headers for API tools
      - Timeout enforcement
      - Error wrapping (tool errors become structured dicts, not exceptions)

    Args:
        tool_name: Name of the tool to execute.
        arguments: Arguments from the LLM's tool_call.
        timeout_seconds: Maximum execution time.
        base_url: BV-BRC API base URL to inject.
        headers: HTTP headers to inject (e.g., auth).

    Returns:
        Dict with the tool's result, or an error dict if execution failed.
    """
    func = TOOL_DISPATCH.get(tool_name)
    if func is None:
        return {
            "error": f"Unknown tool: '{tool_name}'",
            "available_tools": sorted(TOOL_DISPATCH.keys()),
        }

    # Inject base_url and headers for API-calling tools
    api_tools = {"search_data", "facet_query", "probe_data"}
    if tool_name in api_tools:
        if base_url and "base_url" not in arguments:
            arguments["base_url"] = base_url
        if headers and "headers" not in arguments:
            arguments["headers"] = headers

    try:
        result = await asyncio.wait_for(
            func(**arguments),
            timeout=timeout_seconds,
        )
        return result

    except asyncio.TimeoutError:
        return {
            "error": f"Tool '{tool_name}' timed out after {timeout_seconds}s",
            "tool": tool_name,
            "arguments": arguments,
        }

    except TypeError as e:
        # Argument mismatch (wrong params from LLM)
        return {
            "error": f"Invalid arguments for tool '{tool_name}': {str(e)}",
            "tool": tool_name,
            "arguments": arguments,
        }

    except Exception as e:
        return {
            "error": f"Tool '{tool_name}' failed: {type(e).__name__}: {str(e)}",
            "tool": tool_name,
            "arguments": arguments,
            "traceback": traceback.format_exc(),
        }


def truncate_result(result: Dict[str, Any], max_chars: int = 8000) -> str:
    """
    Serialize a tool result to JSON, truncating if too large.

    Large results can overwhelm the LLM's context window. This function
    serializes the result and truncates it with a note if it exceeds
    max_chars.

    Args:
        result: The tool result dict.
        max_chars: Maximum characters for the serialized output.

    Returns:
        JSON string of the result, possibly truncated.
    """
    serialized = json.dumps(result, indent=2, default=str)

    if len(serialized) <= max_chars:
        return serialized

    # Truncate and add a note
    # Try to preserve structure: if there are results, truncate the list
    if "results" in result and isinstance(result["results"], list):
        num_results = len(result["results"])
        # Find how many records we can fit
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
