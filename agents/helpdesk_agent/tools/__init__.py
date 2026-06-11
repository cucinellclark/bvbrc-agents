"""
Tool dispatcher for the BV-BRC Helpdesk Agent.

Maps tool names (from the LLM's tool_calls) to their async implementation
functions. The `execute_tool` function handles argument unpacking, timeout
enforcement, and error wrapping.
"""

from __future__ import annotations

import asyncio
import json
import traceback
from typing import Any, Dict

from helpdesk_agent.tools.helpdesk import query_helpdesk
from helpdesk_agent.tools.services import list_services, get_service_schema

# ---------------------------------------------------------------------------
# Dispatch table: tool name -> async callable
# ---------------------------------------------------------------------------
TOOL_DISPATCH: Dict[str, Any] = {
    "query_helpdesk": query_helpdesk,
    "list_services": list_services,
    "get_service_schema": get_service_schema,
}


async def execute_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    timeout_seconds: float = 30.0,
    config: Any = None,
) -> Dict[str, Any]:
    """
    Execute a tool by name with the given arguments.

    Handles:
      - Looking up the tool function
      - Injecting config for tools that need it
      - Timeout enforcement
      - Error wrapping (tool errors become structured dicts, not exceptions)

    Args:
        tool_name: Name of the tool to execute.
        arguments: Arguments from the LLM's tool_call.
        timeout_seconds: Maximum execution time.
        config: AgentConfig to inject into tool calls.

    Returns:
        Dict with the tool's result, or an error dict if execution failed.
    """
    func = TOOL_DISPATCH.get(tool_name)
    if func is None:
        return {
            "error": f"Unknown tool: '{tool_name}'",
            "available_tools": sorted(TOOL_DISPATCH.keys()),
        }

    # Inject config for all tools
    if config is not None:
        arguments["config"] = config

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
            "arguments": {k: v for k, v in arguments.items() if k != "config"},
        }

    except TypeError as e:
        # Argument mismatch (wrong params from LLM)
        return {
            "error": f"Invalid arguments for tool '{tool_name}': {str(e)}",
            "tool": tool_name,
            "arguments": {k: v for k, v in arguments.items() if k != "config"},
        }

    except Exception as e:
        return {
            "error": f"Tool '{tool_name}' failed: {type(e).__name__}: {str(e)}",
            "tool": tool_name,
            "arguments": {k: v for k, v in arguments.items() if k != "config"},
            "traceback": traceback.format_exc(),
        }


def truncate_result(result: Dict[str, Any], max_chars: int = 8000) -> str:
    """
    Serialize a tool result to JSON, truncating if too large.

    Args:
        result: The tool result dict.
        max_chars: Maximum characters for the serialized output.

    Returns:
        JSON string of the result, possibly truncated.
    """
    serialized = json.dumps(result, indent=2, default=str)

    if len(serialized) <= max_chars:
        return serialized

    # Try to preserve structure: if there are results, truncate the list
    if "results" in result and isinstance(result["results"], list):
        num_results = len(result["results"])
        truncated = dict(result)
        for n in range(num_results, 0, -1):
            truncated["results"] = result["results"][:n]
            truncated["_truncated"] = {
                "total_results": num_results,
                "shown": n,
                "note": f"Showing {n} of {num_results} results.",
            }
            serialized = json.dumps(truncated, indent=2, default=str)
            if len(serialized) <= max_chars:
                return serialized

    # Fallback: hard truncate
    return serialized[:max_chars] + f"\n... [TRUNCATED at {max_chars} chars]"
