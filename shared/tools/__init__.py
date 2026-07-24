"""
Shared tool infrastructure for BV-BRC agents.

Provides:
  - ``execute_tool``: Generic tool dispatcher with timeout and error handling.
  - ``truncate_result``: Base result-to-JSON serializer with smart truncation.

Individual agents may override ``truncate_result`` (e.g. the workspace
agent's version does representative sampling across types).
"""

from __future__ import annotations

import asyncio
import json
import traceback
from typing import Any, Dict


async def execute_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    dispatch_table: Dict[str, Any],
    timeout_seconds: float = 30.0,
    config: Any = None,
    headers: Dict[str, str] | None = None,
    inject_config: bool = True,
    inject_headers: bool = True,
) -> Dict[str, Any]:
    """Execute a tool by name from the given dispatch table.

    Handles:
      - Looking up the tool function in *dispatch_table*
      - Optionally injecting *config* and *headers* into the call arguments
      - Timeout enforcement
      - Error wrapping (tool errors become structured dicts, not exceptions)

    Args:
        tool_name: Name of the tool to execute.
        arguments: Arguments from the LLM's ``tool_call``.
        dispatch_table: Mapping of ``{tool_name: async_callable}``.
        timeout_seconds: Maximum execution time.
        config: Agent config to inject (any Pydantic model).
        headers: HTTP headers to inject (e.g. auth).
        inject_config: Whether to inject ``config`` into arguments.
        inject_headers: Whether to inject ``headers`` into arguments.

    Returns:
        Dict with the tool's result, or an error dict if execution failed.
    """
    func = dispatch_table.get(tool_name)
    if func is None:
        return {
            "error": f"Unknown tool: '{tool_name}'",
            "available_tools": sorted(dispatch_table.keys()),
        }

    # Inject config and headers if the tool accepts them
    if inject_config and config is not None and "config" not in arguments:
        arguments["config"] = config
    if inject_headers and headers is not None and "headers" not in arguments:
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
            "arguments": _safe_args(arguments),
        }

    except TypeError as e:
        return {
            "error": f"Invalid arguments for tool '{tool_name}': {str(e)}",
            "tool": tool_name,
            "arguments": _safe_args(arguments),
        }

    except Exception as e:
        return {
            "error": f"Tool '{tool_name}' failed: {type(e).__name__}: {str(e)}",
            "tool": tool_name,
            "arguments": _safe_args(arguments),
            "traceback": traceback.format_exc(),
        }


def truncate_result(result: Dict[str, Any], max_chars: int = 8000) -> str:
    """Serialize a tool result to JSON, truncating if too large.

    Tries to preserve structure by progressively removing items from
    list-valued keys (``results``, ``records``, ``files``, ``ids``,
    ``items``).  Falls back to hard truncation.

    Args:
        result: The tool result dict.
        max_chars: Maximum characters for the serialized output.

    Returns:
        JSON string of the result, possibly truncated.
    """
    serialized = json.dumps(result, indent=2, default=str)

    if len(serialized) <= max_chars:
        return serialized

    # Try to preserve structure by truncating list-valued keys
    for list_key in ("results", "records", "files", "ids", "items"):
        if list_key in result and isinstance(result[list_key], list):
            num_items = len(result[list_key])
            truncated = dict(result)
            for n in range(num_items, 0, -1):
                truncated[list_key] = result[list_key][:n]
                truncated["_truncated"] = {
                    "total": num_items,
                    "shown": n,
                    "note": f"Showing {n} of {num_items} items.",
                }
                serialized = json.dumps(truncated, indent=2, default=str)
                if len(serialized) <= max_chars:
                    return serialized

    # Fallback: hard truncate
    return serialized[:max_chars] + f"\n... [TRUNCATED at {max_chars} chars]"


def _safe_args(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Strip config/headers from arguments for safe error reporting."""
    return {k: v for k, v in arguments.items() if k not in ("config", "headers")}
