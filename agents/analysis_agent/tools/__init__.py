"""
Tool dispatcher for the BV-BRC Analysis Agent.

Maps tool names (from the LLM's tool_calls) to their async implementation
functions. Reuses workspace tools for browsing/reading, adds local
get_expected_outputs and get_job_details tools.

The `execute_tool` function handles argument unpacking, timeout enforcement,
and error wrapping. The `truncate_result` function is imported from the
workspace agent's tools package.
"""

from __future__ import annotations

import asyncio
import json
import traceback
from typing import Any, Dict

from workspace_agent.tools.browse import workspace_browse, get_file_metadata
from workspace_agent.tools.read import read_file_preview
from workspace_agent.tools._mcp_imports import get_json_rpc
from analysis_agent.output_knowledge import get_expected_outputs as _get_expected_outputs


async def _get_expected_outputs_async(
    service_name: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Async wrapper for the local get_expected_outputs lookup.

    This is a local lookup (no API call), but we wrap it as async to match
    the signature expected by execute_tool.
    """
    # Ignore injected config/headers -- this is a local lookup
    kwargs.pop("config", None)
    kwargs.pop("headers", None)
    return _get_expected_outputs(service_name)


# ---------------------------------------------------------------------------
# get_job_details: resolve task IDs to job metadata (status, output path, etc.)
# ---------------------------------------------------------------------------

_service_api_instance = None

_APP_SERVICE_URL = "https://p3.theseed.org/services/app_service"


def _get_service_api() -> Any:
    """Create or reuse a JsonRpcCaller instance for the BV-BRC app service."""
    global _service_api_instance
    if _service_api_instance is None:
        json_rpc_mod = get_json_rpc()
        _service_api_instance = json_rpc_mod.JsonRpcCaller(
            service_url=_APP_SERVICE_URL,
            timeout=30,
        )
    return _service_api_instance


async def get_job_details(
    task_ids: list[str],
    stdout: bool = False,
    stderr: bool = False,
    config: Any = None,
    headers: Dict[str, str] | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Query BV-BRC job details by task ID.

    Calls AppService.query_tasks via JSON-RPC to retrieve job metadata
    including status, parameters (with output_path/output_file), and
    optionally stdout/stderr logs.
    """
    # Extract auth token from headers
    token = None
    if headers and "Authorization" in headers:
        token = headers["Authorization"]

    if not token:
        return {
            "error": "Authentication required to query job details.",
            "errorType": "AUTHENTICATION_ERROR",
            "source": "bvbrc-service",
        }

    if not task_ids or not isinstance(task_ids, list):
        return {
            "error": "task_ids (list) parameter is required.",
            "errorType": "INVALID_PARAMETERS",
            "source": "bvbrc-service",
        }

    # Ensure all IDs are strings
    task_ids_str = [str(tid) for tid in task_ids]

    # Import and call the MCP server's query_tasks function
    import sys
    from pathlib import Path

    mcp_path = str(Path(__file__).resolve().parent.parent.parent.parent / "mcp_server")
    if mcp_path not in sys.path:
        sys.path.insert(0, mcp_path)

    from functions.service_functions import query_tasks

    api = _get_service_api()
    return await query_tasks(
        api=api,
        token=token,
        params={"task_ids": task_ids_str},
        fetch_stdout=stdout,
        fetch_stderr=stderr,
    )


# ---------------------------------------------------------------------------
# Dispatch table: tool name -> async callable
# ---------------------------------------------------------------------------
TOOL_DISPATCH: Dict[str, Any] = {
    "workspace_browse": workspace_browse,
    "get_file_metadata": get_file_metadata,
    "read_file_preview": read_file_preview,
    "get_expected_outputs": _get_expected_outputs_async,
    "get_job_details": get_job_details,
}


async def execute_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    timeout_seconds: float = 30.0,
    config: Any = None,
    headers: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """
    Execute a tool by name with the given arguments.

    Handles:
      - Looking up the tool function
      - Injecting config/headers for tools that need them
      - Timeout enforcement
      - Error wrapping (tool errors become structured dicts, not exceptions)
    """
    func = TOOL_DISPATCH.get(tool_name)
    if func is None:
        return {
            "error": f"Unknown tool: '{tool_name}'",
            "available_tools": sorted(TOOL_DISPATCH.keys()),
        }

    # Inject config and headers for all workspace tools
    if config is not None and "config" not in arguments:
        arguments["config"] = config
    if headers is not None and "headers" not in arguments:
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
            "arguments": {
                k: v for k, v in arguments.items()
                if k not in ("config", "headers")
            },
        }

    except TypeError as e:
        # Argument mismatch (wrong params from LLM)
        return {
            "error": f"Invalid arguments for tool '{tool_name}': {str(e)}",
            "tool": tool_name,
            "arguments": {
                k: v for k, v in arguments.items()
                if k not in ("config", "headers")
            },
        }

    except Exception as e:
        return {
            "error": f"Tool '{tool_name}' failed: {type(e).__name__}: {str(e)}",
            "tool": tool_name,
            "arguments": {
                k: v for k, v in arguments.items()
                if k not in ("config", "headers")
            },
            "traceback": traceback.format_exc(),
        }


# Import truncate_result from workspace_agent tools to avoid duplication
from workspace_agent.tools import truncate_result  # noqa: E402
