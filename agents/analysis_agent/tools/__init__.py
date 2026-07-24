"""
Tool dispatcher for the BV-BRC Analysis Agent.

Maps tool names (from the LLM's tool_calls) to their async implementation
functions. Reuses shared workspace and data tools, adds local
get_expected_outputs and get_job_details tools.
"""

from __future__ import annotations

import asyncio
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

from shared.tools import execute_tool as _shared_execute_tool
from shared.tools.workspace import (
    workspace_browse,
    get_file_metadata,
    read_file_preview,
)
from shared.tools.data import search_data
from shared.tools.similar_genome import find_similar_genomes
from shared.tools.literature import search_literature
from shared.tools._mcp_imports import get_json_rpc
from analysis_agent.output_knowledge import (
    get_expected_outputs as _get_expected_outputs,
)

# Import the workspace agent's sophisticated truncate_result
from workspace_agent.tools import truncate_result  # noqa: E402


async def _get_expected_outputs_async(
    service_name: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Async wrapper for the local get_expected_outputs lookup.

    This is a local lookup (no API call), but we wrap it as async to match
    the signature expected by execute_tool.
    """
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

    task_ids_str = [str(tid) for tid in task_ids]

    from shared.tools._mcp_imports import get_service_functions

    service_fn = get_service_functions()

    api = _get_service_api()
    return await service_fn.query_tasks(
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
    "search_data": search_data,
    "get_expected_outputs": _get_expected_outputs_async,
    "get_job_details": get_job_details,
    "find_similar_genomes": find_similar_genomes,
    "search_literature": search_literature,
}


async def execute_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    timeout_seconds: float = 30.0,
    config: Any = None,
    headers: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Execute a tool by name with the given arguments.

    Delegates to the shared ``execute_tool`` with this agent's dispatch table.
    """
    return await _shared_execute_tool(
        tool_name=tool_name,
        arguments=arguments,
        dispatch_table=TOOL_DISPATCH,
        timeout_seconds=timeout_seconds,
        config=config,
        headers=headers,
    )
