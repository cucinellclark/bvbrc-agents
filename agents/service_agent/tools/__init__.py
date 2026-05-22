"""
Tool dispatcher for the BV-BRC Service Agent v2.

Maps tool names (from the LLM's tool_calls) to their async implementation
functions. The `execute_tool` function handles argument unpacking, timeout
enforcement, and error wrapping.
"""

from __future__ import annotations

import asyncio
import json
import traceback
from typing import Any, Dict

from service_agent.tools.service import list_services, get_service_schema, plan_service
from service_agent.tools.workspace import workspace_browse, read_file_info
from service_agent.tools.data import search_data
from service_agent.tools.groups import get_genome_group, get_feature_group
from service_agent.tools.workflow import compose_workflow, submit_workflow
from service_agent.tools.sra import get_sra_metadata
from service_agent.tools.plan_tools import create_workflow_plan

# ---------------------------------------------------------------------------
# Dispatch table: tool name -> async callable
# ---------------------------------------------------------------------------
TOOL_DISPATCH: Dict[str, Any] = {
    # Phase 1 tools
    "create_workflow_plan": create_workflow_plan,
    "list_services": list_services,
    "get_sra_metadata": get_sra_metadata,
    # Phase 2 tools
    "get_service_schema": get_service_schema,
    "plan_service": plan_service,
    "workspace_browse": workspace_browse,
    "read_file_info": read_file_info,
    "search_data": search_data,
    "get_genome_group": get_genome_group,
    "get_feature_group": get_feature_group,
    # Phase 3 (programmatic, but keeping in dispatch for completeness)
    "compose_workflow": compose_workflow,
    # Submission tool (used when user confirms a planned workflow)
    "submit_workflow": submit_workflow,
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

    # Inject config for all tools
    if config is not None and "config" not in arguments:
        arguments["config"] = config

    # Inject headers for tools that need auth
    auth_tools = {
        "plan_service", "compose_workflow", "workspace_browse",
        "read_file_info", "search_data", "get_genome_group",
        "get_feature_group", "submit_workflow",
    }
    if tool_name in auth_tools and headers is not None:
        if "headers" not in arguments:
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

    # Try to preserve structure: if there are results/records, truncate the list
    for list_key in ("results", "records", "files", "ids"):
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
