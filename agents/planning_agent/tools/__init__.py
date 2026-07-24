"""
Planning agent tool execution dispatcher.

Handles the planning-specific tools (ask_clarification, create_plan,
list_agents) plus shared reconnaissance tools (workspace_browse,
search_data, list_gowe_workflows) for gathering context during planning.
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from typing import Any, Dict

from shared.tools.workspace import workspace_browse, get_file_metadata
from shared.tools.data import search_data
from shared.tools.gowe import list_gowe_workflows
from shared.tools.similar_genome import find_similar_genomes
from shared.tools.literature import search_literature

logger = logging.getLogger(__name__)

# Shared tools available to the planning agent for reconnaissance
_SHARED_TOOL_DISPATCH: Dict[str, Any] = {
    "workspace_browse": workspace_browse,
    "get_file_metadata": get_file_metadata,
    "search_data": search_data,
    "list_gowe_workflows": list_gowe_workflows,
    "find_similar_genomes": find_similar_genomes,
    "search_literature": search_literature,
}


async def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    agent_catalog: list[dict[str, Any]] | None = None,
    config: Any = None,
    headers: Dict[str, str] | None = None,
) -> dict[str, Any]:
    """Dispatch a tool call to the appropriate handler.

    Planning-specific tools (ask_clarification, create_plan, list_agents)
    are handled locally.  Shared tools (workspace_browse, search_data,
    list_gowe_workflows) are dispatched to the shared tool library.

    Args:
        tool_name: Name of the tool to execute.
        arguments: Tool arguments from the LLM.
        agent_catalog: List of available agents (for list_agents tool).
        config: Agent configuration (injected into shared tools).
        headers: HTTP headers with auth token (injected into shared tools).

    Returns:
        Tool result dict.
    """
    from planning_agent.tools.plan_tools import (
        handle_ask_clarification,
        handle_create_plan,
        handle_list_agents,
    )

    start = time.time()
    try:
        # Planning-specific tools
        if tool_name == "ask_clarification":
            result = handle_ask_clarification(arguments)
        elif tool_name == "create_plan":
            result = handle_create_plan(arguments)
        elif tool_name == "list_agents":
            result = handle_list_agents(agent_catalog or [])
        # Shared reconnaissance tools
        elif tool_name in _SHARED_TOOL_DISPATCH:
            func = _SHARED_TOOL_DISPATCH[tool_name]
            if config is not None and "config" not in arguments:
                arguments["config"] = config
            if headers is not None and "headers" not in arguments:
                arguments["headers"] = headers
            result = await asyncio.wait_for(func(**arguments), timeout=30.0)
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        duration_ms = (time.time() - start) * 1000
        logger.info(f"Tool {tool_name} completed in {duration_ms:.0f}ms")
        return result

    except asyncio.TimeoutError:
        return {
            "error": f"Tool '{tool_name}' timed out after 30s",
            "tool": tool_name,
        }

    except Exception as e:
        logger.exception(f"Tool {tool_name} failed: {e}")
        return {"error": str(e), "traceback": traceback.format_exc()}
