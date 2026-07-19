"""Planning agent tool execution dispatcher."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


async def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    agent_catalog: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Dispatch a tool call to the appropriate handler.

    Args:
        tool_name: Name of the tool to execute.
        arguments: Tool arguments from the LLM.
        agent_catalog: List of available agents (for list_agents tool).

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
        if tool_name == "ask_clarification":
            result = handle_ask_clarification(arguments)
        elif tool_name == "create_plan":
            result = handle_create_plan(arguments)
        elif tool_name == "list_agents":
            result = handle_list_agents(agent_catalog or [])
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        duration_ms = (time.time() - start) * 1000
        logger.info(f"Tool {tool_name} completed in {duration_ms:.0f}ms")
        return result

    except Exception as e:
        logger.exception(f"Tool {tool_name} failed: {e}")
        return {"error": str(e)}
