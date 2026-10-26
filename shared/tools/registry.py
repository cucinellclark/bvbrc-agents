"""Universal tool registry for all BV-BRC agents.

Provides:
  - ``TOOL_DISPATCH``: Universal dispatch table mapping tool names to async callables.
  - ``ALL_TOOL_SCHEMAS``: All 32 tool schemas (re-exported from schemas.py).
  - ``TOOL_SCHEMA_MAP``: Name-to-schema lookup (re-exported from schemas.py).

All agents import from here. No more per-agent dispatch tables.

Usage::

    from shared.tools.registry import TOOL_DISPATCH, ALL_TOOL_SCHEMAS

    result = await execute_tool(
        tool_name="search_data",
        arguments={"collection": "genome", "query": "*"},
        dispatch_table=TOOL_DISPATCH,
        config=config,
        headers=headers,
    )
"""

from __future__ import annotations

from typing import Any, Dict

# Re-export schemas
from shared.tools.schemas import (  # noqa: F401
    ALL_TOOL_SCHEMAS,
    TOOL_SCHEMA_MAP,
)

# ---------------------------------------------------------------------------
# Shared tool implementations (always available)
# ---------------------------------------------------------------------------
from bvbrc_mcp.registry import TOOL_DISPATCH as BVBRC_TOOL_DISPATCH

from agents.planning_agent.tools.plan_tools import (
    handle_ask_clarification,
    handle_create_plan,
    handle_list_agents,
)


async def _ask_clarification_async(
    questions: list | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Async wrapper for the synchronous ask_clarification validator."""
    kwargs.pop("config", None)
    kwargs.pop("headers", None)
    args = {"questions": questions} if questions is not None else kwargs
    return handle_ask_clarification(args)


async def _create_plan_async(
    title: str | None = None,
    description: str | None = None,
    steps: list | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Async wrapper for the synchronous create_plan validator."""
    kwargs.pop("config", None)
    kwargs.pop("headers", None)
    args: Dict[str, Any] = {}
    if title is not None:
        args["title"] = title
    if description is not None:
        args["description"] = description
    if steps is not None:
        args["steps"] = steps
    args.update({k: v for k, v in kwargs.items() if k not in ("config", "headers")})
    return handle_create_plan(args)


async def _list_agents_async(
    config: Any = None,
    headers: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Async wrapper for the synchronous list_agents handler."""
    # The agent_catalog is passed via the agent loop, not via LLM arguments.
    # When called from the universal dispatcher, return the static catalog.
    return handle_list_agents([])


# ---------------------------------------------------------------------------
# Universal dispatch table: tool name -> async callable
# ---------------------------------------------------------------------------

# The BV-BRC tools (bvbrc_mcp) plus the harness's own planning tools.
TOOL_DISPATCH: Dict[str, Any] = {
    **BVBRC_TOOL_DISPATCH,
    "ask_clarification": _ask_clarification_async,
    "create_plan": _create_plan_async,
    "list_agents": _list_agents_async,
}
