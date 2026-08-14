"""Universal tool registry for all BV-BRC agents.

Provides:
  - ``TOOL_DISPATCH``: Universal dispatch table mapping tool names to async callables.
  - ``ALL_TOOL_SCHEMAS``: All 23 tool schemas (re-exported from schemas.py).
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
from shared.tools.data import search_data, facet_query, probe_data
from shared.tools.collections import list_collections, get_collection_fields
from shared.tools.workspace import workspace_browse, get_file_metadata, read_file_preview
from shared.tools.gowe import list_gowe_workflows, get_workflow_inputs, submit_gowe_job
from shared.tools.groups import create_group
from shared.tools.sra import get_sra_metadata
from shared.tools.similar_genome import find_similar_genomes
from shared.tools.literature import search_literature

# ---------------------------------------------------------------------------
# Helpdesk-specific tools
# ---------------------------------------------------------------------------
from agents.helpdesk_agent.tools.helpdesk import query_helpdesk
from agents.helpdesk_agent.tools.services import list_services, get_service_schema

# ---------------------------------------------------------------------------
# Analysis-specific tools
# ---------------------------------------------------------------------------
from agents.analysis_agent.tools import (
    _get_expected_outputs_async as get_expected_outputs,
    get_job_details,
)

# ---------------------------------------------------------------------------
# Planning-specific tools (wrapped as async for uniform interface)
# ---------------------------------------------------------------------------
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

TOOL_DISPATCH: Dict[str, Any] = {
    # Data tools
    "search_data": search_data,
    "facet_query": facet_query,
    "probe_data": probe_data,
    "list_collections": list_collections,
    "get_collection_fields": get_collection_fields,
    # Workspace tools
    "workspace_browse": workspace_browse,
    "get_file_metadata": get_file_metadata,
    "read_file_preview": read_file_preview,
    # GoWe tools
    "list_gowe_workflows": list_gowe_workflows,
    "get_workflow_inputs": get_workflow_inputs,
    "submit_gowe_job": submit_gowe_job,
    # Group tools
    "create_group": create_group,
    # SRA tools
    "get_sra_metadata": get_sra_metadata,
    # Genome similarity
    "find_similar_genomes": find_similar_genomes,
    # Literature
    "search_literature": search_literature,
    # Helpdesk tools
    "query_helpdesk": query_helpdesk,
    "list_services": list_services,
    "get_service_schema": get_service_schema,
    # Analysis tools
    "get_expected_outputs": get_expected_outputs,
    "get_job_details": get_job_details,
    # Planning tools
    "ask_clarification": _ask_clarification_async,
    "create_plan": _create_plan_async,
    "list_agents": _list_agents_async,
}
