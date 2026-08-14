"""
Tool dispatcher for the BV-BRC Helpdesk Agent.

Maps tool names (from the LLM's tool_calls) to their async implementation
functions.  Includes helpdesk-specific tools (query_helpdesk, list_services,
get_service_schema) plus shared tools for workspace/data/gowe reconnaissance.
"""

from __future__ import annotations

from typing import Any, Dict

from shared.tools import execute_tool as _shared_execute_tool, truncate_result
from shared.tools.workspace import (
    workspace_browse,
    get_file_metadata,
    read_file_preview,
)
from shared.tools.data import search_data
from shared.tools.gowe import list_gowe_workflows, get_workflow_inputs
from shared.tools.similar_genome import find_similar_genomes
from shared.tools.literature import search_literature

from helpdesk_agent.tools.helpdesk import query_helpdesk
from helpdesk_agent.tools.services import list_services, get_service_schema

# ---------------------------------------------------------------------------
# Dispatch table: tool name -> async callable
# ---------------------------------------------------------------------------
TOOL_DISPATCH: Dict[str, Any] = {
    # Helpdesk-specific tools
    "query_helpdesk": query_helpdesk,
    "list_services": list_services,
    "get_service_schema": get_service_schema,
    # Shared workspace tools
    "workspace_browse": workspace_browse,
    "get_file_metadata": get_file_metadata,
    "read_file_preview": read_file_preview,
    # Shared data tools
    "search_data": search_data,
    # Shared GoWe tools
    "list_gowe_workflows": list_gowe_workflows,
    "get_workflow_inputs": get_workflow_inputs,
    # Shared similar genome finder
    "find_similar_genomes": find_similar_genomes,
    # Shared literature RAG
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
