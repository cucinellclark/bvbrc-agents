"""
Tool dispatcher for the BV-BRC Service Agent.

Maps tool names (from the LLM's tool_calls) to their async implementation
functions.  All tool implementations come from the shared tool library.
"""

from __future__ import annotations

from typing import Any, Dict

from shared.tools import execute_tool as _shared_execute_tool, truncate_result
from shared.tools.gowe import list_gowe_workflows, get_workflow_inputs, submit_gowe_job
from shared.tools.workspace import (
    workspace_browse,
    get_file_metadata,
    read_file_preview,
)
from shared.tools.data import search_data
from shared.tools.groups import create_group
from shared.tools.sra import get_sra_metadata
from shared.tools.similar_genome import find_similar_genomes
from shared.tools.literature import search_literature

# ---------------------------------------------------------------------------
# Dispatch table: tool name -> async callable
# ---------------------------------------------------------------------------
TOOL_DISPATCH: Dict[str, Any] = {
    # GoWe workflow tools (primary flow)
    "list_gowe_workflows": list_gowe_workflows,
    "get_workflow_inputs": get_workflow_inputs,
    "submit_gowe_job": submit_gowe_job,
    # Context-gathering tools (shared)
    "workspace_browse": workspace_browse,
    "read_file_info": get_file_metadata,  # Aliased for backward compat with tool schemas
    "search_data": search_data,
    "create_group": create_group,
    "get_sra_metadata": get_sra_metadata,
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
