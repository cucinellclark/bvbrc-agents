"""Cancel handler -- cancels a running or pending workflow."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from service_agent.models import AgentConfig, AgentResult, AgentState

# Shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "shared"))
from agent_utils import emit_progress  # noqa: E402

logger = logging.getLogger(__name__)

ProgressCallback = Any


async def handle_cancel(
    workflow_id: str,
    config: AgentConfig,
    state: AgentState,
    progress_callback: ProgressCallback | None = None,
) -> AgentResult:
    """Cancel a running or pending workflow via the workflow engine.

    Direct engine call -- no LLM involved.

    Args:
        workflow_id: The engine-issued workflow ID to cancel.
        config: Agent configuration.
        state: Current agent state (will be mutated).
        progress_callback: Optional progress callback.

    Returns:
        AgentResult with status and operation_message.
    """
    await emit_progress(progress_callback, 0, 1, f"Cancelling workflow {workflow_id}...")

    if not config.bvbrc_auth_token:
        state.status = "error"
        state.error_message = (
            "Authentication required to cancel workflows. "
            "Please log in and try again."
        )
        return state.to_result()

    try:
        _ensure_mcp_path(config)
        from common.workflow_engine_client import WorkflowEngineClient

        client = WorkflowEngineClient(base_url=config.workflow_engine_url)

        # The cancel endpoint: POST /workflows/{id}/cancel
        result = await client.cancel_workflow(workflow_id, config.bvbrc_auth_token)

        new_status = result.get("status", "cancelled")
        state.workflow_id = workflow_id
        state.status = "completed"
        state.current_phase = "done"
        state.operation_message = (
            f"Workflow **{workflow_id}** has been cancelled. "
            f"Status: **{new_status}**."
        )

        logger.info("Workflow %s cancelled: status=%s", workflow_id, new_status)

    except Exception as e:
        logger.error("Failed to cancel workflow %s: %s", workflow_id, e)
        state.status = "error"
        state.error_message = (
            f"Failed to cancel workflow {workflow_id}: "
            f"{type(e).__name__}: {e}"
        )

    await emit_progress(progress_callback, 1, 1, "Done.")
    return state.to_result()


def _ensure_mcp_path(config: AgentConfig) -> None:
    """Add the MCP server path to sys.path if needed."""
    mcp_path = config.mcp_server_path
    if mcp_path and mcp_path not in sys.path:
        sys.path.insert(0, mcp_path)
