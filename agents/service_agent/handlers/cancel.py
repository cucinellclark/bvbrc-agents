"""Cancel handler -- cancels a running or pending submission via GoWe."""

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
    """Cancel a running or pending submission via GoWe.

    Same submission_id vs workflow_id logic as the status handler:
    if state has a submission_id, use it; otherwise fall back to
    workflow_id.

    Direct engine call -- no LLM involved.

    Args:
        workflow_id: The workflow or submission ID to cancel.
        config: Agent configuration.
        state: Current agent state (will be mutated).
        progress_callback: Optional progress callback.

    Returns:
        AgentResult with status and operation_message.
    """
    # Try submission_id first, fall back to workflow_id
    sub_id = state.submission_id or workflow_id

    await emit_progress(progress_callback, 0, 1, f"Cancelling submission {sub_id}...")

    if not config.bvbrc_auth_token:
        state.status = "error"
        state.error_message = (
            "Authentication required to cancel submissions. "
            "Please log in and try again."
        )
        return state.to_result()

    try:
        _ensure_mcp_path(config)
        from common.gowe_client import GoWeClient

        client = GoWeClient(base_url=config.gowe_url)
        result = await client.cancel_submission(
            sub_id, auth_token=config.bvbrc_auth_token,
        )

        new_state = result.get("state", "CANCELLED")
        state.workflow_id = workflow_id
        state.submission_id = result.get("id", sub_id)
        state.status = "completed"
        state.current_phase = "done"
        state.operation_message = (
            f"Submission **{sub_id}** has been cancelled. "
            f"State: **{new_state}**."
        )

        logger.info("Submission %s cancelled: state=%s", sub_id, new_state)

    except Exception as e:
        logger.error("Failed to cancel submission %s: %s", sub_id, e)
        state.status = "error"
        state.error_message = (
            f"Failed to cancel submission {sub_id}: "
            f"{type(e).__name__}: {e}"
        )

    await emit_progress(progress_callback, 1, 1, "Done.")
    return state.to_result()


def _ensure_mcp_path(config: AgentConfig) -> None:
    """Add the MCP server path to sys.path if needed."""
    mcp_path = config.mcp_server_path
    if mcp_path and mcp_path not in sys.path:
        sys.path.insert(0, mcp_path)
