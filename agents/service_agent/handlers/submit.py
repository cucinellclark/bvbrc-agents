"""Submit handler -- submits a planned workflow for execution via GoWe."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from service_agent.models import AgentConfig, AgentResult, AgentState

# Shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "shared"))
# Also add repo root so `shared` package imports work
if str(Path(__file__).resolve().parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from agent_utils import emit_progress  # noqa: E402

logger = logging.getLogger(__name__)

ProgressCallback = Any


async def handle_submit(
    workflow_id: str,
    config: AgentConfig,
    state: AgentState,
    progress_callback: ProgressCallback | None = None,
) -> AgentResult:
    """Submit an already-planned workflow to GoWe for execution.

    Creates a GoWe submission using the workflow_id (from registration)
    and the submission_inputs (from Phase 3 compose). Direct engine call
    -- no LLM involved.

    Args:
        workflow_id: The GoWe-issued workflow ID to submit.
        config: Agent configuration (provides auth token and GoWe URL).
        state: Current agent state (will be mutated).
        progress_callback: Optional progress callback.

    Returns:
        AgentResult with status and operation_message.
    """
    await emit_progress(
        progress_callback, 0, 1, f"Submitting workflow {workflow_id}..."
    )

    if not config.bvbrc_auth_token:
        state.status = "error"
        state.error_message = (
            "Authentication required to submit workflows. Please log in and try again."
        )
        return state.to_result()

    try:
        _ensure_mcp_path(config)
        from common.gowe_client import GoWeClient

        client = GoWeClient(base_url=config.gowe_url)

        # Get submission inputs -- prefer state, fall back to regenerating
        inputs = state.submission_inputs
        if not inputs and state.completed_steps:
            from service_agent.cwl.generator import generate_submission_inputs

            inputs = generate_submission_inputs(state.completed_steps)

        result = await client.create_submission(
            workflow_id=workflow_id,
            inputs=inputs or {},
            auth_token=config.bvbrc_auth_token,
        )

        submission_id = result.get("id", "")
        status = result.get("state", "PENDING")
        state.workflow_id = workflow_id
        state.submission_id = submission_id
        state.status = "completed"
        state.current_phase = "done"
        state.operation_message = (
            f"Your job has been submitted and is being processed. "
            f"You will be notified when it completes."
        )

        logger.info(
            "Workflow %s submitted: submission_id=%s, state=%s",
            workflow_id,
            submission_id,
            status,
        )

    except Exception as e:
        logger.error("Failed to submit workflow %s: %s", workflow_id, e)
        state.status = "error"
        state.error_message = (
            f"Failed to submit workflow {workflow_id}: {type(e).__name__}: {e}"
        )

    await emit_progress(progress_callback, 1, 1, "Done.")
    return state.to_result()


def _ensure_mcp_path(config: AgentConfig) -> None:
    """Add the MCP server path to sys.path if needed."""
    mcp_path = config.mcp_server_path
    if mcp_path and mcp_path not in sys.path:
        sys.path.insert(0, mcp_path)
