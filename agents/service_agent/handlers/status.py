"""Status handler -- checks the status of a submission via GoWe."""

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


async def handle_status(
    workflow_id: str,
    config: AgentConfig,
    state: AgentState,
    progress_callback: ProgressCallback | None = None,
) -> AgentResult:
    """Check the status of a submission via GoWe.

    GoWe status checks use submission_id. If the state has a submission_id,
    use it; otherwise, fall back to using workflow_id as a submission_id
    (the classifier may have stored it that way).

    GoWe returns a three-level state hierarchy:
    - Submission: PENDING/RUNNING/COMPLETED/FAILED/CANCELLED
    - StepInstances: WAITING/READY/DISPATCHED/RUNNING/COMPLETED/FAILED/SKIPPED
    - Tasks: PENDING/SCHEDULED/QUEUED/RUNNING/SUCCESS/FAILED/SKIPPED

    Direct engine call -- no LLM involved.

    Args:
        workflow_id: The workflow or submission ID to check.
        config: Agent configuration.
        state: Current agent state (will be mutated).
        progress_callback: Optional progress callback.

    Returns:
        AgentResult with status and operation_message.
    """
    # Try submission_id first, fall back to workflow_id
    sub_id = state.submission_id or workflow_id

    await emit_progress(progress_callback, 0, 1, f"Checking status of submission {sub_id}...")

    try:
        _ensure_mcp_path(config)
        from common.gowe_client import GoWeClient

        client = GoWeClient(base_url=config.gowe_url)
        result = await client.get_submission(
            sub_id, auth_token=config.bvbrc_auth_token,
        )

        sub_state = result.get("state", "UNKNOWN")
        state.workflow_id = workflow_id
        state.submission_id = result.get("id", sub_id)
        state.status = "completed"
        state.current_phase = "done"

        # Build a human-readable status message with three-level hierarchy
        lines = [f"Submission **{sub_id}** status: **{sub_state}**"]

        # Include step-level status if available
        step_instances = result.get("step_instances") or result.get("steps")
        if step_instances and isinstance(step_instances, list):
            lines.append("")
            lines.append("**Steps:**")
            for step in step_instances:
                step_name = (
                    step.get("step_id")
                    or step.get("name")
                    or step.get("cwl_step_id", "?")
                )
                step_status = step.get("state") or step.get("status", "?")
                lines.append(f"  - {step_name}: {step_status}")

                # Include task-level detail if available
                tasks = step.get("tasks") or []
                for task in tasks:
                    task_id = task.get("id") or task.get("task_id", "?")
                    task_state = task.get("state", "?")
                    executor = task.get("executor_type", "")
                    executor_str = f" (executor: {executor})" if executor else ""
                    lines.append(
                        f"    Task {task_id}: {task_state}{executor_str}"
                    )

        # Include timing if available
        created = result.get("created_at")
        updated = result.get("updated_at")
        if created:
            lines.append(f"\nCreated: {created}")
        if updated:
            lines.append(f"Last updated: {updated}")

        state.operation_message = "\n".join(lines)
        logger.info("Submission %s status: %s", sub_id, sub_state)

    except Exception as e:
        logger.error("Failed to get status for submission %s: %s", sub_id, e)
        state.status = "error"
        state.error_message = (
            f"Failed to check status of submission {sub_id}: "
            f"{type(e).__name__}: {e}"
        )

    await emit_progress(progress_callback, 1, 1, "Done.")
    return state.to_result()


def _ensure_mcp_path(config: AgentConfig) -> None:
    """Add the MCP server path to sys.path if needed."""
    mcp_path = config.mcp_server_path
    if mcp_path and mcp_path not in sys.path:
        sys.path.insert(0, mcp_path)
