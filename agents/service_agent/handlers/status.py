"""Status handler -- checks the status of a workflow."""

from __future__ import annotations

import json
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
    """Check the status of a workflow via the workflow engine.

    Direct engine call -- no LLM involved.

    Args:
        workflow_id: The engine-issued workflow ID to check.
        config: Agent configuration.
        state: Current agent state (will be mutated).
        progress_callback: Optional progress callback.

    Returns:
        AgentResult with status and operation_message.
    """
    await emit_progress(progress_callback, 0, 1, f"Checking status of workflow {workflow_id}...")

    try:
        _ensure_mcp_path(config)
        from common.workflow_engine_client import WorkflowEngineClient

        client = WorkflowEngineClient(base_url=config.workflow_engine_url)
        result = await client.get_workflow_status(workflow_id)

        wf_status = result.get("status", "unknown")
        state.workflow_id = workflow_id
        state.status = "completed"
        state.current_phase = "done"

        # Build a human-readable status message
        lines = [f"Workflow **{workflow_id}** status: **{wf_status}**"]

        # Include step-level status if available
        steps = result.get("steps") or result.get("step_statuses")
        if steps and isinstance(steps, list):
            lines.append("")
            lines.append("**Steps:**")
            for step in steps:
                step_name = step.get("step_id") or step.get("name", "?")
                step_status = step.get("status", "?")
                lines.append(f"  - {step_name}: {step_status}")

        # Include timing if available
        created = result.get("created_at")
        updated = result.get("updated_at")
        if created:
            lines.append(f"\nCreated: {created}")
        if updated:
            lines.append(f"Last updated: {updated}")

        state.operation_message = "\n".join(lines)
        logger.info("Workflow %s status: %s", workflow_id, wf_status)

    except Exception as e:
        logger.error("Failed to get status for workflow %s: %s", workflow_id, e)
        state.status = "error"
        state.error_message = (
            f"Failed to check status of workflow {workflow_id}: "
            f"{type(e).__name__}: {e}"
        )

    await emit_progress(progress_callback, 1, 1, "Done.")
    return state.to_result()


def _ensure_mcp_path(config: AgentConfig) -> None:
    """Add the MCP server path to sys.path if needed."""
    mcp_path = config.mcp_server_path
    if mcp_path and mcp_path not in sys.path:
        sys.path.insert(0, mcp_path)
