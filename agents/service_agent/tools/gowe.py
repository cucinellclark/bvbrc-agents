"""
Tool wrappers for GoWe workflow engine operations.

These tools let the LLM discover available workflows and their input
schemas directly from GoWe, which is the single source of truth for
what services/pipelines are available.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, Optional

from service_agent.models import AgentConfig

logger = logging.getLogger(__name__)


def _ensure_mcp_path(config: AgentConfig | None = None) -> None:
    cfg = config or AgentConfig()
    mcp_path = cfg.mcp_server_path
    if mcp_path and mcp_path not in sys.path:
        sys.path.insert(0, mcp_path)


async def list_gowe_workflows(
    config: Optional[AgentConfig] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    List all available workflows registered in the GoWe workflow engine.

    Returns a simplified list with workflow id, name, description,
    step count, and class for each workflow.
    """
    cfg = config or AgentConfig()

    try:
        _ensure_mcp_path(cfg)
        from common.gowe_client import GoWeClient

        client = GoWeClient(base_url=cfg.gowe_url)
        auth = cfg.bvbrc_auth_token or ""
        workflows, pagination = await client.list_workflows(
            auth_token=auth,
            limit=100,
        )

        # Return a clean summary the LLM can use to pick from
        result = []
        for wf in workflows or []:
            result.append(
                {
                    "id": wf.get("id"),
                    "name": wf.get("name"),
                    "description": wf.get("description", ""),
                    "step_count": wf.get("step_count", 1),
                    "class": wf.get("class", ""),
                }
            )

        return {
            "workflows": result,
            "count": len(result),
        }

    except Exception as e:
        return {
            "error": f"Failed to list GoWe workflows: {type(e).__name__}: {e}",
        }


async def get_workflow_inputs(
    workflow_id: str,
    config: Optional[AgentConfig] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Get the input schema for a specific GoWe workflow.

    Returns the list of inputs with their id, type, required flag,
    default value, and documentation.

    Args:
        workflow_id: The GoWe workflow ID (e.g., 'wf_abc123').
    """
    cfg = config or AgentConfig()

    if not workflow_id:
        return {"error": "workflow_id is required."}

    try:
        _ensure_mcp_path(cfg)
        from common.gowe_client import GoWeClient

        client = GoWeClient(base_url=cfg.gowe_url)
        auth = cfg.bvbrc_auth_token or ""
        inputs = await client.get_workflow_inputs(
            workflow_id,
            auth_token=auth,
        )

        return {
            "workflow_id": workflow_id,
            "inputs": inputs or [],
            "count": len(inputs or []),
        }

    except Exception as e:
        return {
            "error": (
                f"Failed to get inputs for workflow '{workflow_id}': "
                f"{type(e).__name__}: {e}"
            ),
        }


async def submit_gowe_job(
    workflow_id: str,
    inputs: Dict[str, Any],
    config: Optional[AgentConfig] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Submit a job to GoWe with the given workflow ID and populated inputs.

    Args:
        workflow_id: The GoWe workflow ID.
        inputs: Dict of input values matching the workflow's input schema.
    """
    cfg = config or AgentConfig()

    if not workflow_id:
        return {"error": "workflow_id is required."}

    auth = cfg.bvbrc_auth_token
    if not auth:
        return {"error": "No authentication token available."}

    try:
        _ensure_mcp_path(cfg)
        from common.gowe_client import GoWeClient

        # Clean up the inputs before submission
        from service_agent.phases.compose import _clean_record_values

        cleaned_inputs = {}
        for key, value in inputs.items():
            if value is None:
                continue
            # Strip URI prefixes from string values
            if isinstance(value, str):
                if value.startswith("ws://"):
                    value = value[len("ws://") :]
                elif value.startswith("workspace:"):
                    value = value[len("workspace:") :]
            # Clean nested records (e.g., paired_end_libs)
            if isinstance(value, list) and value and isinstance(value[0], dict):
                value = _clean_record_values(value)
            cleaned_inputs[key] = value

        client = GoWeClient(base_url=cfg.gowe_url)
        result = await client.create_submission(
            workflow_id=workflow_id,
            inputs=cleaned_inputs,
            auth_token=auth,
        )

        submission_id = result.get("id", "")
        logger.info(
            "Job submitted: workflow_id=%s, submission_id=%s, state=%s",
            workflow_id,
            submission_id,
            result.get("state"),
        )

        return {
            "workflow_id": workflow_id,
            "submission_id": submission_id,
            "status": result.get("state", "PENDING"),
            "message": "Job submitted successfully to GoWe.",
        }

    except Exception as e:
        logger.error("Failed to submit job: %s", e)
        return {
            "error": f"GoWe submission failed: {type(e).__name__}: {e}",
            "workflow_id": workflow_id,
        }
