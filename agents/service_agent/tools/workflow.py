"""
Tool wrapper for workflow composition and submission.

Uses the CWL generator (M1) for composition and GoWeClient (M0) for
submission. Replaces the legacy workflow_composition_functions and
WorkflowEngineClient paths.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from service_agent.models import AgentConfig

logger = logging.getLogger(__name__)


def _ensure_mcp_path(config: AgentConfig | None = None) -> None:
    """Add the MCP server root to sys.path if not already present."""
    cfg = config or AgentConfig()
    mcp_path = cfg.mcp_server_path
    if mcp_path and mcp_path not in sys.path:
        sys.path.insert(0, mcp_path)


def _extract_user_id(headers: Optional[Dict[str, str]]) -> str:
    """Extract user_id from auth token."""
    if headers and "Authorization" in headers:
        token = headers["Authorization"]
        try:
            for part in token.split("|"):
                if part.startswith("un="):
                    return part[3:]
        except Exception:
            pass
    return "anonymous"


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def compose_workflow(
    steps: List[Dict[str, Any]],
    workflow_name: Optional[str] = None,
    config: Optional[AgentConfig] = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Bundle multiple planned service steps into a CWL workflow document.

    Uses the CWL generator from M1 to produce a CWL v1.2 $graph document
    instead of the legacy JSON manifest.

    Args:
        steps: List of planned steps, each with step_name, service_name,
               params, and optional depends_on.
        workflow_name: Optional workflow name. Auto-generated if omitted.
        config: Agent configuration.
        headers: HTTP headers with auth token.

    Returns:
        Dict with cwl_document, submission_inputs, workflow_name, and step_count.
    """
    from service_agent.cwl.generator import (
        generate_cwl_workflow,
        generate_submission_inputs,
    )
    from service_agent.models import StepPlan, ValidatedStep, WorkflowPlan

    user_id = _extract_user_id(headers)

    try:
        # Build WorkflowPlan and ValidatedStep objects from the raw dicts
        step_plans = []
        completed_steps: Dict[str, ValidatedStep] = {}

        for step_dict in steps:
            step_id = step_dict.get("step_name", step_dict.get("step_id", f"step_{len(step_plans)}"))
            service_name = step_dict.get("service_name", "unknown")
            depends_on = step_dict.get("depends_on", [])

            step_plan = StepPlan(
                step_id=step_id,
                service_name=service_name,
                intent=step_dict.get("intent", f"Run {service_name}"),
                depends_on=depends_on,
            )
            step_plans.append(step_plan)

            validated = ValidatedStep(
                step_id=step_id,
                service_name=service_name,
                api_name=step_dict.get("api_name", service_name),
                params=step_dict.get("params", {}),
                output_patterns=step_dict.get("output_patterns", {}),
                depends_on=depends_on,
            )
            completed_steps[step_id] = validated

        wf_name = workflow_name or "copilot-workflow"
        plan = WorkflowPlan(
            workflow_name=wf_name,
            description=f"Workflow with {len(steps)} step(s)",
            steps=step_plans,
        )
        plan.compute_topological_order()

        cwl_doc = generate_cwl_workflow(plan, completed_steps, user_id=user_id)
        submission_inputs = generate_submission_inputs(completed_steps, user_id=user_id)

        return {
            "cwl_document": cwl_doc,
            "submission_inputs": submission_inputs,
            "workflow_name": wf_name,
            "step_count": len(steps),
        }

    except Exception as e:
        return {
            "error": f"CWL workflow composition failed: {type(e).__name__}: {str(e)}",
        }


async def submit_workflow(
    workflow_id: str,
    config: Optional[AgentConfig] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Submit an already-registered workflow for execution via GoWe.

    Creates a GoWe submission for the given workflow_id.

    Args:
        workflow_id: The GoWe workflow ID (from register_workflow).
        config: Agent configuration (provides auth token and GoWe URL).

    Returns:
        Dict with workflow_id, submission_id, status, and message on success,
        or error details on failure.
    """
    cfg = config or AgentConfig()

    if not workflow_id:
        return {
            "error": "workflow_id is required",
            "hint": "Provide the workflow_id from a previously registered workflow.",
        }

    auth_token = cfg.bvbrc_auth_token
    if not auth_token:
        return {
            "error": "No authentication token available",
            "hint": "An auth token is required to submit workflows.",
        }

    try:
        _ensure_mcp_path(cfg)
        from common.gowe_client import GoWeClient

        client = GoWeClient(base_url=cfg.gowe_url)
        result = await client.create_submission(
            workflow_id=workflow_id,
            inputs={},
            auth_token=auth_token,
        )

        submission_id = result.get("id", "")
        logger.info(
            "Workflow %s submitted via GoWe: submission_id=%s, state=%s",
            workflow_id, submission_id, result.get("state"),
        )
        return {
            "workflow_id": workflow_id,
            "submission_id": submission_id,
            "status": result.get("state", "PENDING"),
            "message": "Workflow submitted for execution via GoWe",
        }

    except Exception as e:
        logger.error("Failed to submit workflow %s via GoWe: %s", workflow_id, e)
        return {
            "error": f"GoWe submission failed: {type(e).__name__}: {str(e)}",
            "workflow_id": workflow_id,
        }
