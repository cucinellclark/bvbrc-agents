"""
Tool wrapper for workflow composition and submission.

Translates agent tool-call arguments into the MCP server's
workflow_composition_functions module and the workflow engine client.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, List, Optional
from types import ModuleType

from service_agent.models import AgentConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MCP server import helper
# ---------------------------------------------------------------------------

_workflow_composition_functions: Optional[ModuleType] = None
_path_added: bool = False


def _ensure_path(config: AgentConfig | None = None) -> None:
    global _path_added
    if _path_added:
        return
    cfg = config or AgentConfig()
    mcp_path = cfg.mcp_server_path
    if mcp_path and mcp_path not in sys.path:
        sys.path.insert(0, mcp_path)
        _path_added = True


def _get_workflow_composition_functions(config: AgentConfig | None = None) -> ModuleType:
    global _workflow_composition_functions
    if _workflow_composition_functions is None:
        _ensure_path(config)
        from functions import workflow_composition_functions
        _workflow_composition_functions = workflow_composition_functions
    return _workflow_composition_functions


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
# Tool implementation
# ---------------------------------------------------------------------------

async def compose_workflow(
    steps: List[Dict[str, Any]],
    workflow_name: Optional[str] = None,
    config: Optional[AgentConfig] = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Bundle multiple planned service steps into a single workflow manifest.

    Args:
        steps: List of planned steps, each with step_name, service_name,
               params, and optional depends_on.
        workflow_name: Optional workflow name. Auto-generated if omitted.
        config: Agent configuration.
        headers: HTTP headers with auth token.

    Returns:
        Dict with workflow_id, status, manifest, and step_count.
    """
    wf_fn = _get_workflow_composition_functions(config)
    user_id = _extract_user_id(headers)

    try:
        result = wf_fn.compose_workflow_manifest(
            steps=steps,
            user_id=user_id,
            workflow_name=workflow_name,
        )
        return result

    except Exception as e:
        return {
            "error": f"Workflow composition failed: {type(e).__name__}: {str(e)}",
        }


async def submit_workflow(
    workflow_id: str,
    config: Optional[AgentConfig] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Submit an already-planned workflow for execution by workflow_id.

    Calls the workflow engine's submit_planned_workflow endpoint.

    Args:
        workflow_id: The engine-issued workflow ID.
        config: Agent configuration (provides auth token and engine URL).

    Returns:
        Dict with workflow_id, status, and message on success,
        or error details on failure.
    """
    cfg = config or AgentConfig()

    if not workflow_id:
        return {
            "error": "workflow_id is required",
            "hint": "Provide the workflow_id from a previously planned workflow.",
        }

    auth_token = cfg.bvbrc_auth_token
    if not auth_token:
        return {
            "error": "No authentication token available",
            "hint": "An auth token is required to submit workflows.",
        }

    try:
        _ensure_path(cfg)
        from common.workflow_engine_client import WorkflowEngineClient

        client = WorkflowEngineClient(base_url=cfg.workflow_engine_url)
        result = await client.submit_planned_workflow(workflow_id, auth_token)

        logger.info("Workflow %s submitted successfully: status=%s", workflow_id, result.get("status"))
        return {
            "workflow_id": result.get("workflow_id", workflow_id),
            "status": result.get("status", "pending"),
            "message": "Workflow submitted for execution",
        }

    except Exception as e:
        logger.error("Failed to submit workflow %s: %s", workflow_id, e)
        return {
            "error": f"Workflow submission failed: {type(e).__name__}: {str(e)}",
            "workflow_id": workflow_id,
        }
