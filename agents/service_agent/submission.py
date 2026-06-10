"""Workflow submission utilities for the BV-BRC Service Agent v2.

This module lives *outside* the three-phase planning pipeline.  It is a
thin async wrapper around GoWe's REST API that the caller / orchestrator
invokes after planning is complete and the user has confirmed they want
to submit.

Typical usage::

    from service_agent.submission import submit_workflow, get_workflow_status

    result = await run_agent(query, config)
    if result.status == "completed":
        submission = await submit_workflow(result, config)
        print(submission)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

from service_agent.models import AgentConfig, AgentResult, SubmissionResult


# ---------------------------------------------------------------------------
# MCP path helper
# ---------------------------------------------------------------------------

def _ensure_mcp_path(config: AgentConfig) -> None:
    """Add the MCP server path to sys.path if needed."""
    mcp_path = config.mcp_server_path
    if mcp_path and mcp_path not in sys.path:
        sys.path.insert(0, mcp_path)


# ---------------------------------------------------------------------------
# Manifest / CWL document extraction
# ---------------------------------------------------------------------------

class SubmissionError(Exception):
    """Raised when GoWe rejects or cannot process a submission."""

    def __init__(
        self,
        message: str,
        error_type: str = "UNKNOWN_ERROR",
        status_code: int | None = None,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code


def extract_workflow_definition(agent_result: AgentResult) -> Dict[str, Any]:
    """Extract the CWL document from an AgentResult.

    The compose phase now stores a CWL v1.2 $graph document in the
    manifest under the ``"cwl_document"`` key. This function extracts
    it for submission to GoWe.

    For backward compatibility, also handles the legacy format where
    a ``"manifest"`` key contains the old WorkflowDefinition.

    Returns:
        A CWL document dict suitable for GoWe registration.

    Raises:
        ValueError: If the AgentResult has no usable CWL document or manifest.
    """
    outer = agent_result.manifest
    if not outer or not isinstance(outer, dict):
        raise ValueError("AgentResult has no manifest to submit.")

    # New format: CWL document under "cwl_document" key
    cwl_doc = outer.get("cwl_document")
    if isinstance(cwl_doc, dict) and "$graph" in cwl_doc:
        return cwl_doc

    # Also check if agent_result carries the CWL document directly
    if agent_result.cwl_document and isinstance(agent_result.cwl_document, dict):
        return agent_result.cwl_document

    # Legacy fallback: the outer dict itself might be a WorkflowDefinition
    if "steps" in outer and "workflow_name" in outer:
        return outer

    # Legacy fallback: nested manifest key
    definition = outer.get("manifest")
    if isinstance(definition, dict) and "steps" in definition:
        return definition

    raise ValueError(
        "Cannot locate a valid CWL document or WorkflowDefinition in "
        "AgentResult.manifest. Expected a 'cwl_document' key with a "
        "CWL $graph document."
    )


def _extract_submission_inputs(agent_result: AgentResult) -> Dict[str, Any]:
    """Extract submission inputs from an AgentResult.

    Returns:
        Submission inputs dict, or empty dict if not available.
    """
    outer = agent_result.manifest
    if isinstance(outer, dict):
        inputs = outer.get("submission_inputs")
        if isinstance(inputs, dict):
            return inputs
    return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def submit_workflow(
    agent_result: AgentResult,
    config: AgentConfig,
) -> SubmissionResult:
    """Submit a completed AgentResult to GoWe.

    Extracts the CWL document from the manifest, registers it with GoWe
    if needed, then creates a submission.

    Args:
        agent_result: A completed AgentResult with a manifest.
        config: AgentConfig with ``gowe_url`` and ``bvbrc_auth_token``.

    Returns:
        SubmissionResult with the GoWe-assigned workflow_id, submission_id,
        and status.

    Raises:
        ValueError: If the AgentResult has no manifest or no auth token.
        SubmissionError: If GoWe rejects the submission.
    """
    if not config.bvbrc_auth_token:
        raise ValueError(
            "Cannot submit workflow: no auth token configured. "
            "Provide a BV-BRC auth token via --auth-token-file or config."
        )

    _ensure_mcp_path(config)
    from common.gowe_client import GoWeClient, GoWeError

    client = GoWeClient(base_url=config.gowe_url)
    base = config.gowe_url.rstrip("/")

    try:
        # If we already have a workflow_id from registration, use it
        workflow_id = agent_result.workflow_id
        cwl_doc = extract_workflow_definition(agent_result)

        if not workflow_id:
            # Register the CWL document first
            print(
                f"Registering CWL workflow with GoWe: {base}",
                file=sys.stderr,
            )
            reg_result = await client.register_workflow(
                cwl_doc, config.bvbrc_auth_token,
            )
            workflow_id = reg_result.get("id", "unknown")

        # Get submission inputs
        inputs = _extract_submission_inputs(agent_result)

        # Create the submission
        print(
            f"Creating GoWe submission for workflow {workflow_id}",
            file=sys.stderr,
        )
        sub_result = await client.create_submission(
            workflow_id=workflow_id,
            inputs=inputs,
            auth_token=config.bvbrc_auth_token,
        )

        submission_id = sub_result.get("id", "")
        status = sub_result.get("state", "PENDING")

        return SubmissionResult(
            workflow_id=workflow_id,
            submission_id=submission_id,
            status=status,
            engine_url=base,
            status_url=f"{base}/api/v1/submissions/{submission_id}",
        )

    except GoWeError as exc:
        return SubmissionResult(
            workflow_id=agent_result.workflow_id or "",
            submission_id=None,
            status="submission_failed",
            engine_url=base,
            status_url="",
            error=str(exc),
        )


async def validate_workflow(
    agent_result: AgentResult,
    config: AgentConfig,
) -> Dict[str, Any]:
    """Validate a completed AgentResult's CWL document via GoWe dry_run.

    Extracts the CWL document, registers it (if not already registered),
    then performs a dry run to validate inputs and DAG.

    Args:
        agent_result: A completed AgentResult with a manifest.
        config: AgentConfig with GoWe URL/timeout and auth token.

    Returns:
        Dict with ``valid`` (bool), ``dry_run`` (bool), ``errors``,
        ``warnings``, ``execution_order``, etc.
    """
    _ensure_mcp_path(config)
    from common.gowe_client import GoWeClient, GoWeError

    client = GoWeClient(base_url=config.gowe_url)
    auth_token = config.bvbrc_auth_token or ""

    try:
        cwl_doc = extract_workflow_definition(agent_result)
        workflow_id = agent_result.workflow_id

        if not workflow_id:
            # Register first to get a workflow_id for dry_run
            reg_result = await client.register_workflow(cwl_doc, auth_token)
            workflow_id = reg_result.get("id", "unknown")

        inputs = _extract_submission_inputs(agent_result)

        print(
            f"Validating workflow {workflow_id} via GoWe dry_run",
            file=sys.stderr,
        )
        return await client.dry_run(
            workflow_id=workflow_id,
            inputs=inputs,
            auth_token=auth_token,
        )

    except GoWeError as exc:
        return {
            "valid": False,
            "error": str(exc),
            "error_type": exc.error_type,
            "status_code": exc.status_code,
        }
    except ValueError as exc:
        return {
            "valid": False,
            "error": str(exc),
            "error_type": "EXTRACTION_FAILED",
        }


async def get_workflow_status(
    workflow_id: str,
    config: AgentConfig,
) -> Dict[str, Any]:
    """Query GoWe for the current status of a submission.

    Note: In GoWe, workflow_id and submission_id are different concepts.
    This function treats the provided ID as a submission_id (which is
    what callers typically want after submitting a workflow).

    Args:
        workflow_id: The GoWe submission identifier (or workflow identifier).
        config: AgentConfig with ``gowe_url``.

    Returns:
        Status dict from GoWe (id, state, step_instances, etc.).

    Raises:
        SubmissionError: If GoWe is unreachable or returns an error.
    """
    _ensure_mcp_path(config)
    from common.gowe_client import GoWeClient, GoWeError

    try:
        client = GoWeClient(base_url=config.gowe_url)
        return await client.get_submission(
            workflow_id, auth_token=config.bvbrc_auth_token,
        )
    except GoWeError as exc:
        raise SubmissionError(
            str(exc), error_type=exc.error_type, status_code=exc.status_code,
        ) from exc


async def check_engine_health(config: AgentConfig) -> bool:
    """Return True if GoWe is reachable and healthy."""
    _ensure_mcp_path(config)
    from common.gowe_client import GoWeClient

    try:
        client = GoWeClient(base_url=config.gowe_url)
        return await client.is_healthy()
    except Exception:
        return False
