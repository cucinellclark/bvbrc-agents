"""Workflow submission utilities for the BV-BRC Service Agent v2.

This module lives *outside* the three-phase planning pipeline.  It is a
thin async wrapper around the workflow engine REST API that the caller /
orchestrator invokes after planning is complete and the user has confirmed
they want to submit.

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
from typing import Any, Dict

import httpx

from service_agent.models import AgentConfig, AgentResult, SubmissionResult


# ---------------------------------------------------------------------------
# Manifest extraction
# ---------------------------------------------------------------------------

def extract_workflow_definition(agent_result: AgentResult) -> Dict[str, Any]:
    """Extract the engine-ready WorkflowDefinition from an AgentResult.

    The compose phase stores its output on ``AgentResult.manifest``.  That
    dict has a nested ``"manifest"`` key containing the actual
    WorkflowDefinition that the workflow engine expects.  This helper
    unwraps it and strips planner/compose metadata so the payload is clean
    for submission.

    Returns:
        A dict suitable for ``POST /api/v1/workflows/submit``.

    Raises:
        ValueError: If the AgentResult has no usable manifest.
    """
    outer = agent_result.manifest
    if not outer or not isinstance(outer, dict):
        raise ValueError("AgentResult has no manifest to submit.")

    # The compose phase wraps the actual definition under "manifest"
    definition = outer.get("manifest")
    if isinstance(definition, dict) and "steps" in definition:
        return _clean_definition(definition)

    # Fallback: the outer dict itself might already be a WorkflowDefinition
    # (e.g. if a caller built the manifest differently).
    if "steps" in outer and "workflow_name" in outer:
        return _clean_definition(outer)

    raise ValueError(
        "Cannot locate a valid WorkflowDefinition in AgentResult.manifest. "
        "Expected a 'manifest' key with 'steps' and 'workflow_name'."
    )


def _clean_definition(definition: Dict[str, Any]) -> Dict[str, Any]:
    """Strip fields that the workflow engine rejects or assigns itself."""
    cleaned = json.loads(json.dumps(definition))

    for field in (
        "workflow_id", "status", "created_at", "updated_at",
        "submitted_at", "started_at", "completed_at", "error_message",
        "execution_metadata", "log_file_path", "auth_token",
        "step_count", "steps_summary",
    ):
        cleaned.pop(field, None)

    if isinstance(cleaned.get("steps"), list):
        for step in cleaned["steps"]:
            if not isinstance(step, dict):
                continue
            for step_field in (
                "step_id", "status", "task_id", "submitted_at",
                "started_at", "completed_at", "elapsed_time",
                "error_message",
            ):
                step.pop(step_field, None)

    return cleaned


# ---------------------------------------------------------------------------
# Workflow engine HTTP helpers
# ---------------------------------------------------------------------------

class SubmissionError(Exception):
    """Raised when the workflow engine rejects or cannot process a submission."""

    def __init__(
        self,
        message: str,
        error_type: str = "UNKNOWN_ERROR",
        status_code: int | None = None,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code


async def _engine_post(
    url: str,
    payload: Dict[str, Any] | None,
    auth_token: str,
    timeout: int,
) -> Dict[str, Any]:
    """POST *payload* to *url* and return the parsed JSON response."""
    headers = {"Content-Type": "application/json", "Authorization": auth_token}
    client_timeout = httpx.Timeout(timeout)

    try:
        async with httpx.AsyncClient(timeout=client_timeout) as client:
            kwargs: dict[str, Any] = {"headers": headers}
            if payload is not None:
                kwargs["json"] = payload
            resp = await client.post(url, **kwargs)
            body = resp.text
            if resp.status_code in (200, 201):
                return resp.json()
            try:
                detail = resp.json().get("detail", body)
            except Exception:
                detail = body
            error_type = (
                "VALIDATION_FAILED" if resp.status_code == 400
                else "NOT_FOUND" if resp.status_code == 404
                else "ENGINE_ERROR" if resp.status_code == 500
                else "UNKNOWN_ERROR"
            )
            raise SubmissionError(str(detail), error_type, resp.status_code)
    except SubmissionError:
        raise
    except httpx.ConnectError as exc:
        raise SubmissionError(
            f"Cannot connect to workflow engine at {url}. Is it running?",
            "CONNECTION_FAILED",
        ) from exc
    except Exception as exc:
        raise SubmissionError(
            f"Unexpected error: {exc}",
            "UNKNOWN_ERROR",
        ) from exc


async def _engine_get(url: str, timeout: int) -> Dict[str, Any]:
    """GET *url* and return the parsed JSON response."""
    client_timeout = httpx.Timeout(timeout)
    try:
        async with httpx.AsyncClient(timeout=client_timeout) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return resp.json()
            body = resp.text
            error_type = (
                "NOT_FOUND" if resp.status_code == 404 else "QUERY_FAILED"
            )
            raise SubmissionError(
                f"Engine returned {resp.status_code}: {body}",
                error_type, resp.status_code,
            )
    except SubmissionError:
        raise
    except httpx.ConnectError as exc:
        raise SubmissionError(
            f"Cannot connect to workflow engine at {url}. Is it running?",
            "CONNECTION_FAILED",
        ) from exc
    except Exception as exc:
        raise SubmissionError(
            f"Unexpected error: {exc}", "UNKNOWN_ERROR",
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def submit_workflow(
    agent_result: AgentResult,
    config: AgentConfig,
) -> SubmissionResult:
    """Submit a completed AgentResult to the workflow engine.

    Extracts the inner WorkflowDefinition from the manifest, POSTs it to
    the engine's ``/workflows/submit`` endpoint, and returns a
    :class:`SubmissionResult`.

    Args:
        agent_result: A completed AgentResult with a manifest.
        config: AgentConfig with ``workflow_engine_url``,
            ``workflow_engine_timeout``, and ``bvbrc_auth_token``.

    Returns:
        SubmissionResult with the engine-assigned workflow_id and status.

    Raises:
        ValueError: If the AgentResult has no manifest or no auth token.
        SubmissionError: If the engine rejects the submission.
    """
    if not config.bvbrc_auth_token:
        raise ValueError(
            "Cannot submit workflow: no auth token configured. "
            "Provide a BV-BRC auth token via --auth-token-file or config."
        )

    definition = extract_workflow_definition(agent_result)
    base = config.workflow_engine_url.rstrip("/")
    url = f"{base}/workflows/submit"

    print(f"Submitting workflow to engine: {url}", file=sys.stderr)

    try:
        resp = await _engine_post(
            url, definition, config.bvbrc_auth_token,
            config.workflow_engine_timeout,
        )
        workflow_id = resp.get("workflow_id", "unknown")
        return SubmissionResult(
            workflow_id=workflow_id,
            status=resp.get("status", "pending"),
            engine_url=base,
            status_url=f"{base}/workflows/{workflow_id}/status",
        )
    except SubmissionError as exc:
        return SubmissionResult(
            workflow_id="",
            status="submission_failed",
            engine_url=base,
            status_url="",
            error=str(exc),
        )


async def validate_workflow(
    agent_result: AgentResult,
    config: AgentConfig,
) -> Dict[str, Any]:
    """Validate a completed AgentResult's manifest against the workflow engine.

    Extracts the inner WorkflowDefinition, POSTs it to the engine's
    ``/workflows/validate`` endpoint, and returns the validation result.

    Args:
        agent_result: A completed AgentResult with a manifest.
        config: AgentConfig with engine URL/timeout and auth token.

    Returns:
        Dict with ``valid`` (bool), ``workflow_json`` (normalized),
        ``warnings``, ``auto_fixes``, and ``message``.
    """
    definition = extract_workflow_definition(agent_result)
    return await validate_workflow_json(definition, config)


async def validate_workflow_json(
    workflow_json: Dict[str, Any],
    config: AgentConfig,
) -> Dict[str, Any]:
    """Validate a raw workflow JSON dict against the workflow engine.

    This is the lower-level entry point for callers that already have a
    WorkflowDefinition dict (e.g. loaded from a JSON file) rather than an
    AgentResult.

    Args:
        workflow_json: A WorkflowDefinition dict.
        config: AgentConfig with engine URL/timeout and auth token.

    Returns:
        Dict with ``valid`` (bool), ``workflow_json`` (normalized),
        ``warnings``, ``auto_fixes``, and ``message``; or ``valid=False``
        with error details on failure.
    """
    base = config.workflow_engine_url.rstrip("/")
    url = f"{base}/workflows/validate"
    auth_token = config.bvbrc_auth_token or ""

    print(f"Validating workflow against engine: {url}", file=sys.stderr)

    try:
        cleaned = _clean_definition(workflow_json)
        return await _engine_post(url, cleaned, auth_token, config.workflow_engine_timeout)
    except SubmissionError as exc:
        return {
            "valid": False,
            "error": str(exc),
            "error_type": exc.error_type,
            "status_code": exc.status_code,
        }


async def get_workflow_status(
    workflow_id: str,
    config: AgentConfig,
) -> Dict[str, Any]:
    """Query the workflow engine for the current status of a workflow.

    Args:
        workflow_id: The engine-assigned workflow identifier.
        config: AgentConfig with ``workflow_engine_url`` and
            ``workflow_engine_timeout``.

    Returns:
        Status dict from the engine (workflow_id, status, steps, etc.).

    Raises:
        SubmissionError: If the engine is unreachable or returns an error.
    """
    base = config.workflow_engine_url.rstrip("/")
    url = f"{base}/workflows/{workflow_id}/status"
    return await _engine_get(url, config.workflow_engine_timeout)


async def check_engine_health(config: AgentConfig) -> bool:
    """Return True if the workflow engine is reachable and healthy."""
    base = config.workflow_engine_url.rstrip("/")
    url = f"{base}/health"
    try:
        data = await _engine_get(url, timeout=5)
        return data.get("mongodb") == "connected"
    except Exception:
        return False
