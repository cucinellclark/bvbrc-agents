"""
Shared GoWe workflow-engine tools.

These tools let any agent discover available workflows, inspect their
input schemas, and submit jobs to the GoWe CWL engine.

Each function accepts a generic *config* object (needs ``gowe_url``,
``bvbrc_auth_token``, ``mcp_server_path`` attributes) and optional
*headers*.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from shared.tools._mcp_imports import get_gowe_client

logger = logging.getLogger(__name__)


def _extract_user_from_token(auth_token: str | None) -> str | None:
    """Extract the ``un=<user>`` value from a BV-BRC auth token string."""
    if not auth_token:
        return None
    import re
    match = re.search(r"(?:^|[|&])un=([^|&]+)", auth_token)
    return match.group(1).strip() if match else None


def _get_client(config: Any = None):
    """Build a ``GoWeClient`` from config."""
    mod = get_gowe_client(getattr(config, "mcp_server_path", None))
    gowe_url = getattr(config, "gowe_url", None) or "http://140.221.78.67:12007"
    return mod.GoWeClient(base_url=gowe_url)


def _get_auth(config: Any = None) -> str:
    return getattr(config, "bvbrc_auth_token", None) or ""


# ---------------------------------------------------------------------------
# Identifier validation
# ---------------------------------------------------------------------------

_SRR_RE = re.compile(r"^[SED]RR\d{5,}$")
_GENOME_ID_RE = re.compile(r"^\d+\.\d+$")

# Field name -> (pattern, human description).  Values may be a string, a
# list of strings, or a list of records containing the field.
_IDENTIFIER_RULES: Dict[str, tuple[re.Pattern, str]] = {
    "srr_ids": (_SRR_RE, "SRA run accession like SRR1234567"),
    "srr_accession": (_SRR_RE, "SRA run accession like SRR1234567"),
    "reference_genome_id": (_GENOME_ID_RE, "BV-BRC genome ID like 83332.12"),
    "genome_id": (_GENOME_ID_RE, "BV-BRC genome ID like 83332.12"),
    "genome_ids": (_GENOME_ID_RE, "BV-BRC genome ID like 83332.12"),
    "db_genome_list": (_GENOME_ID_RE, "BV-BRC genome ID like 83332.12"),
}


def _check_identifiers(inputs: Dict[str, Any]) -> str | None:
    """Return an error string if a known identifier field has a malformed value.

    Walks top-level values and one level of records (e.g. srr_libs[].srr_accession).
    Returns on the first bad value found.
    """
    for field_name, (pattern, description) in _IDENTIFIER_RULES.items():
        # Check top-level fields
        value = inputs.get(field_name)
        if value is not None:
            bad = _find_bad_value(value, pattern)
            if bad is not None:
                hint = ""
                if pattern is _SRR_RE:
                    hint = " If this is a workspace file, use single_end_libs/paired_end_libs instead."
                return (
                    f"Invalid value for {field_name}: '{bad}' is not a valid "
                    f"{description}.{hint}"
                )

        # Check one level of records (e.g. srr_libs[].srr_accession)
        for key, val in inputs.items():
            if not isinstance(val, list):
                continue
            for item in val:
                if not isinstance(item, dict):
                    continue
                nested = item.get(field_name)
                if nested is not None:
                    bad = _find_bad_value(nested, pattern)
                    if bad is not None:
                        hint = ""
                        if pattern is _SRR_RE:
                            hint = " If this is a workspace file, use the file path inputs instead."
                        return (
                            f"Invalid value for {key}[].{field_name}: '{bad}' "
                            f"is not a valid {description}.{hint}"
                        )
    return None


def _find_bad_value(value: Any, pattern: re.Pattern) -> str | None:
    """Check a value (string or list of strings) against *pattern*.

    Returns the first non-matching string, or ``None`` if all match.
    """
    if isinstance(value, str):
        if not pattern.match(value):
            return value
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, str) and not pattern.match(item):
                return item
    return None


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


async def list_gowe_workflows(
    config: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """List all available workflows registered in GoWe.

    Returns a simplified list with workflow ``id``, ``name``, and
    ``description`` for each workflow.  Kept deliberately small so the
    full catalog fits in one tool result -- the LLM must see every entry.
    """
    try:
        client = _get_client(config)
        auth = _get_auth(config)
        workflows, pagination = await client.list_workflows(
            auth_token=auth,
            limit=100,
        )

        result = []
        for wf in workflows or []:
            result.append(
                {
                    "id": wf.get("id"),
                    "name": wf.get("name"),
                    "description": wf.get("description", ""),
                }
            )

        return {"workflows": result, "count": len(result)}

    except Exception as e:
        return {"error": f"Failed to list workflows: {type(e).__name__}: {e}"}


async def get_workflow_inputs(
    workflow_id: str,
    config: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Get the input schema for a specific GoWe workflow.

    Args:
        workflow_id: The GoWe workflow ID (e.g. ``'wf_abc123'``).

    Returns:
        Dict with ``workflow_id``, ``inputs`` list, and ``count``.
    """
    if not workflow_id:
        return {"error": "workflow_id is required."}

    try:
        client = _get_client(config)
        auth = _get_auth(config)
        inputs = await client.get_workflow_inputs(workflow_id, auth_token=auth)

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
    config: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Submit a job to GoWe with the given workflow ID and populated inputs.

    Args:
        workflow_id: The GoWe workflow ID.
        inputs: Dict of input values matching the workflow's input schema.
    """
    if not workflow_id:
        return {"error": "workflow_id is required."}

    auth = _get_auth(config)
    if not auth:
        return {"error": "No authentication token available."}

    # ----- Identifier validation -----
    id_error = _check_identifiers(inputs)
    if id_error:
        logger.warning("Identifier check failed for workflow %s: %s", workflow_id, id_error)
        return {"error": id_error, "workflow_id": workflow_id}

    try:
        client = _get_client(config)

        # Clean up the inputs before submission
        cleaned_inputs: Dict[str, Any] = {}
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

        # ----- Output path rewriting -----
        # Always ensure output_path is a valid absolute workspace path.
        # When session context is available, place results under the
        # chat session folder: /<user>/home/chats/<session_uuid>/<subfolder>
        # Otherwise, fall back to the user's home directory.
        session_id = getattr(config, "session_id", None)
        workspace_path = getattr(config, "workspace_path", None)

        if "output_path" in cleaned_inputs:
            original_output_path = cleaned_inputs["output_path"]
            # Extract the last path segment as the descriptive subfolder
            subfolder = original_output_path.rstrip("/").rsplit("/", 1)[-1]

            if session_id and workspace_path:
                # Primary path: place under session folder
                session_base = f"{workspace_path}/chats/{session_id}"
                cleaned_inputs["output_path"] = f"{session_base}/{subfolder}"
                logger.info(
                    "Rewrote output_path: %s -> %s (session=%s)",
                    original_output_path,
                    cleaned_inputs["output_path"],
                    session_id,
                )
            elif not original_output_path.startswith("/"):
                # Fallback: session context missing but output_path is
                # relative (bare folder name).  Construct an absolute path
                # from the auth token's username so BV-BRC doesn't reject
                # the submission.
                user = _extract_user_from_token(auth)
                if user:
                    cleaned_inputs["output_path"] = f"/{user}/home/{subfolder}"
                    logger.warning(
                        "Session context missing (session_id=%s, "
                        "workspace_path=%s). Fell back to user home: "
                        "%s -> %s",
                        session_id,
                        workspace_path,
                        original_output_path,
                        cleaned_inputs["output_path"],
                    )
                else:
                    logger.error(
                        "Cannot rewrite relative output_path %r: no "
                        "session context and cannot extract user from "
                        "auth token.",
                        original_output_path,
                    )

        # ----- GoWe submission labels -----
        labels: Dict[str, str] | None = None
        if session_id:
            labels = {"session_id": session_id, "source": "copilot"}

        result = await client.create_submission(
            workflow_id=workflow_id,
            inputs=cleaned_inputs,
            auth_token=auth,
            labels=labels,
        )

        submission_id = result.get("id", "")
        logger.info(
            "Job submitted: workflow_id=%s, submission_id=%s, state=%s",
            workflow_id,
            submission_id,
            result.get("state"),
        )

        out_path = cleaned_inputs.get("output_path", "")
        return {
            "workflow_id": workflow_id,
            "submission_id": submission_id,
            "status": result.get("state", "PENDING"),
            "message": (
                f"Job submitted. Results will be saved to {out_path} "
                f"(the chats/ session folder in the user's home workspace). "
                f"Tell the user this path. When the job finishes, a completion "
                f"message is added to this chat — reopen or refresh the chat "
                f"to see it, or check the Jobs list."
                if out_path else "Job submitted."
            ),
            "output_path": out_path,
        }

    except Exception as e:
        logger.error("Failed to submit job: %s", e)
        return {
            "error": f"Job submission failed: {type(e).__name__}: {e}",
            "workflow_id": workflow_id,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_record_values(records: list) -> list:
    """Strip URI prefixes from nested record values (e.g. paired-end libs)."""
    cleaned = []
    for rec in records:
        if not isinstance(rec, dict):
            cleaned.append(rec)
            continue
        clean_rec = {}
        for k, v in rec.items():
            if isinstance(v, str):
                if v.startswith("ws://"):
                    v = v[len("ws://") :]
                elif v.startswith("workspace:"):
                    v = v[len("workspace:") :]
            clean_rec[k] = v
        cleaned.append(clean_rec)
    return cleaned
