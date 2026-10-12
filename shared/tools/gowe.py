"""
Shared GoWe workflow-engine tools.

These tools let any agent discover available workflows, inspect their
input schemas, and submit jobs to the GoWe CWL engine.

Each function accepts a generic *config* object (needs ``gowe_url``,
``bvbrc_auth_token``, ``mcp_server_path`` attributes) and optional
*headers*.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from shared.tools._mcp_imports import get_gowe_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Duplicate-submission guard
# ---------------------------------------------------------------------------
# The same job reached GoWe several times on 2026-10-09: two "go ahead"
# messages each submitted it (nbowers, jobs 23731569 / 23731575), and one
# card click became five concurrent turns (parrello, five Comparative Systems
# jobs in 5 s).  A job is a duplicate when the same chat (or user) submits the
# same workflow with the same inputs inside the window.  output_path and
# output_file are left out of the comparison: the model invents a new folder
# name each turn, so including them would let every repeat through.
#
# Process-local: the orchestrator runs as one process.  The check and the
# reservation happen with no await in between, so concurrent turns in the
# event loop cannot both pass.

DUPLICATE_SUBMIT_WINDOW_S = 600.0
_DEDUP_IGNORED_INPUTS = frozenset({"output_path", "output_file"})
# key -> (submission_id or None while the submit is in flight, monotonic time)
_recent_submissions: Dict[str, tuple[Optional[str], float]] = {}


def _submission_key(owner: str, workflow_id: str, inputs: Dict[str, Any]) -> str:
    material = {k: v for k, v in inputs.items() if k not in _DEDUP_IGNORED_INPUTS}
    raw = json.dumps([owner, workflow_id, material], sort_keys=True, default=str)
    return hashlib.sha1(raw.encode()).hexdigest()


def _reserve_submission(key: str) -> Optional[Dict[str, Any]]:
    """Reserve *key*, or return the refusal if it was submitted recently."""
    now = time.monotonic()
    for k, (_, t) in list(_recent_submissions.items()):
        if now - t > DUPLICATE_SUBMIT_WINDOW_S:
            del _recent_submissions[k]
    if key in _recent_submissions:
        sub_id, t = _recent_submissions[key]
        when = f"{int(now - t)} s ago"
        what = f"as {sub_id}" if sub_id else "and is still being submitted"
        return {
            "error": (
                f"This exact job (same workflow and inputs) was already submitted "
                f"from this chat {when} {what}. It was NOT submitted again. Tell "
                f"the user it is already running and do not retry."
            ),
            "already_submitted": True,
            "duplicate_of": sub_id,
        }
    _recent_submissions[key] = (None, now)
    return None


# ---------------------------------------------------------------------------
# Output folder placement
# ---------------------------------------------------------------------------
# Every chat-submitted job writes under /<user>/home/chats/<session>/.  The
# model picks the subfolder name: the folder the user named ("to the Testing
# folder"), or a descriptive one.  The code only steps in when that name is
# unusable, and says so, because users were not told when their folder was
# replaced (Chris, dovie, cmao) and the chat folder passed as-is nested
# chats/<sid>/<sid> (parrello, Oct 9).

_RESERVED_FOLDER_NAMES = frozenset({"", "home", "chats", ".", ".."})


def _generated_folder_name(workflow_ref: str, now: datetime | None = None) -> str:
    label = "" if workflow_ref.startswith("wf_") else re.sub(r"[^A-Za-z0-9]+", "", workflow_ref)
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M")
    return f"{label or 'job'}_{stamp}"


def place_output_path(
    requested: Any,
    session_base: str,
    session_id: str,
    workflow_ref: str,
    now: datetime | None = None,
) -> tuple[str, Optional[str]]:
    """Return ``(final_output_path, note)`` for a job in this chat's folder.

    *note* is a user-facing sentence when the requested folder was moved or
    renamed, else ``None``.
    """
    req = requested.strip() if isinstance(requested, str) else ""
    trimmed = req.rstrip("/")

    # Already inside this chat's folder: keep it.
    if trimmed.startswith(session_base + "/"):
        return trimmed, None

    last = trimmed.rsplit("/", 1)[-1] if trimmed else ""
    if (
        not trimmed
        or trimmed == session_base
        or last == session_id
        or last in _RESERVED_FOLDER_NAMES
        or last.endswith("@patricbrc.org")
        or last.endswith("@bvbrc")
    ):
        final = f"{session_base}/{_generated_folder_name(workflow_ref, now)}"
        why = (
            "No output folder name was given"
            if not trimmed
            else f"`{req}` is not a usable folder name for a job"
        )
        return final, (
            f"{why}, so the results go to a new folder in this chat's "
            f"workspace folder: `{final}`."
        )

    final = f"{session_base}/{last}"
    if "/" in trimmed:
        return final, (
            f"Copilot keeps every job's results in this chat's workspace folder, "
            f"so `{req}` was saved as `{final}` (the folder name `{last}` was "
            f"kept)."
        )
    return final, None


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
# Workflow health
# ---------------------------------------------------------------------------

# Workflows that fail inside the BV-BRC app itself, where nothing on our side
# can fix it. Surfaced by list_gowe_workflows as `status: "experimental"` with
# a reason, so an agent can warn the user before spending their compute.
#
# Keyed by the workflow name as GoWe registers it (case-insensitive). Evidence
# and suggested upstream fixes: Agents/upstream_app_spec_issues.md
#
# Remove an entry once the app is fixed and a submission succeeds — do not let
# this list rot into a permanent denylist.
KNOWN_BROKEN_WORKFLOWS: Dict[str, str] = {
    "tree sort": (
        "Known broken upstream (0 of 5 submissions have ever succeeded). Fails "
        "inside the BV-BRC TreeSort app on both input paths: a genome group "
        "crashes in prepare_input_file, and a FASTA file fails in tree_sort "
        "without logging any reason. Not fixable from BV-BRC Copilot. Tell the "
        "user this before submitting, and offer to help another way."
    ),
    "protein structure prediction": (
        "Known broken upstream (0 of 1 submissions have succeeded). The app's "
        "own preflight fails with 'Illegal division by zero' before any work "
        "starts. Not fixable from BV-BRC Copilot. Tell the user this before "
        "submitting."
    ),
}


def _workflow_status(name: str | None) -> tuple[str, str | None]:
    """Return ``(status, reason)`` for a workflow name."""
    if not name:
        return "supported", None
    reason = KNOWN_BROKEN_WORKFLOWS.get(name.strip().lower())
    if reason:
        return "experimental", reason
    return "supported", None

# ---------------------------------------------------------------------------
# Workflow reference resolution
# ---------------------------------------------------------------------------

# GoWe's resolveWorkflow already accepts an id OR a name, on both
# GET /workflows/:id/inputs and POST /submissions. The failure mode is not the
# server: it is the LLM inventing a UUID that matches nothing.
#
# Observed 2026-10-08, HA Subtype never submitted. The agent tried
#   wf_893a5074-258b-4682-b9fa-6cb79a52943e   <- HA's prefix + MetaCATS's suffix
#   wf_893a5074-2c3c-488a-a044-2ed684ec7f50   <- a digit off the real id
# then gave up, because the only feedback was a bare "GoWeError: No ...".
#
# So: tool schemas now ask for the NAME, and a reference that resolves to
# nothing comes back naming the close matches instead of dead-ending.


# id -> name, filled by _resolve_workflow_ref from the catalog it already
# fetches, so checks keyed on the workflow name work when the model passes
# a wf_<uuid>.
_workflow_names: Dict[str, str] = {}


def _workflow_name(workflow_ref: str, workflow_id: str) -> str:
    if not workflow_ref.startswith("wf_"):
        return workflow_ref
    return _workflow_names.get(workflow_id, workflow_ref)


# ---------------------------------------------------------------------------
# Organism check for annotation workflows
# ---------------------------------------------------------------------------
# taxonomy_id is optional in the app specs, our CWL and GoWe /inputs, but the
# BV-BRC website never submits CGA or annotation without one, and the app
# does not cope: mshukla's CGA (scientific_name "metagenome", no taxonomy_id)
# failed in annotation on "Cannot parse 'taxon_id:'"; marcnguy's CGA was sent
# scientific_name "bau_sim" (a read-file prefix).  service_status.md has both.

_ORGANISM_REQUIRED_WORKFLOWS = frozenset({"comprehensivegenomeanalysis", "genomeannotation"})
_NOT_AN_ORGANISM_RE = re.compile(
    r"/|\.(fa|fasta|fna|faa|fastq|fq|gz|txt|contigs?)\b|^[SED]RR\d+|^\d+\.\d+$",
    re.IGNORECASE,
)
_NOT_AN_ORGANISM_WORDS = frozenset({"metagenome", "unknown", "sample", "na", "none", "organism"})


def _check_organism(workflow_name: str, inputs: Dict[str, Any]) -> Optional[tuple[str, str]]:
    """Return ``(kind, reason)`` when the organism inputs are unusable, else None.

    *kind* is ``"wrong_workflow"`` (a metagenome) or ``"missing_organism"``.
    """
    if re.sub(r"[^a-z]", "", workflow_name.lower()) not in _ORGANISM_REQUIRED_WORKFLOWS:
        return None
    name = str(inputs.get("scientific_name") or "").strip()
    tax = inputs.get("taxonomy_id")
    tax_ok = (isinstance(tax, int) and not isinstance(tax, bool) and tax > 0) or (
        isinstance(tax, str) and tax.strip().isdigit() and int(tax) > 0
    )
    lookup = (
        "Ask the user which organism this genome is (if they have not said), "
        "then look up its id with search_data(collection=\"taxonomy\", "
        "query='taxon_name:\"<Genus species>\" AND taxon_rank:species', "
        "select=[\"taxon_id\",\"taxon_name\"]) and use the entry whose "
        "taxon_name matches exactly. Never use a filename or sample name as the organism."
    )
    if name.lower() in ("metagenome",) or "metagenom" in name.lower():
        return "wrong_workflow", (
            f"{workflow_name} annotates a single organism's genome; it cannot run on a "
            f"metagenome. For a metagenomic sample use Metagenomic Binning (then annotate "
            f"the bins) or Taxonomic Classification. Tell the user this; do not submit."
        )
    if not name or name.lower() in _NOT_AN_ORGANISM_WORDS or _NOT_AN_ORGANISM_RE.search(name):
        what = f"scientific_name {name!r} is not an organism name" if name else "scientific_name is missing"
        return "missing_organism", f"{workflow_name} needs the organism: {what}. {lookup}"
    if not tax_ok:
        return "missing_organism", (
            f"{workflow_name} needs the NCBI taxonomy_id for {name!r} (the BV-BRC "
            f"website always sends it, and annotation fails without it). {lookup}"
        )
    return None


async def _resolve_workflow_ref(ref: str, config: Any = None) -> tuple[str | None, str | None]:
    """Resolve a workflow name or id to an id.

    Returns ``(workflow_id, None)`` on success, or ``(None, error_message)``
    with the nearest catalog entries named so the caller can retry. A value
    that already matches exactly is returned untouched, so the usual path
    costs one list call and no guessing.
    """
    import difflib

    try:
        client = _get_client(config)
        workflows, _ = await client.list_workflows(auth_token=_get_auth(config), limit=200)
    except Exception:
        # Catalog unreachable: let the caller's own call produce the error.
        return ref, None

    by_id = {w.get("id"): w for w in workflows or [] if w.get("id")}
    _workflow_names.update({i: w.get("name", "") for i, w in by_id.items()})
    if ref in by_id:
        return ref, None
    for w in workflows or []:
        if (w.get("name") or "").strip().lower() == ref.strip().lower():
            return w["id"], None

    names = [w.get("name", "") for w in workflows or []]
    close = difflib.get_close_matches(ref, names, n=3, cutoff=0.4)
    if not close and ref.startswith("wf_"):
        # A mistyped UUID resembles no name, so suggest by id similarity.
        close = [
            by_id[i].get("name", i)
            for i in difflib.get_close_matches(ref, list(by_id), n=3, cutoff=0.6)
        ]
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    return None, (
        f"No workflow matches {ref!r}. Pass the workflow NAME exactly as "
        f"list_gowe_workflows reported it, not a UUID you reconstructed."
        f"{hint} Call list_gowe_workflows to see the catalog."
    )


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
        experimental = 0
        for wf in workflows or []:
            status, reason = _workflow_status(wf.get("name"))
            entry = {
                "id": wf.get("id"),
                "name": wf.get("name"),
                "description": wf.get("description", ""),
                "status": status,
            }
            if reason:
                entry["status_reason"] = reason
                experimental += 1
            result.append(entry)

        out: Dict[str, Any] = {"workflows": result, "count": len(result)}
        if experimental:
            out["note"] = (
                f"{experimental} workflow(s) are marked status=experimental "
                "with a status_reason. Do not submit those without telling the "
                "user why they are likely to fail."
            )
        return out

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

    resolved, err = await _resolve_workflow_ref(workflow_id, config)
    if err:
        return {"error": err}

    try:
        client = _get_client(config)
        auth = _get_auth(config)
        inputs = await client.get_workflow_inputs(resolved, auth_token=auth)

        return {
            "workflow_id": resolved,
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

    # ----- Workflow reference -----
    # Accepts a name or an id; a reference matching nothing comes back naming
    # the close matches rather than as a bare GoWe error.
    workflow_ref = workflow_id
    resolved, ref_error = await _resolve_workflow_ref(workflow_id, config)
    if ref_error:
        return {"error": ref_error, "workflow_id": workflow_id}
    workflow_id = resolved

    # ----- Identifier validation -----
    id_error = _check_identifiers(inputs)
    if id_error:
        logger.warning("Identifier check failed for workflow %s: %s", workflow_id, id_error)
        return {"error": id_error, "workflow_id": workflow_id}

    # ----- Organism (CGA / Genome Annotation) -----
    # Fixable by asking the user, so it is not a failed submission.
    organism_problem = _check_organism(_workflow_name(workflow_ref, workflow_id), inputs)
    if organism_problem:
        kind, reason = organism_problem
        logger.warning("Organism check failed for workflow %s: %s", workflow_id, reason)
        return {
            "error": reason,
            "needs_user_input": True,
            **({"wrong_workflow": True} if kind == "wrong_workflow" else {"missing_input": "taxonomy_id"}),
            "workflow_id": workflow_id,
        }

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
        output_note: Optional[str] = None

        if session_id and workspace_path and "output_path" in cleaned_inputs:
            # Primary path: always under this chat's folder.
            session_base = f"{workspace_path}/chats/{session_id}"
            original_output_path = cleaned_inputs.get("output_path")
            cleaned_inputs["output_path"], output_note = place_output_path(
                original_output_path, session_base, session_id, workflow_ref,
            )
            logger.info(
                "Placed output_path: %r -> %s (session=%s)",
                original_output_path,
                cleaned_inputs["output_path"],
                session_id,
            )
        elif "output_path" in cleaned_inputs:
            original_output_path = cleaned_inputs["output_path"]
            subfolder = original_output_path.rstrip("/").rsplit("/", 1)[-1]
            if not original_output_path.startswith("/"):
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

        dedup_key = _submission_key(
            session_id or _extract_user_from_token(auth) or auth,
            workflow_id,
            cleaned_inputs,
        )
        duplicate = _reserve_submission(dedup_key)
        if duplicate:
            logger.warning(
                "Duplicate submission refused: workflow_id=%s, duplicate_of=%s, session=%s",
                workflow_id, duplicate["duplicate_of"], session_id,
            )
            return {**duplicate, "workflow_id": workflow_id}

        try:
            result = await client.create_submission(
                workflow_id=workflow_id,
                inputs=cleaned_inputs,
                auth_token=auth,
                labels=labels,
            )
        except BaseException:
            _recent_submissions.pop(dedup_key, None)
            raise

        submission_id = result.get("id", "")
        if submission_id:
            _recent_submissions[dedup_key] = (submission_id, time.monotonic())
        else:
            _recent_submissions.pop(dedup_key, None)
        logger.info(
            "Job submitted: workflow_id=%s, submission_id=%s, state=%s",
            workflow_id,
            submission_id,
            result.get("state"),
        )

        out_path = cleaned_inputs.get("output_path", "")
        response = {
            "workflow_id": workflow_id,
            "submission_id": submission_id,
            "status": result.get("state", "PENDING"),
            "message": (
                f"Job submitted. Results will be saved to {out_path} "
                f"(the chats/ session folder in the user's home workspace). "
                f"Tell the user this exact path. When the job finishes, a "
                f"completion message is added to this chat — reopen or refresh "
                f"the chat to see it, or check the Jobs list."
                if out_path else "Job submitted."
            ),
            "output_path": out_path,
        }
        if output_note:
            response["output_path_note"] = output_note
        return response

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
