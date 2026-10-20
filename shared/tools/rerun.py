"""``rerun_jobs``: prepare a rerun of earlier BV-BRC jobs, with no LLM.

D4 Phase 2 (``current_plans/multi_step_requests.md`` §5).  The LLM used to do
this by hand -- browse, ``get_job_details``, ``list_gowe_workflows``,
``get_workflow_inputs``, ``submit_gowe_job`` -- nine tools and 92 s for one
job, and timeouts for a folder (nbowers 866d843e).  The data is all there:

1. job results under a folder (``workspace_browse``, recursive) or task ids;
2. each job's app and original ``parameters`` (``AppService.query_tasks``);
3. the GoWe workflow that runs that app (``steps[].hints.bvbrc_app_id``);
4. inputs = the parameters the workflow declares, output under this chat's
   folder as ``<output_file>_rerun``;
5. a GoWe dry run of each job, so a job the registered workflow would
   refuse is listed as skipped instead of failing after the user's Yes.

Nothing is submitted here.  The result's ``prepared_actions`` become
``blocked_actions`` (``BaseAgentState.record_execution``), the user answers
one Yes / No / Other question, and Yes replays each ``submit_gowe_job``
verbatim -- through the duplicate guard, the identifier checks and output
placement like any other submission.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from shared.tools.app_service import query_app_tasks
from shared.tools import gowe
from shared.tools.gowe import (
    APP_WORKFLOW_TTL_S,
    _check_identifiers,
    _workflow_status,
    place_output_path,
)
from shared.tools.roles import check_roles, labels_for_numbered_conditions, with_roles
from shared.tools.workspace import workspace_browse

logger = logging.getLogger(__name__)

MAX_FOLDER_DEPTH = 4
MAX_JOB_ITEMS = 200
# Same limit as a prepared set and as replay (MAX_PREPARED_JOBS, gateway
# MAX_REPLAY_ACTIONS): a bigger set could be prepared but never confirmed.
MAX_RERUN_JOBS = 10

_OUTPUT_KEYS = ("output_path", "output_file")
_RUNNING_STATES = frozenset({"queued", "in-progress", "pending", "running"})

# (gowe_url, workflow_id) -> (fetched_at, declared input ids)
_inputs_cache: Dict[tuple[str, str], tuple[float, frozenset[str]]] = {}


def _token(config: Any, headers: Optional[Dict[str, str]]) -> Optional[str]:
    if headers and headers.get("Authorization"):
        return headers["Authorization"]
    return getattr(config, "bvbrc_auth_token", None) or None


async def _declared_inputs(workflow_id: str, config: Any) -> frozenset[str]:
    key = (getattr(config, "gowe_url", None) or "", workflow_id)
    hit = _inputs_cache.get(key)
    if hit and time.monotonic() - hit[0] < APP_WORKFLOW_TTL_S:
        return hit[1]
    client = gowe._get_client(config)
    inputs = await client.get_workflow_inputs(workflow_id, auth_token=gowe._get_auth(config))
    declared = frozenset(i["id"] for i in inputs or [] if isinstance(i, dict) and i.get("id"))
    _inputs_cache[key] = (time.monotonic(), declared)
    return declared


def _depth_under(path: str, folder: str) -> int:
    """Folder levels between *folder* and the job result at *path*."""
    rel = path[len(folder):].strip("/") if path.startswith(folder) else path.strip("/")
    return max(0, rel.count("/"))


def _subfolder(path: str, folder: str) -> str:
    """The first-level subfolder of *folder* that holds *path*."""
    rel = path[len(folder):].strip("/") if path.startswith(folder) else path.strip("/")
    head = rel.split("/", 1)[0] if "/" in rel else ""
    return f"{folder}/{head}" if head else folder


def _canonical(inputs: Dict[str, Any]) -> str:
    return json.dumps(
        {k: v for k, v in inputs.items() if k not in _OUTPUT_KEYS},
        sort_keys=True, default=str,
    )


def _last_segment(path: Any) -> str:
    return str(path or "").rstrip("/").rsplit("/", 1)[-1]


async def _job_results(folder: str, config: Any, headers: Dict[str, str]) -> Dict[str, Any]:
    """Job results under *folder*: ``{"items": [...], "deeper": n}`` or an error dict."""
    listing = await workspace_browse(
        path=folder,
        workspace_types=["job_result"],
        num_results=MAX_JOB_ITEMS + 1,
        config=config,
        headers=headers,
    )
    if not isinstance(listing, dict) or listing.get("error"):
        return listing if isinstance(listing, dict) else {"error": "Workspace listing failed."}
    inner = listing.get("result") or {}
    base = (inner.get("path") or folder).rstrip("/")
    raw = [i for i in inner.get("items") or [] if isinstance(i, dict) and i.get("type") == "job_result"]
    if len(raw) > MAX_JOB_ITEMS:
        return {
            "error": (
                f"More than {MAX_JOB_ITEMS} job results under {base}. Ask the user "
                f"which subfolder to rerun."
            ),
            "errorType": "TOO_MANY",
            "folder": base,
        }
    items = [i for i in raw if _depth_under(i.get("path", ""), base) <= MAX_FOLDER_DEPTH]
    return {"items": items, "deeper": len(raw) - len(items), "folder": base}


async def rerun_jobs(
    folder: Optional[str] = None,
    task_ids: Optional[List[str]] = None,
    include_failed: bool = True,
    config: Any = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Prepare reruns of the jobs under *folder* and/or with *task_ids*.

    Returns ``{"prepared_actions": [...], "skipped": [...], "collapsed": [...],
    "note": ...}``, or an error dict (``errorType`` ``TOO_MANY`` when more than
    10 jobs could run; nothing is prepared then).
    """
    if not folder and not task_ids:
        return {
            "error": "Give a folder (path) or task_ids of the jobs to rerun.",
            "errorType": "INVALID_PARAMETERS",
        }
    token = _token(config, headers)
    if not token:
        return {
            "error": "Authentication required to read the original jobs.",
            "errorType": "AUTHENTICATION_ERROR",
        }
    auth_headers = {"Authorization": token}

    # ---- 1. Find jobs ------------------------------------------------------
    jobs: Dict[str, Dict[str, Any]] = {}  # task_id -> {task_id, app, success, path}
    deeper = 0
    base = None
    if folder:
        found = await _job_results(folder, config, auth_headers)
        if found.get("error"):
            return found
        base = found["folder"]
        deeper = found["deeper"]
        for item in found["items"]:
            job = item.get("job") or {}
            tid = str(job.get("task_id") or "")
            if not tid:
                continue
            success = job.get("success")
            jobs[tid] = {
                "task_id": tid,
                "app": job.get("app"),
                "success": None if success is None else bool(success),
                "path": item.get("path", ""),
            }
    for tid in task_ids or []:
        tid = str(tid).strip()
        if tid and tid not in jobs:
            jobs[tid] = {"task_id": tid, "app": None, "success": None, "path": ""}

    if not jobs:
        where = base or folder
        return {
            "error": f"No job results found under {where}.",
            "errorType": "NO_JOBS",
            "folder": where,
        }

    skipped: List[Dict[str, Any]] = []

    def skip(job: Dict[str, Any], reason: str) -> None:
        skipped.append({"task_id": job["task_id"], "app": job.get("app"), "reason": reason})

    if not include_failed:
        for tid in [t for t, j in jobs.items() if j["success"] is False]:
            skip(jobs.pop(tid), "original failed (include_failed is false)")

    # ---- 2. Original parameters --------------------------------------------
    try:
        tasks = await query_app_tasks(list(jobs), token) if jobs else {}
    except Exception as e:
        return {"error": f"Could not read the original jobs: {e}", "errorType": "API_ERROR"}

    # ---- 3. App -> workflow ------------------------------------------------
    try:
        catalog = await gowe.workflow_catalog(config)
    except Exception as e:
        return {"error": f"Could not reach the workflow catalog: {e}", "errorType": "API_ERROR"}
    by_app: Dict[str, Dict[str, Any]] = {}
    for entry in catalog.values():
        if entry.get("bvbrc_app_id"):
            by_app.setdefault(entry["bvbrc_app_id"], entry)

    session_id = getattr(config, "session_id", None)
    workspace_path = getattr(config, "workspace_path", None)
    session_base = f"{workspace_path}/chats/{session_id}" if session_id and workspace_path else None

    # ---- 4. Inputs ---------------------------------------------------------
    candidates: List[Dict[str, Any]] = []
    for tid, job in jobs.items():
        task = tasks.get(tid)
        if not task:
            skip(job, "job not found, or not one of this user's jobs")
            continue
        app = task.get("app") or task.get("application_name") or job.get("app")
        job["app"] = app
        status = str(task.get("status") or "").lower()
        if status == "failed" and job["success"] is None:
            job["success"] = False
            if not include_failed:
                skip(job, "original failed (include_failed is false)")
                continue
        if status in _RUNNING_STATES:
            skip(job, f"still running ({status}); rerun it once it has finished")
            continue
        workflow = by_app.get(app or "")
        if not workflow:
            skip(job, f"cannot rerun here: the {app} app is not in Copilot's workflow catalog")
            continue
        wf_status, wf_reason = _workflow_status(workflow["name"])
        if wf_status == "experimental":
            skip(job, f"{workflow['name']} is marked experimental: {wf_reason}")
            continue
        params = task.get("parameters") if isinstance(task.get("parameters"), dict) else {}
        try:
            declared = await _declared_inputs(workflow["id"], config)
        except Exception as e:
            skip(job, f"could not read the inputs of {workflow['name']}: {e}")
            continue
        inputs = labels_for_numbered_conditions(
            workflow["name"], {k: v for k, v in params.items() if k in declared}
        )
        dropped = sorted(k for k in params if k not in declared and not k.startswith("_"))
        id_error = _check_identifiers(inputs) or check_roles(workflow["name"], inputs)[0]
        if id_error:
            skip(job, f"the recorded inputs fail a check: {id_error}")
            continue

        original_folder = params.get("output_path")
        original_file = str(params.get("output_file") or f"job_{tid}")
        if session_base:
            inputs["output_path"], _ = place_output_path(
                original_folder, session_base, session_id, workflow["name"],
            )
        elif original_folder:
            inputs["output_path"] = original_folder
        inputs["output_file"] = f"{original_file}_rerun"

        candidates.append({
            "job": job,
            "workflow": workflow,
            "inputs": inputs,
            "dropped": dropped,
            "original_folder": _last_segment(original_folder) or _last_segment(job.get("path")),
        })

    # ---- 5. Dedupe ---------------------------------------------------------
    runnable: List[Dict[str, Any]] = []
    collapsed: List[Dict[str, str]] = []
    seen: Dict[tuple[str, str], str] = {}
    for c in candidates:
        key = (c["workflow"]["id"], _canonical(c["inputs"]))
        if key in seen:
            collapsed.append({"task_id": c["job"]["task_id"], "same_as": seen[key]})
            continue
        seen[key] = c["job"]["task_id"]
        runnable.append(c)

    # ---- 6. Cap ------------------------------------------------------------
    if len(runnable) > MAX_RERUN_JOBS:
        counts: Dict[str, int] = {}
        for c in runnable:
            sub = _subfolder(c["job"].get("path") or "", base) if base else "(task ids)"
            counts[sub] = counts.get(sub, 0) + 1
        return {
            "error": (
                f"{len(runnable)} jobs could be rerun; one message can prepare at most "
                f"{MAX_RERUN_JOBS}. Nothing was prepared. Ask the user which subfolder "
                f"(or which task ids) to rerun."
            ),
            "errorType": "TOO_MANY",
            "runnable_count": len(runnable),
            "subfolder_counts": counts,
            "jobs": [
                {"task_id": c["job"]["task_id"], "app": c["job"]["app"],
                 "workflow": c["workflow"]["name"], "original_failed": c["job"]["success"] is False}
                for c in runnable
            ],
            "skipped": skipped,
        }

    # ---- 7. Dry run --------------------------------------------------------
    # The registered workflow can be stricter than the app was when the job
    # first ran (Docking's "named_library", Oct 14).  A job GoWe would refuse
    # is skipped here rather than failing after the user's Yes.  A dry run
    # that errors keeps the job: the real submit decides.
    refusals = await asyncio.gather(
        *(gowe.dry_run_refusal(c["workflow"]["id"], c["inputs"], config) for c in runnable)
    )
    kept = []
    for c, refusal in zip(runnable, refusals):
        if refusal:
            skip(c["job"], f"GoWe refuses the recorded inputs for {c['workflow']['name']}: {refusal}")
        else:
            kept.append(c)
    runnable = kept

    # ---- 8. Prepared actions -----------------------------------------------
    used_outputs: set[tuple[str, str]] = set()
    prepared: List[Dict[str, Any]] = []
    for c in runnable:
        inputs = c["inputs"]
        out = (str(inputs.get("output_path") or ""), inputs["output_file"])
        n = 2
        while out in used_outputs:
            out = (out[0], f"{c['inputs']['output_file']}_{n}")
            n += 1
        inputs["output_file"] = out[1]
        used_outputs.add(out)

        job = c["job"]
        failed = " (original failed)" if job["success"] is False else ""
        summary = with_roles(
            f"{c['workflow']['name']} ← job {job['task_id']} ({c['original_folder']}){failed}",
            check_roles(c["workflow"]["name"], inputs)[1],
        )
        action: Dict[str, Any] = {
            "tool": "submit_gowe_job",
            "summary": summary,
            "arguments": {"workflow_id": c["workflow"]["id"], "inputs": inputs},
            "origin": "rerun_jobs",
            "original": {"task_id": job["task_id"], "app": job["app"], "success": job["success"]},
        }
        if c["dropped"]:
            action["dropped_inputs"] = c["dropped"]
        prepared.append(action)

    if prepared:
        notes = [
            f"Prepared {len(prepared)} rerun{'s' if len(prepared) != 1 else ''}; nothing has "
            f"been submitted. The user confirms the set with one question (Yes / No / Other).",
            "Inputs are reused exactly as recorded; a file deleted since the original run "
            "makes that job fail.",
        ]
    else:
        notes = ["No job could be prepared for a rerun; nothing has been submitted."]
    if collapsed:
        notes.append(
            f"{len(collapsed)} job{'s were' if len(collapsed) != 1 else ' was'} identical to "
            f"another in the set and collapsed into it."
        )
    if deeper:
        notes.append(
            f"{deeper} job result{'s' if deeper != 1 else ''} deeper than {MAX_FOLDER_DEPTH} "
            f"folder levels {'were' if deeper != 1 else 'was'} left out."
        )
    if skipped:
        notes.append("Tell the user each skipped job and why.")
    logger.info(
        "rerun_jobs: %d prepared, %d skipped, %d collapsed (folder=%s, task_ids=%s)",
        len(prepared), len(skipped), len(collapsed), base, task_ids,
    )
    return {
        "prepared_actions": prepared,
        "skipped": skipped,
        "collapsed": collapsed,
        "note": " ".join(notes),
    }
