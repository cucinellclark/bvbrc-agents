"""
Tool dispatcher for the BV-BRC Analysis Agent.

Maps tool names (from the LLM's tool_calls) to their async implementation
functions. Reuses shared workspace and data tools, adds local
get_expected_outputs and get_job_details tools.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
import traceback
from pathlib import Path
from typing import Any, Dict

from shared.tools import execute_tool as _shared_execute_tool
from shared.tools.workspace import (
    workspace_browse,
    get_file_metadata,
    read_file_preview,
)
from shared.tools.data import search_data
from shared.tools.similar_genome import find_similar_genomes
from shared.tools.literature import search_literature
from shared.tools.app_service import get_app_service_api, query_tasks_raw
from analysis_agent.output_knowledge import (
    get_expected_outputs as _get_expected_outputs,
)

# Import the workspace agent's sophisticated truncate_result
from shared.tools import truncate_result  # noqa: E402


async def _get_expected_outputs_async(
    service_name: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Async wrapper for the local get_expected_outputs lookup.

    This is a local lookup (no API call), but we wrap it as async to match
    the signature expected by execute_tool.
    """
    kwargs.pop("config", None)
    kwargs.pop("headers", None)
    return _get_expected_outputs(service_name)


# ---------------------------------------------------------------------------
# get_job_details: resolve task IDs to job metadata (status, output path, etc.)
# ---------------------------------------------------------------------------

# The AppService client is shared with rerun_jobs (shared/tools/app_service.py).
_get_service_api = get_app_service_api


async def get_job_details(
    task_ids: list[str],
    stdout: bool = False,
    stderr: bool = False,
    config: Any = None,
    headers: Dict[str, str] | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Query BV-BRC job details by task ID.

    Calls AppService.query_tasks via JSON-RPC to retrieve job metadata
    including status, parameters (with output_path/output_file), and
    optionally stdout/stderr logs.
    """
    token = None
    if headers and "Authorization" in headers:
        token = headers["Authorization"]

    if not token:
        return {
            "error": "Authentication required to query job details.",
            "errorType": "AUTHENTICATION_ERROR",
            "source": "bvbrc-service",
        }

    if not task_ids or not isinstance(task_ids, list):
        return {
            "error": "task_ids (list) parameter is required.",
            "errorType": "INVALID_PARAMETERS",
            "source": "bvbrc-service",
        }

    task_ids_str = [str(tid) for tid in task_ids]

    result = await query_tasks_raw(
        task_ids_str,
        token,
        api=_get_service_api(),
        fetch_stdout=stdout,
        fetch_stderr=stderr,
    )

    # Attach an output_check to every job. BV-BRC reports a job whose app
    # failed as "completed", and a re-run with the same output name leaves
    # the previous run's files in place — the analysis agent analysed such
    # stale files and called a failed job a success (2026-10-08).
    jobs = _iter_jobs(result)
    if jobs:
        checks = await asyncio.gather(
            *(check_job_outputs(job, config=config, headers=headers) for job in jobs),
            return_exceptions=True,
        )
        for job, check in zip(jobs, checks):
            if isinstance(check, Exception):
                check = {"verdict": "unknown", "message": f"Output check failed: {check}"}
            if check:
                job["output_check"] = check
    return result


def _iter_jobs(result: Any) -> list[dict[str, Any]]:
    """Job dicts inside query_tasks' ``{"data": [{task_id: job, ...}]}``."""
    jobs: list[dict[str, Any]] = []
    if not isinstance(result, dict):
        return jobs
    for block in result.get("data") or []:
        if isinstance(block, dict):
            jobs.extend(j for j in block.values() if isinstance(j, dict))
    return jobs


def _parse_time(value: Any) -> datetime | None:
    """Parse BV-BRC times ("2026-02-15 03:48:39" or "2026-02-11T18:41:21Z"), UTC."""
    if not value:
        return None
    text = str(value).strip().replace("T", " ").rstrip("Z")
    try:
        return datetime.strptime(text[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# Files written up to this long before the recorded start still count as
# this run's (clock skew between the app host and the workspace).
_STALE_SLACK = timedelta(minutes=5)


async def check_job_outputs(
    job: dict[str, Any],
    config: Any = None,
    headers: Dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """Check that a job's output folder holds this run's successful results.

    Returns ``{"verdict", "message", ...}`` or None when the job has no
    output_path/output_file. Verdicts:
      ok                    job_result says success, no stale files
      failed                job_result success=0, or JobFailed.html from this run
      outputs_from_other_run  the job_result object belongs to another task id
      stale_files           success, but some files predate this run
      no_outputs            output folder missing
      unknown               could not tell
    """
    params = job.get("parameters") or {}
    output_path = (params.get("output_path") or "").rstrip("/")
    output_file = params.get("output_file") or ""
    if not output_path or not output_file:
        return None
    task_id = str(job.get("id") or "")
    started = _parse_time(job.get("start_time"))
    output_folder = f"{output_path}/.{output_file}"

    parent, folder = await asyncio.gather(
        workspace_browse(path=output_path, num_results=500, config=config, headers=headers),
        workspace_browse(path=output_folder, num_results=500, config=config, headers=headers),
    )

    def items_of(res: Any) -> list[dict[str, Any]] | None:
        if not isinstance(res, dict) or res.get("error") or res.get("errorType"):
            return None
        inner = res.get("result", res)
        items = inner.get("items") if isinstance(inner, dict) else None
        return [i for i in items if isinstance(i, dict)] if isinstance(items, list) else None

    check: dict[str, Any] = {"output_folder": output_folder}

    job_result = next(
        (i for i in items_of(parent) or [] if i.get("name") == output_file and i.get("type") == "job_result"),
        None,
    )
    job_meta = (job_result or {}).get("job") or {}
    if job_meta:
        check["job_result"] = job_meta

    files = items_of(folder)
    if files is None:
        check["verdict"] = "no_outputs" if job_result is None else "unknown"
        check["message"] = (
            f"No output folder at {output_folder}. The job produced no results there; "
            "do not analyse other folders as if they were this job's output."
        )
        return check

    stale = []
    failed_report = None
    for f in files:
        created = _parse_time(f.get("creation_time"))
        # A re-run reuses existing subfolders and replaces the files in
        # them, so a folder's creation time says nothing about its contents.
        is_stale = bool(
            f.get("type") != "folder"
            and started and created and created < started - _STALE_SLACK
        )
        if f.get("name") == "JobFailed.html" and not is_stale:
            failed_report = f.get("path")
        if is_stale:
            stale.append(f.get("name"))
    check["file_count"] = len(files)
    if failed_report:
        check["job_failed_report"] = failed_report
    if stale:
        check["stale_files"] = len(stale)
        check["stale_examples"] = stale[:8]

    other_task = job_meta.get("task_id") and str(job_meta["task_id"]) != task_id
    success = job_meta.get("success")
    if other_task:
        check["verdict"] = "outputs_from_other_run"
        check["message"] = (
            f"The job_result at this path belongs to task {job_meta['task_id']}, not {task_id}: "
            "a later or earlier run wrote to the same output name. Tell the user; do not "
            "present these files as this job's results."
        )
    elif success == 0 or failed_report:
        check["verdict"] = "failed"
        check["message"] = (
            "This job FAILED even though its status is 'completed' "
            + ("(job_result success=0)" if success == 0 else "")
            + (f"; read {failed_report} for the error" if failed_report else "")
            + ". "
            + (f"{len(stale)} file(s) in the folder predate this run and belong to an earlier run. "
               if stale else "")
            + "Report the failure and its cause; do not analyse leftover files as results."
        )
    elif stale:
        check["verdict"] = "stale_files"
        check["message"] = (
            f"{len(stale)} file(s) predate this run (left by an earlier run with the same "
            "output name). Use only files created by this run, and tell the user."
        )
    elif success == 1:
        check["verdict"] = "ok"
        check["message"] = "job_result reports success and all files are from this run."
    else:
        check["verdict"] = "unknown"
        check["message"] = "No job_result success flag found; check the output files for errors."
    return check


# ---------------------------------------------------------------------------
# Dispatch table: tool name -> async callable
# ---------------------------------------------------------------------------
TOOL_DISPATCH: Dict[str, Any] = {
    "workspace_browse": workspace_browse,
    "get_file_metadata": get_file_metadata,
    "read_file_preview": read_file_preview,
    "search_data": search_data,
    "get_expected_outputs": _get_expected_outputs_async,
    "get_job_details": get_job_details,
    "find_similar_genomes": find_similar_genomes,
    "search_literature": search_literature,
}


async def execute_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    timeout_seconds: float = 30.0,
    config: Any = None,
    headers: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Execute a tool by name with the given arguments.

    Delegates to the shared ``execute_tool`` with this agent's dispatch table.
    """
    return await _shared_execute_tool(
        tool_name=tool_name,
        arguments=arguments,
        dispatch_table=TOOL_DISPATCH,
        timeout_seconds=timeout_seconds,
        config=config,
        headers=headers,
    )
