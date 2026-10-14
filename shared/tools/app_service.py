"""BV-BRC AppService access shared by tools.

``query_app_tasks`` is the ``AppService.query_tasks`` call behind the analysis
agent's ``get_job_details``.  It lives in ``shared/`` so that
``shared/tools/rerun.py`` can read a job's original parameters without
importing an agent package.
"""

from __future__ import annotations

from typing import Any, Dict

from shared.tools import _mcp_imports

APP_SERVICE_URL = "https://p3.theseed.org/services/app_service"

_service_api_instance = None


def get_app_service_api() -> Any:
    """Create or reuse a JsonRpcCaller for the BV-BRC app service."""
    global _service_api_instance
    if _service_api_instance is None:
        json_rpc_mod = _mcp_imports.get_json_rpc()
        _service_api_instance = json_rpc_mod.JsonRpcCaller(
            service_url=APP_SERVICE_URL,
            timeout=30,
        )
    return _service_api_instance


async def query_tasks_raw(
    task_ids: list[str],
    token: str,
    api: Any = None,
    fetch_stdout: bool = False,
    fetch_stderr: bool = False,
) -> Dict[str, Any]:
    """``AppService.query_tasks`` as the MCP function returns it.

    The shape is ``{"data": [{task_id: job, ...}], "source": ...}`` or an
    ``{"error": ...}`` dict.
    """
    service_fn = _mcp_imports.get_service_functions()
    return await service_fn.query_tasks(
        api=api if api is not None else get_app_service_api(),
        token=token,
        params={"task_ids": [str(t) for t in task_ids]},
        fetch_stdout=fetch_stdout,
        fetch_stderr=fetch_stderr,
    )


def jobs_by_task_id(result: Any) -> Dict[str, Dict[str, Any]]:
    """``{task_id: job}`` from a ``query_tasks`` result (ids as strings)."""
    jobs: Dict[str, Dict[str, Any]] = {}
    if not isinstance(result, dict):
        return jobs
    for block in result.get("data") or []:
        if isinstance(block, dict):
            for task_id, job in block.items():
                if isinstance(job, dict):
                    jobs[str(task_id)] = job
    return jobs


async def query_app_tasks(task_ids: list[str], token: str, **kwargs: Any) -> Dict[str, Dict[str, Any]]:
    """``{task_id: job}`` for the given ids; ids the user cannot see are absent.

    Raises ``RuntimeError`` when the service returned an error.
    """
    result = await query_tasks_raw(task_ids, token, **kwargs)
    if isinstance(result, dict) and result.get("error"):
        raise RuntimeError(str(result["error"]))
    return jobs_by_task_id(result)
