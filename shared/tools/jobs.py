"""
Shared job listing tool.

Wraps the MCP server's ``service_functions.list_jobs`` to enumerate
user jobs with optional filtering, sorting, and pagination.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from shared.tools._mcp_imports import get_service_functions


def _extract_token(
    config: Any = None,
    headers: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Extract BV-BRC auth token from config or headers."""
    token = getattr(config, "bvbrc_auth_token", None)
    if token:
        return token
    if headers:
        return headers.get("Authorization")
    return None


def _extract_user_id(token: str) -> Optional[str]:
    """Extract user ID from a BV-BRC auth token."""
    if not token:
        return None
    try:
        return token.split("|")[0].replace("un=", "")
    except Exception:
        return None


async def list_jobs(
    limit: int = 20,
    offset: int = 0,
    sort_by: str = "submit_time",
    sort_dir: str = "desc",
    status: Optional[str] = None,
    service: Optional[str] = None,
    search: Optional[str] = None,
    include_archived: bool = False,
    config: Any = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """List user jobs with optional filtering and sorting.

    Args:
        limit: Maximum jobs to return (default 20).
        offset: Number of jobs to skip for pagination.
        sort_by: Field to sort by (default ``submit_time``).
        sort_dir: Sort direction (``asc`` or ``desc``).
        status: Filter by job status (e.g. ``completed``, ``failed``).
        service: Filter by service name (e.g. ``genome_assembly``).
        search: Free-text search filter.
        include_archived: Include archived jobs.
        config: Agent configuration (provides auth token, MCP path).
        headers: HTTP headers (auth fallback).

    Returns:
        Dict with ``items`` (list of job summaries), ``total``, ``count``,
        pagination metadata, and a ``ui_grid`` payload.
    """
    token = _extract_token(config, headers)
    if not token:
        return {
            "error": "No authentication token available.",
            "source": "shared-tools",
        }

    user_id = _extract_user_id(token)

    svc_fn = get_service_functions(getattr(config, "mcp_server_path", None))

    # The MCP server's JsonRpcCaller is stored on the service_functions module
    # as ``_api``.  It is set during MCP server startup by
    # ``register_service_tools()``.  For the shared-tools path we need the
    # ``list_jobs`` async function which requires an ``api`` argument.
    # The lazy-import bridge gives us the module; ``list_jobs`` is a
    # module-level async function that accepts ``api`` as the first arg.
    api = getattr(svc_fn, "_api", None)
    if api is None:
        # Fallback: import the JsonRpcCaller from the MCP common layer.
        from shared.tools._mcp_imports import get_json_rpc
        json_rpc = get_json_rpc(getattr(config, "mcp_server_path", None))
        # Build a caller pointed at the BV-BRC app service endpoint
        bvbrc_url = getattr(config, "bvbrc_workspace_url", "https://p3.theseed.org/services/app_service")
        # Replace workspace_service path with app_service if needed
        if "workspace_service" in bvbrc_url:
            bvbrc_url = bvbrc_url.replace("workspace_service", "app_service")
        elif not bvbrc_url.endswith("app_service"):
            bvbrc_url = "https://p3.theseed.org/services/app_service"
        api = json_rpc.JsonRpcCaller(bvbrc_url)

    try:
        result = await svc_fn.list_jobs(
            api=api,
            token=token,
            user_id=user_id,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            sort_dir=sort_dir,
            status=status,
            service=service,
            search=search,
            include_archived=include_archived,
        )

        # Attach a link to the BV-BRC jobs page so the LLM can include
        # a clickable markdown link in its response.
        if isinstance(result, dict) and not result.get("error"):
            result["jobs_page_url"] = "https://www.bv-brc.org/job/"

        return result

    except Exception as e:
        return {
            "error": f"list_jobs failed: {type(e).__name__}: {str(e)}",
            "source": "shared-tools",
        }
