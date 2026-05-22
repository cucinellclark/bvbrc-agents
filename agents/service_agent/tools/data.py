"""
Tool wrapper for BV-BRC Solr data queries.

Thin async wrapper that translates agent tool-call arguments into direct
Solr API queries. Used by the Service Agent to find genome IDs, feature IDs,
and other data needed as inputs to service planning.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional
from types import ModuleType

from service_agent.models import AgentConfig


# ---------------------------------------------------------------------------
# MCP server import helper
# ---------------------------------------------------------------------------

_data_functions: Optional[ModuleType] = None
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


def _get_data_functions(config: AgentConfig | None = None) -> ModuleType:
    global _data_functions
    if _data_functions is None:
        _ensure_path(config)
        from functions import data_functions
        _data_functions = data_functions
    return _data_functions


def _extract_token(headers: Optional[Dict[str, str]]) -> Optional[str]:
    if headers and "Authorization" in headers:
        return headers["Authorization"]
    return None


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------

async def search_data(
    collection: str,
    query: str,
    select: Optional[List[str]] = None,
    limit: Optional[int] = 25,
    count_only: Optional[bool] = False,
    config: Optional[AgentConfig] = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Query a BV-BRC Solr collection for data needed by service planning.

    Args:
        collection: Solr collection name (e.g., 'genome').
        query: Solr query string.
        select: Fields to return.
        limit: Maximum records to return (capped at 50 for service planning).
        count_only: If True, return only the count.
        config: Agent configuration.
        headers: HTTP headers with auth token.

    Returns:
        Dict with query results.
    """
    cfg = config or AgentConfig()
    data_fn = _get_data_functions(config)
    token = _extract_token(headers)

    if limit is None:
        limit = 25
    if count_only is None:
        count_only = False

    limit = min(limit, 50)

    select_str = None
    if select:
        select_str = ",".join(select)

    options: Dict[str, Any] = {}
    if select_str:
        options["select"] = select_str
    if limit:
        options["limit"] = limit

    try:
        base_url = cfg.bvbrc_api_url

        # Build headers for auth
        req_headers = {}
        if token:
            req_headers["Authorization"] = token

        result = await data_fn.query_direct(
            core=collection,
            filter_str=query,
            options=options,
            base_url=base_url,
            headers=req_headers if req_headers else None,
            countOnly=count_only,
            batch_size=limit,
        )

        if count_only:
            return {
                "numFound": result.get("numFound", result.get("count", 0)),
            }

        return {
            "results": result.get("results", []),
            "count": len(result.get("results", [])),
            "numFound": result.get("numFound", 0),
        }

    except Exception as e:
        return {
            "error": f"Data search failed: {type(e).__name__}: {str(e)}",
            "collection": collection,
            "query": query,
        }
