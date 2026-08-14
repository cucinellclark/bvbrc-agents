"""
Shared group-creation tool for BV-BRC genome and feature groups.

Creates a group from a Solr query: runs the query to collect matching IDs,
then writes the group to the user's workspace via the Workspace JSON-RPC API.

Any agent can import ``create_group`` to build groups in one step without
needing the LLM to separately query data and then construct a group.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

from shared.tools._mcp_imports import get_group_functions, get_json_rpc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GROUP_TYPE_CONFIG = {
    "genome_group": {
        "id_field": "genome_id",
        "default_collection": "genome",
        "display_name": "genome group",
    },
    "feature_group": {
        "id_field": "feature_id",
        "default_collection": "genome_feature",
        "display_name": "feature group",
    },
}

_DEFAULT_LIMIT = 500
_MAX_LIMIT = 25000
_BATCH_SIZE = 1000


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_token(headers: Optional[Dict[str, str]]) -> Optional[str]:
    if headers and "Authorization" in headers:
        return headers["Authorization"]
    return None


def _get_solr_query_fn():
    """Lazy import of agent_data_tools.solr_query."""
    import importlib
    from pathlib import Path

    agent_data_path = str(
        Path(__file__).resolve().parent.parent.parent
        / "agents"
        / "data_agent"
    )
    if agent_data_path not in sys.path:
        sys.path.insert(0, agent_data_path)

    mod = importlib.import_module("mcp_tools.agent_data_tools")
    return mod.solr_query


async def _fetch_ids_by_query(
    collection: str,
    query: str,
    id_field: str,
    limit: int,
    token: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch unique IDs matching a Solr query using cursor pagination.

    Returns:
        Dict with ``ids`` (list), ``total_matching`` (int from numFound),
        and ``error`` (str) if something went wrong.
    """
    solr_query_fn = _get_solr_query_fn()

    # First, get the total count so we can report it
    count_result = await solr_query_fn(
        collection=collection,
        query=query,
        count_only=True,
        token=token,
        base_url=base_url,
    )

    if "error" in count_result:
        return count_result

    total_matching = count_result.get("numFound", 0)
    if total_matching == 0:
        return {
            "ids": [],
            "total_matching": 0,
            "error": f"Query returned 0 results in the '{collection}' collection. "
            "Check your query and try again.",
        }

    # Fetch IDs in batches via cursor pagination
    all_ids: list[str] = []
    seen: set[str] = set()
    cursor_id: Optional[str] = "*"

    while cursor_id and len(all_ids) < limit:
        batch_limit = min(_BATCH_SIZE, limit - len(all_ids))

        result = await solr_query_fn(
            collection=collection,
            query=query,
            select=[id_field],
            limit=batch_limit,
            cursor_id=cursor_id,
            token=token,
            base_url=base_url,
        )

        if "error" in result:
            # If we already have some IDs, continue with what we have
            if all_ids:
                break
            return result

        docs = result.get("results", [])
        if not docs:
            break

        for doc in docs:
            if isinstance(doc, dict) and id_field in doc:
                id_val = str(doc[id_field])
                if id_val not in seen:
                    seen.add(id_val)
                    all_ids.append(id_val)
                    if len(all_ids) >= limit:
                        break

        cursor_id = result.get("nextCursorId")

    return {
        "ids": all_ids,
        "total_matching": total_matching,
    }


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------


async def create_group(
    group_name: str,
    group_type: str,
    collection: str,
    query: str,
    limit: int = _DEFAULT_LIMIT,
    config: Any = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Create a genome or feature group from a Solr query.

    Runs the query against BV-BRC Solr to fetch matching IDs (up to
    *limit*), then creates the group in the user's workspace.

    Args:
        group_name: Name for the new group.
        group_type: ``"genome_group"`` or ``"feature_group"``.
        collection: Solr collection to query (e.g. ``"genome"``,
                    ``"genome_feature"``).
        query: Solr query string (Lucene syntax, same as ``search_data``).
        limit: Maximum number of IDs to include (default 500, max 25000).
            Use this to cap group size for services with input limits.
        config: Agent configuration object.
        headers: HTTP headers with auth token.

    Returns:
        Dict with ``name``, ``path``, ``count``, ``total_matching``,
        ``group_type``, ``query_used``, ``message``; or ``error``.
    """
    # --- Validate inputs ---
    if not group_name or not group_name.strip():
        return {"error": "Group name is required."}

    group_name = group_name.strip()

    if group_type not in _GROUP_TYPE_CONFIG:
        return {
            "error": f"Invalid group_type: '{group_type}'. "
            f"Must be one of: {', '.join(_GROUP_TYPE_CONFIG.keys())}",
        }

    token = _extract_token(headers)
    if not token:
        return {"error": "Authentication required. No auth token provided."}

    # Clamp limit
    limit = max(1, min(limit, _MAX_LIMIT))

    type_cfg = _GROUP_TYPE_CONFIG[group_type]
    id_field = type_cfg["id_field"]
    display = type_cfg["display_name"]

    base_url = getattr(config, "bvbrc_api_url", None)

    # --- Step 1: Fetch IDs by query ---
    fetch_result = await _fetch_ids_by_query(
        collection=collection,
        query=query,
        id_field=id_field,
        limit=limit,
        token=token,
        base_url=base_url,
    )

    if "error" in fetch_result and not fetch_result.get("ids"):
        return fetch_result

    ids = fetch_result["ids"]
    total_matching = fetch_result.get("total_matching", len(ids))

    if not ids:
        return {
            "error": f"Query matched {total_matching} records but no "
            f"'{id_field}' values could be extracted.",
            "query": query,
            "collection": collection,
        }

    # --- Step 2: Create the group via Workspace API ---
    gf = get_group_functions(getattr(config, "mcp_server_path", None))
    json_rpc_mod = get_json_rpc(getattr(config, "mcp_server_path", None))

    ws_url = (
        getattr(config, "bvbrc_workspace_url", None)
        or "https://p3.theseed.org/services/Workspace"
    )
    ws_timeout = getattr(config, "tool_timeout_seconds", 30)
    api = json_rpc_mod.JsonRpcCaller(service_url=ws_url, timeout=ws_timeout)

    result = await gf.create_group(
        api=api,
        name=group_name,
        id_list=ids,
        group_type=group_type,
        token=token,
    )

    if "error" in result:
        return result

    # --- Build response ---
    count = len(ids)
    path = result.get("path", "")

    if count < total_matching:
        message = (
            f"Created {display} '{group_name}' with {count} {id_field}(s) "
            f"(limited from {total_matching:,} total matches)."
        )
    else:
        message = (
            f"Created {display} '{group_name}' with {count} {id_field}(s)."
        )

    return {
        "name": group_name,
        "path": path,
        "count": count,
        "total_matching": total_matching,
        "group_type": group_type,
        "query_used": query,
        "collection": collection,
        "limit_applied": limit if count < total_matching else None,
        "message": message,
        "source": "bvbrc-workspace",
    }
