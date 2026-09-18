"""
Shared group tools for BV-BRC genome and feature groups.

Provides three tools for the agent toolbox:

  ``list_groups``     – List all genome or feature groups in the user's workspace.
  ``get_group_ids``   – Retrieve the member IDs of a group by name.
  ``create_group``    – Create a group from a Solr query.

All three delegate to the core implementations in
``mcp_server/functions/group_functions.py`` via the lazy-import bridge.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

from shared.tools._mcp_imports import get_group_functions, get_json_rpc


# ---------------------------------------------------------------------------
# Shared helpers (api / token resolution, same pattern as workspace.py)
# ---------------------------------------------------------------------------

_api_instance = None


def _get_api(config: Any = None) -> Any:
    """Create or reuse a ``JsonRpcCaller`` for workspace API calls."""
    global _api_instance
    if _api_instance is None:
        json_rpc_mod = get_json_rpc(getattr(config, "mcp_server_path", None))
        ws_url = (
            getattr(config, "bvbrc_workspace_url", None)
            or "https://p3.theseed.org/services/Workspace"
        )
        timeout = getattr(config, "tool_timeout_seconds", 30)
        _api_instance = json_rpc_mod.JsonRpcCaller(
            service_url=ws_url,
            timeout=timeout,
        )
    return _api_instance


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
# Public tool: list_groups
# ---------------------------------------------------------------------------


async def list_groups(
    group_type: str,
    config: Any = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """List all genome or feature groups in the user's workspace.

    Searches the user's entire home directory for groups of the requested
    type.  Returns a flat list of group names, paths, and creation dates.

    Args:
        group_type: ``"genome_group"`` or ``"feature_group"``.
        config: Agent configuration object.
        headers: HTTP headers with auth token.

    Returns:
        Dict with ``groups`` (list of name/path dicts), ``count``, and
        ``message``; or ``error``.
    """
    if group_type not in _GROUP_TYPE_CONFIG:
        return {
            "error": f"Invalid group_type: '{group_type}'. "
            f"Must be one of: {', '.join(_GROUP_TYPE_CONFIG.keys())}",
        }

    token = _extract_token(headers)
    if not token:
        return {"error": "Authentication required. No auth token provided."}

    gf = get_group_functions(getattr(config, "mcp_server_path", None))
    api = _get_api(config)

    try:
        result = await gf.list_groups(
            api=api,
            group_type=group_type,
            token=token,
        )
    except Exception as e:
        return {
            "error": f"Failed to list groups: {type(e).__name__}: {e}",
            "source": "bvbrc-workspace",
        }

    if "error" in result:
        return result

    # Flatten the MCP response to a simpler structure for agents.
    # The MCP version returns {result: {items, ui_grid, ...}, call: {...}}.
    inner = result.get("result", result)
    items = inner.get("items", [])
    group_names = inner.get("group_names", [n.get("name", "") for n in items])
    count = inner.get("count", len(items))
    display = _GROUP_TYPE_CONFIG[group_type]["display_name"]

    if group_names:
        message = f"Found {count} {display}(s): {', '.join(group_names)}."
    else:
        message = f"No {display}s found in workspace."

    return {
        "groups": [{"name": n} for n in group_names],
        "count": count,
        "group_type": group_type,
        "message": message,
        "source": "bvbrc-workspace",
    }


# ---------------------------------------------------------------------------
# Public tool: get_group_ids
# ---------------------------------------------------------------------------


async def get_group_ids(
    group_name: str,
    group_type: str,
    config: Any = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Get the member IDs of a genome or feature group by name.

    The group is looked up by name automatically — the system searches
    the user's default group folder and will find the group even if the
    name casing doesn't match exactly.

    If the name is ambiguous (matches multiple groups), returns a list of
    candidates so the user can clarify.

    Args:
        group_name: Name of the group (e.g. ``"My E. coli genomes"``).
            Do NOT provide a workspace path — just the name.
        group_type: ``"genome_group"`` or ``"feature_group"``.
        config: Agent configuration object.
        headers: HTTP headers with auth token.

    Returns:
        Dict with ``genome_ids`` or ``feature_ids`` (list), ``count``,
        ``name``, ``path``; or ``error`` / disambiguation candidates.
    """
    if not group_name or not group_name.strip():
        return {"error": "Group name is required."}

    if group_type not in _GROUP_TYPE_CONFIG:
        return {
            "error": f"Invalid group_type: '{group_type}'. "
            f"Must be one of: {', '.join(_GROUP_TYPE_CONFIG.keys())}",
        }

    token = _extract_token(headers)
    if not token:
        return {"error": "Authentication required. No auth token provided."}

    gf = get_group_functions(getattr(config, "mcp_server_path", None))
    api = _get_api(config)

    try:
        return await gf.get_group_ids(
            api=api,
            name=group_name.strip(),
            group_type=group_type,
            token=token,
        )
    except Exception as e:
        return {
            "error": f"Failed to get group IDs: {type(e).__name__}: {e}",
            "source": "bvbrc-workspace",
        }


# ---------------------------------------------------------------------------
# Public tool: create_group
# ---------------------------------------------------------------------------


async def create_group(
    group_name: str,
    group_type: str,
    collection: str,
    query: str,
    limit: int = _DEFAULT_LIMIT,
    if_exists: str = "error",
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
        if_exists: What to do when a group of that name already exists:
            ``"error"`` (default, returns ``errorType: ALREADY_EXISTS``),
            ``"append"`` (add the new IDs to the existing group, deduplicated)
            or ``"replace"`` (overwrite it).  There is no separate
            add-to-group API; append is read-merge-overwrite, as on the
            website.
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

    if_exists = (if_exists or "error").strip().lower()
    if if_exists not in ("error", "append", "replace"):
        return {
            "error": f"if_exists must be 'error', 'append' or 'replace' (got {if_exists!r}).",
            "errorType": "INVALID_PARAMETERS",
        }

    result = await gf.create_group(
        api=api,
        name=group_name,
        id_list=ids,
        group_type=group_type,
        token=token,
        if_exists=if_exists,
    )

    if "error" in result:
        return result

    # --- Build response ---
    action = result.get("action", "created")
    count = result.get("count", len(ids))
    path = result.get("path", "")
    limited = len(ids) < total_matching
    limit_note = f" (query limited to {len(ids)} of {total_matching:,} matches)" if limited else ""

    if action == "appended":
        message = (
            f"Added {result.get('added', 0)} new {id_field}(s) to {display} "
            f"'{group_name}' ({result.get('already_present', 0)} already present, "
            f"{count} total){limit_note}."
        )
    elif action == "replaced":
        was = result.get("previous_count")
        message = (
            f"Replaced {display} '{group_name}': now {count} {id_field}(s)"
            f"{f' (was {was})' if was is not None else ''}{limit_note}."
        )
    else:
        message = f"Created {display} '{group_name}' with {count} {id_field}(s){limit_note}."

    return {
        "name": group_name,
        "path": path,
        "count": count,
        "action": action,
        "added": result.get("added", count),
        **({"already_present": result["already_present"]} if "already_present" in result else {}),
        **({"previous_count": result["previous_count"]} if "previous_count" in result else {}),
        "total_matching": total_matching,
        "group_type": group_type,
        "query_used": query,
        "collection": collection,
        "limit_applied": limit if count < total_matching else None,
        "message": message,
        "source": "bvbrc-workspace",
    }
