"""
Group Resolution and Management Functions

Provides shared utilities for resolving, listing, creating, and reading
genome groups and feature groups in the BV-BRC workspace.

All group path resolution is centralized here so that tool layers and
service tools can call a single function instead of duplicating
name-to-path logic.
"""

from common.json_rpc import JsonRpcCaller
from functions.workspace_functions import workspace_search
from typing import List, Optional, Dict, Any
import json
import sys


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GROUP_TYPE_CONFIG = {
    "genome_group": {
        "default_folder": "Genome Groups",
        "id_field": "genome_id",
        "display_name": "genome group",
    },
    "feature_group": {
        "default_folder": "Feature Groups",
        "id_field": "feature_id",
        "display_name": "feature group",
    },
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_user_id_from_token(token: str) -> Optional[str]:
    """Extract user ID from a BV-BRC auth token."""
    if not token:
        return None
    try:
        return token.split("|")[0].replace("un=", "")
    except Exception:
        return None


def _user_home(user_id: str) -> str:
    return f"/{user_id}/home"


def _default_group_folder(user_id: str, group_type: str) -> str:
    """Return the default workspace folder for the given group type."""
    cfg = GROUP_TYPE_CONFIG[group_type]
    return f"{_user_home(user_id)}/{cfg['default_folder']}"


def _default_group_path(user_id: str, group_type: str, name: str) -> str:
    """Return the default full path for a group by name."""
    return f"{_default_group_folder(user_id, group_type)}/{name}"


def _normalize_name(name: str) -> str:
    """Lowercase + strip for comparison."""
    return (name or "").strip().lower()


def _build_grid_payload(
    entity_type: str,
    items: list = None,
    result_type: str = "list_result",
    pagination: dict = None,
    sort: dict = None,
    source: str = "bvbrc-workspace",
    columns: list = None,
    selectable: bool = True,
    multi_select: bool = True,
    sortable: bool = True,
) -> dict:
    """Build a structured grid payload for frontend rendering."""
    return {
        "schema_version": "1.0",
        "entity_type": entity_type,
        "source": source,
        "result_type": result_type,
        "capabilities": {
            "selectable": selectable,
            "multi_select": multi_select,
            "sortable": sortable,
        },
        "pagination": pagination,
        "sort": sort,
        "columns": columns or [],
        "items": items or [],
    }


# ---------------------------------------------------------------------------
# Core workspace calls (thin wrappers around JSON-RPC)
# ---------------------------------------------------------------------------


async def _workspace_get(api: JsonRpcCaller, path: str, token: str, metadata_only: bool = False) -> Dict[str, Any]:
    """
    Fetch a single workspace object. Returns a dict with 'metadata' and
    optionally 'data', or an 'error' key on failure.
    """
    try:
        import requests as _requests  # for unquote only
        path = _requests.utils.unquote(path)

        result = await api.acall(
            "Workspace.get",
            {"objects": [path], "metadata_only": metadata_only},
            1,
            token,
        )

        if not result or not result[0] or not result[0][0] or not result[0][0][0] or not result[0][0][0][4]:
            return {"error": "Object not found", "errorType": "NOT_FOUND", "source": "bvbrc-workspace"}

        meta_array = result[0][0][0]
        metadata = {
            "name": meta_array[0],
            "type": meta_array[1],
            "path": meta_array[2],
            "creation_time": meta_array[3],
            "id": meta_array[4],
            "owner_id": meta_array[5],
            "size": meta_array[6],
            "userMeta": meta_array[7],
            "autoMeta": meta_array[8],
        }

        if metadata_only:
            return {"metadata": metadata, "source": "bvbrc-workspace"}

        data = result[0][0][1]
        return {"metadata": metadata, "data": data, "source": "bvbrc-workspace"}

    except Exception as e:
        if "Object not found" in str(e):
            return {"error": "Object not found", "errorType": "NOT_FOUND", "source": "bvbrc-workspace"}
        return {"error": f"Workspace error: {e}", "errorType": "API_ERROR", "source": "bvbrc-workspace"}


async def _workspace_ls_groups(api: JsonRpcCaller, folder_path: str, group_type: str, token: str) -> Dict[str, Any]:
    """
    List all objects of *group_type* under *folder_path* (non-recursive).
    Returns {'items': [...], 'count': N} or {'error': ...}.
    """
    try:
        result = await api.acall(
            "Workspace.ls",
            {
                "paths": [folder_path],
                "recursive": False,
                "includeSubDirs": False,
                "query": {"type": [group_type]},
            },
            1,
            token,
        )

        # Flatten the nested response
        flat: list = []
        items = result if isinstance(result, list) else [result]
        for entry in items:
            if isinstance(entry, dict):
                for val in entry.values():
                    if isinstance(val, list):
                        flat.extend(val)
                    else:
                        flat.append(val)
            else:
                flat.append(entry)

        # Filter to actual list items and skip hidden files
        flat = [
            item for item in flat
            if isinstance(item, list) and item and not str(item[0]).startswith(".")
        ]

        return {"items": flat, "count": len(flat), "source": "bvbrc-workspace"}

    except Exception as e:
        if "Object not found" in str(e):
            return {"items": [], "count": 0, "source": "bvbrc-workspace"}
        return {"error": f"Error listing groups: {e}", "errorType": "API_ERROR", "source": "bvbrc-workspace"}


# ---------------------------------------------------------------------------
# Public API: resolve_group_path
# ---------------------------------------------------------------------------


async def resolve_group_path(
    api: JsonRpcCaller,
    name: str,
    group_type: str,
    token: str,
) -> Dict[str, Any]:
    """
    Resolve a group name to a workspace path using a two-tier strategy:

    1. **Exact match** – check ``/{user}/home/{GroupFolder}/{name}``
    2. **Fuzzy match** – list the default folder and do case-insensitive /
       substring matching.

    Returns
    -------
    On single match::

        {"path": "/user/home/Genome Groups/Ecoli", "name": "Ecoli",
         "match_type": "exact"|"fuzzy", "source": "bvbrc-workspace"}

    On multiple fuzzy matches::

        {"candidates": [...], "match_type": "ambiguous",
         "message": "Multiple groups matched ...", "source": "bvbrc-workspace"}

    On no match::

        {"error": "...", "errorType": "NOT_FOUND", "source": "bvbrc-workspace"}
    """
    if not name or not name.strip():
        return {
            "error": f"{GROUP_TYPE_CONFIG[group_type]['display_name']} name is required",
            "errorType": "INVALID_PARAMETERS",
            "source": "bvbrc-workspace",
        }

    user_id = _get_user_id_from_token(token)
    if not user_id:
        return {"error": "Could not extract user ID from token", "errorType": "AUTHENTICATION_ERROR", "source": "bvbrc-workspace"}

    name = name.strip()

    # ── Tier 1: exact match in default folder ──
    exact_path = _default_group_path(user_id, group_type, name)
    result = await _workspace_get(api, exact_path, token, metadata_only=True)
    if "error" not in result:
        return {
            "path": f"{result['metadata']['path']}{result['metadata']['name']}",
            "name": result["metadata"]["name"],
            "match_type": "exact",
            "source": "bvbrc-workspace",
        }

    # ── Tier 2: fuzzy match across entire workspace ──
    home = _user_home(user_id)
    search_result = await workspace_search(
        api,
        paths=[home],
        file_types=[group_type],
        token=token,
    )

    if "error" in search_result:
        return {
            "error": f"Group '{name}' not found and could not search the workspace",
            "errorType": "NOT_FOUND",
            "details": search_result["error"],
            "source": "bvbrc-workspace",
        }

    items = search_result.get("items", [])
    if not items:
        return {
            "error": f"No {GROUP_TYPE_CONFIG[group_type]['display_name']}s found in the workspace.",
            "errorType": "NOT_FOUND",
            "source": "bvbrc-workspace",
        }

    norm_query = _normalize_name(name)
    candidates = []
    for item in items:
        item_name = str(item[0])
        item_path = f"{item[2]}{item_name}"  # path prefix + name
        norm_item = _normalize_name(item_name)

        # Case-insensitive exact match
        if norm_item == norm_query:
            return {
                "path": item_path,
                "name": item_name,
                "match_type": "fuzzy",
                "source": "bvbrc-workspace",
            }

        # Substring match (query in item name or item name in query)
        if norm_query in norm_item or norm_item in norm_query:
            candidates.append({
                "name": item_name,
                "path": item_path,
                "creation_time": item[3] if len(item) > 3 else None,
            })

    if len(candidates) == 1:
        return {
            "path": candidates[0]["path"],
            "name": candidates[0]["name"],
            "match_type": "fuzzy",
            "source": "bvbrc-workspace",
        }

    if len(candidates) > 1:
        group_names = [c["name"] for c in candidates]
        return {
            "candidates": candidates,
            "match_type": "ambiguous",
            "message": f"Multiple {GROUP_TYPE_CONFIG[group_type]['display_name']}s matched '{name}': {', '.join(group_names)}. Please specify the exact name.",
            "source": "bvbrc-workspace",
        }

    # No match at all – list available groups to help the user
    available = [str(item[0]) for item in items]
    return {
        "error": f"No {GROUP_TYPE_CONFIG[group_type]['display_name']} matching '{name}' was found.",
        "errorType": "NOT_FOUND",
        "available_groups": available,
        "message": f"Available {GROUP_TYPE_CONFIG[group_type]['display_name']}s: {', '.join(available)}",
        "source": "bvbrc-workspace",
    }


# ---------------------------------------------------------------------------
# Public API: list_groups
# ---------------------------------------------------------------------------


async def list_groups(
    api: JsonRpcCaller,
    group_type: str,
    token: str,
    folder: Optional[str] = None,
    tool_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    List all groups of the given type in a folder.

    If *folder* is not provided, uses the default group folder
    (e.g. ``/{user}/home/Genome Groups``).

    Returns a structured envelope with ``result``, ``call``, and ``ui_grid``
    so that the frontend can render groups in an interactive grid and the
    LLM receives only a concise summary.

    Returns
    -------
    ::

        {
            "result": {
                "items": [...],
                "tool_name": "list_genome_groups",
                "result_type": "list_result",
                "count": N,
                "path": "/user/home/Genome Groups",
                "source": "bvbrc-workspace",
                "ui_grid": { ... }
            },
            "call": {
                "tool": "list_genome_groups",
                "backend_method": "Workspace.ls",
                "arguments_executed": { ... },
                "replayable": true,
                "replay": { ... }
            }
        }
    """
    user_id = _get_user_id_from_token(token)
    if not user_id:
        return {"error": "Could not extract user ID from token", "errorType": "AUTHENTICATION_ERROR", "source": "bvbrc-workspace"}

    if folder:
        # Resolve relative paths
        if not folder.startswith("/"):
            folder = f"{_user_home(user_id)}/{folder}"
    else:
        folder = _user_home(user_id)

    search_result = await workspace_search(
        api,
        paths=[folder],
        file_types=[group_type],
        token=token,
    )
    if "error" in search_result:
        return search_result

    items = search_result.get("items", [])
    display = GROUP_TYPE_CONFIG[group_type]["display_name"]

    # Derive tool_name if not provided
    if not tool_name:
        tool_name = f"list_{group_type}s"  # e.g. "list_genome_groups"

    groups = []
    group_names = []
    for item in items:
        name = str(item[0])
        group_names.append(name)
        groups.append({
            "name": name,
            "path": f"{item[2]}{name}" if len(item) > 2 else name,
            "type": item[1] if len(item) > 1 else group_type,
            "creation_time": item[3] if len(item) > 3 else None,
            "size": item[6] if len(item) > 6 else None,
        })

    # Build a concise LLM snapshot: count + names only.
    # The full data lives in ui_grid for the frontend / replay.
    if groups:
        names_str = ", ".join(group_names)
        message = (
            f"Found {len(groups)} {display}(s): {names_str}. "
            "The complete details are displayed in the table for the user."
        )
    else:
        message = f"No {display}s found in {folder}."

    return {
        "result": {
            # Snapshot for the LLM — names only, no paths or metadata
            "items": [{"name": g["name"]} for g in groups],
            "tool_name": tool_name,
            "result_type": "search_result",
            "count": len(groups),
            "group_names": group_names,
            "message": message,
            "is_snapshot": True,
            "path": folder,
            "source": "bvbrc-workspace",
            # Full data for the frontend / replay
            "ui_grid": _build_grid_payload(
                entity_type=group_type,
                items=groups,
                result_type="search_result",
                source="bvbrc-workspace",
                sort={"sort_by": "name", "sort_order": "asc"},
                pagination={"limit": len(groups), "offset": 0, "has_more": False},
                columns=[
                    {"key": "name", "label": "Name", "sortable": True},
                    {"key": "type", "label": "Type", "sortable": True},
                    {"key": "creation_time", "label": "Created", "sortable": True},
                    {"key": "size", "label": "Size", "sortable": True},
                ],
            ),
        },
        "call": {
            "tool": tool_name,
            "backend_method": "Workspace.ls",
            "arguments_executed": {
                "group_type": group_type,
                "folder": folder,
            },
            "replayable": True,
            "replay": {
                "path": folder,
                "group_type": group_type,
            },
        },
    }


# ---------------------------------------------------------------------------
# Public API: get_group_ids
# ---------------------------------------------------------------------------


async def get_group_ids(
    api: JsonRpcCaller,
    name: str,
    group_type: str,
    token: str,
) -> Dict[str, Any]:
    """
    Resolve a group by name and return its member IDs.

    Combines :func:`resolve_group_path` with a data fetch.

    Returns
    -------
    ::

        {
            "genome_ids": [...],  # or "feature_ids"
            "count": N,
            "name": "GroupName",
            "path": "/user/home/Genome Groups/GroupName",
            "match_type": "exact"|"fuzzy",
            "source": "bvbrc-workspace"
        }

    Or an error / ambiguous-candidates dict from resolve_group_path.
    """
    # Resolve name → path
    resolved = await resolve_group_path(api, name, group_type, token)

    # Propagate errors and ambiguous results up
    if "error" in resolved or resolved.get("match_type") == "ambiguous":
        return resolved

    group_path = resolved["path"]
    group_name = resolved["name"]
    match_type = resolved["match_type"]

    # Fetch the group data
    obj = await _workspace_get(api, group_path, token, metadata_only=False)
    if "error" in obj:
        return {
            "error": f"Group '{group_name}' was found but could not be read: {obj['error']}",
            "errorType": obj.get("errorType", "API_ERROR"),
            "source": "bvbrc-workspace",
        }

    # Parse the ID list
    try:
        data = json.loads(obj.get("data", "{}")) if isinstance(obj.get("data"), str) else obj.get("data", {})
    except (json.JSONDecodeError, TypeError):
        return {
            "error": f"Group '{group_name}' has invalid data format",
            "errorType": "INVALID_RESPONSE",
            "source": "bvbrc-workspace",
        }

    if not data or "id_list" not in data:
        return {
            "error": f"Group '{group_name}' has no ID list",
            "errorType": "INVALID_RESPONSE",
            "source": "bvbrc-workspace",
        }

    id_field = GROUP_TYPE_CONFIG[group_type]["id_field"]
    ids = data["id_list"].get(id_field, [])
    if not isinstance(ids, list):
        ids = [str(ids)]

    result_key = f"{id_field}s"  # "genome_ids" or "feature_ids"
    return {
        result_key: ids,
        "count": len(ids),
        "name": group_name,
        "path": group_path,
        "match_type": match_type,
        "source": "bvbrc-workspace",
    }


# ---------------------------------------------------------------------------
# Public API: create_group
# ---------------------------------------------------------------------------


async def create_group(
    api: JsonRpcCaller,
    name: str,
    id_list: List[str],
    group_type: str,
    token: str,
) -> Dict[str, Any]:
    """
    Create a genome or feature group in the default folder.

    Parameters
    ----------
    name : str
        Group name.
    id_list : list of str
        Genome or feature IDs to store.
    group_type : str
        ``"genome_group"`` or ``"feature_group"``.
    token : str
        BV-BRC auth token.

    Returns
    -------
    ::

        {
            "name": "GroupName",
            "path": "/user/home/Genome Groups/GroupName",
            "count": N,
            "message": "Created genome group 'GroupName' with N genome(s).",
            "source": "bvbrc-workspace"
        }
    """
    if not name or not name.strip():
        return {
            "error": f"{GROUP_TYPE_CONFIG[group_type]['display_name']} name is required",
            "errorType": "INVALID_PARAMETERS",
            "source": "bvbrc-workspace",
        }

    if not id_list:
        return {
            "error": f"At least one {GROUP_TYPE_CONFIG[group_type]['id_field']} is required",
            "errorType": "INVALID_PARAMETERS",
            "source": "bvbrc-workspace",
        }

    user_id = _get_user_id_from_token(token)
    if not user_id:
        return {"error": "Could not extract user ID from token", "errorType": "AUTHENTICATION_ERROR", "source": "bvbrc-workspace"}

    name = name.strip()
    id_field = GROUP_TYPE_CONFIG[group_type]["id_field"]
    display = GROUP_TYPE_CONFIG[group_type]["display_name"]
    group_path = _default_group_path(user_id, group_type, name)

    content = {
        "id_list": {id_field: id_list},
        "name": name,
    }

    try:
        print(f"Creating {display}: {name}, path: {group_path}, ids: {len(id_list)}", file=sys.stderr)

        result = await api.acall(
            "Workspace.create",
            [{"objects": [[group_path, group_type, {}, content]]}],
            1,
            token,
        )

        return {
            "name": name,
            "path": group_path,
            "count": len(id_list),
            "data": result,
            "message": f"Created {display} '{name}' with {len(id_list)} {id_field}(s).",
            "source": "bvbrc-workspace",
        }

    except Exception as e:
        return {
            "error": f"Error creating {display}: {e}",
            "errorType": "API_ERROR",
            "source": "bvbrc-workspace",
        }
