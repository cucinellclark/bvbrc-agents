"""
Shared group-retrieval tools for BV-BRC genome and feature groups.

Any agent can import ``get_genome_group`` / ``get_feature_group`` to
resolve named groups in the user's workspace to lists of IDs.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from shared.tools._mcp_imports import get_group_functions


def _extract_token(headers: Optional[Dict[str, str]]) -> Optional[str]:
    if headers and "Authorization" in headers:
        return headers["Authorization"]
    return None


async def get_genome_group(
    group_name: str,
    config: Any = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Retrieve genome IDs from a named genome group in the user's workspace.

    Args:
        group_name: The name of the genome group (fuzzy matched).
        config: Agent configuration object (needs ``bvbrc_workspace_url``,
                ``mcp_server_path``).
        headers: HTTP headers with auth token.

    Returns:
        Dict with ``group_name``, ``ids``, ``count``, ``id_field``; or ``error``.
    """
    gf = get_group_functions(getattr(config, "mcp_server_path", None))
    token = _extract_token(headers)
    ws_url = (
        getattr(config, "bvbrc_workspace_url", None)
        or "https://p3.theseed.org/services/Workspace"
    )

    if not token:
        return {"error": "Authentication required. No auth token provided."}

    try:
        result = await gf.get_group_ids(
            api=ws_url,
            name=group_name,
            group_type="genome_group",
            token=token,
        )

        if "error" in result:
            return result

        ids = result.get("ids", result.get("id_list", []))
        return {
            "group_name": group_name,
            "ids": ids,
            "count": len(ids),
            "id_field": "genome_id",
        }

    except Exception as e:
        return {
            "error": f"Genome group retrieval failed: {type(e).__name__}: {str(e)}",
            "group_name": group_name,
        }


async def get_feature_group(
    group_name: str,
    config: Any = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Retrieve feature IDs from a named feature group in the user's workspace.

    Args:
        group_name: The name of the feature group (fuzzy matched).
        config: Agent configuration object.
        headers: HTTP headers with auth token.

    Returns:
        Dict with ``group_name``, ``ids``, ``count``, ``id_field``; or ``error``.
    """
    gf = get_group_functions(getattr(config, "mcp_server_path", None))
    token = _extract_token(headers)
    ws_url = (
        getattr(config, "bvbrc_workspace_url", None)
        or "https://p3.theseed.org/services/Workspace"
    )

    if not token:
        return {"error": "Authentication required. No auth token provided."}

    try:
        result = await gf.get_group_ids(
            api=ws_url,
            name=group_name,
            group_type="feature_group",
            token=token,
        )

        if "error" in result:
            return result

        ids = result.get("ids", result.get("id_list", []))
        return {
            "group_name": group_name,
            "ids": ids,
            "count": len(ids),
            "id_field": "feature_id",
        }

    except Exception as e:
        return {
            "error": f"Feature group retrieval failed: {type(e).__name__}: {str(e)}",
            "group_name": group_name,
        }
