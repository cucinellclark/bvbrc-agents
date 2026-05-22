"""
Tool wrappers for BV-BRC genome and feature group operations.

Thin async wrappers that translate agent tool-call arguments into the
MCP server's group_functions module.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional
from types import ModuleType

from service_agent.models import AgentConfig


# ---------------------------------------------------------------------------
# MCP server import helper
# ---------------------------------------------------------------------------

_group_functions: Optional[ModuleType] = None
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


def _get_group_functions(config: AgentConfig | None = None) -> ModuleType:
    global _group_functions
    if _group_functions is None:
        _ensure_path(config)
        from functions import group_functions
        _group_functions = group_functions
    return _group_functions


def _extract_token(headers: Optional[Dict[str, str]]) -> Optional[str]:
    if headers and "Authorization" in headers:
        return headers["Authorization"]
    return None


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def get_genome_group(
    group_name: str,
    config: Optional[AgentConfig] = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Retrieve genome IDs from a named genome group in the user's workspace.

    Args:
        group_name: The name of the genome group (fuzzy matched).
        config: Agent configuration.
        headers: HTTP headers with auth token.

    Returns:
        Dict with genome IDs and count, or error if not found.
    """
    cfg = config or AgentConfig()
    gf = _get_group_functions(config)
    token = _extract_token(headers)

    if not token:
        return {"error": "Authentication required. No auth token provided."}

    try:
        result = await gf.get_group_ids(
            api=cfg.bvbrc_workspace_url,
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
    config: Optional[AgentConfig] = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Retrieve feature IDs from a named feature group in the user's workspace.

    Args:
        group_name: The name of the feature group (fuzzy matched).
        config: Agent configuration.
        headers: HTTP headers with auth token.

    Returns:
        Dict with feature IDs and count, or error if not found.
    """
    cfg = config or AgentConfig()
    gf = _get_group_functions(config)
    token = _extract_token(headers)

    if not token:
        return {"error": "Authentication required. No auth token provided."}

    try:
        result = await gf.get_group_ids(
            api=cfg.bvbrc_workspace_url,
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
