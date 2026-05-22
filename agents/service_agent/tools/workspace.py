"""
Tool wrappers for BV-BRC workspace operations.

Thin async wrappers that translate agent tool-call arguments into the
MCP server's workspace_functions module.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional
from types import ModuleType

from service_agent.models import AgentConfig


# ---------------------------------------------------------------------------
# MCP server import helper
# ---------------------------------------------------------------------------

_workspace_functions: Optional[ModuleType] = None
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


def _get_workspace_functions(config: AgentConfig | None = None) -> ModuleType:
    global _workspace_functions
    if _workspace_functions is None:
        _ensure_path(config)
        from functions import workspace_functions
        _workspace_functions = workspace_functions
    return _workspace_functions


def _extract_token(headers: Optional[Dict[str, str]]) -> Optional[str]:
    """Extract auth token from headers dict."""
    if headers and "Authorization" in headers:
        return headers["Authorization"]
    return None


def _extract_user_id(headers: Optional[Dict[str, str]]) -> str:
    """Extract user_id from auth token."""
    token = _extract_token(headers)
    if token:
        try:
            for part in token.split("|"):
                if part.startswith("un="):
                    return part[3:]
        except Exception:
            pass
    return "anonymous"


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def workspace_browse(
    path: Optional[str] = None,
    type_filter: Optional[str] = None,
    search: Optional[str] = None,
    config: Optional[AgentConfig] = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Browse files in the user's BV-BRC workspace.

    Args:
        path: Workspace path to browse (defaults to user home).
        type_filter: Filter by file type (e.g., 'reads', 'contigs').
        search: Search term to filter files by name.
        config: Agent configuration.
        headers: HTTP headers with auth token.

    Returns:
        Dict with workspace listing results.
    """
    cfg = config or AgentConfig()
    ws_fn = _get_workspace_functions(config)
    token = _extract_token(headers)
    user_id = _extract_user_id(headers)

    if not token:
        return {"error": "Authentication required for workspace operations. No auth token provided."}

    # Default path to user's home directory
    if not path:
        path = f"/{user_id}/home"

    try:
        file_types = None
        if type_filter:
            file_types = [type_filter]

        filename_search_terms = None
        if search:
            filename_search_terms = [search]

        result = await ws_fn.workspace_browse(
            api=cfg.bvbrc_workspace_url,
            token=token,
            path=path,
            search=search,
            filename_search_terms=filename_search_terms,
            file_types=file_types,
            num_results=50,
            tool_name="workspace_browse",
        )
        return result

    except Exception as e:
        return {
            "error": f"Workspace browse failed: {type(e).__name__}: {str(e)}",
            "path": path,
        }


async def read_file_info(
    path: str,
    config: Optional[AgentConfig] = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Get metadata about a workspace file.

    Args:
        path: Full workspace path to the file.
        config: Agent configuration.
        headers: HTTP headers with auth token.

    Returns:
        Dict with file metadata (name, type, size, etc.).
    """
    cfg = config or AgentConfig()
    ws_fn = _get_workspace_functions(config)
    token = _extract_token(headers)

    if not token:
        return {"error": "Authentication required. No auth token provided."}

    try:
        result = await ws_fn.workspace_get_file_metadata(
            api=cfg.bvbrc_workspace_url,
            path=path,
            token=token,
        )
        return result

    except Exception as e:
        return {
            "error": f"File metadata retrieval failed: {type(e).__name__}: {str(e)}",
            "path": path,
        }
