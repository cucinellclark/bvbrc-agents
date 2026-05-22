"""
Tool wrappers for BV-BRC workspace browsing and metadata operations.

Thin async wrappers that translate agent tool-call arguments into the
MCP server's workspace_functions module.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from workspace_agent.models import AgentConfig
from workspace_agent.tools._mcp_imports import get_workspace_functions, get_json_rpc


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_api_instance = None


def _get_api(config: AgentConfig | None = None) -> Any:
    """Create or reuse a JsonRpcCaller instance for workspace API calls."""
    global _api_instance
    if _api_instance is None:
        cfg = config or AgentConfig()
        json_rpc_mod = get_json_rpc()
        _api_instance = json_rpc_mod.JsonRpcCaller(
            service_url=cfg.bvbrc_workspace_url,
            timeout=cfg.tool_timeout_seconds,
        )
    return _api_instance


def _extract_token(headers: Optional[Dict[str, str]]) -> Optional[str]:
    """Extract auth token from headers dict."""
    if headers and "Authorization" in headers:
        return headers["Authorization"]
    return None


def _extract_user_id(headers: Optional[Dict[str, str]]) -> Optional[str]:
    """Extract user_id from auth token."""
    token = _extract_token(headers)
    if token:
        try:
            for part in token.split("|"):
                if part.startswith("un="):
                    return part[3:]
        except Exception:
            pass
    return None


def _resolve_path(path: Optional[str], user_id: Optional[str]) -> str:
    """Resolve a relative path to absolute workspace path."""
    if not user_id:
        return path or "/"

    home = f"/{user_id}/home"

    if not path or path.strip() == "":
        return home

    path = path.strip()

    # Already absolute with user_id
    if path.startswith(f"/{user_id}/"):
        return path

    # Other absolute path (another user or /public)
    if path.startswith("/"):
        return path

    # "home" by itself
    if path == "home":
        return home

    # Relative -- resolve from home
    return f"{home}/{path}"


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def workspace_browse(
    path: Optional[str] = None,
    name_contains: Optional[List[str]] = None,
    file_extensions: Optional[List[str]] = None,
    workspace_types: Optional[List[str]] = None,
    sort_by: Optional[str] = None,
    sort_order: Optional[str] = None,
    num_results: int = 50,
    config: Optional[AgentConfig] = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Browse and search files in the user's workspace.

    Uses workspace_browse from workspace_functions which handles both
    directory listing (no filters) and recursive search (with filters).

    Returns the full response including items and ui_grid payload.
    """
    cfg = config or AgentConfig()
    ws_fn = get_workspace_functions()
    api = _get_api(cfg)
    token = _extract_token(headers)
    user_id = _extract_user_id(headers)

    if not token:
        return {
            "error": "Authentication required for workspace operations.",
            "errorType": "AUTHENTICATION_ERROR",
            "source": "bvbrc-workspace",
        }

    # Clamp num_results: at least 1, at most 500 (prevent unbounded queries)
    if not num_results or num_results < 1:
        num_results = 50
    num_results = min(num_results, 500)

    resolved_path = _resolve_path(path, user_id)

    try:
        result = await ws_fn.workspace_browse(
            api=api,
            token=token,
            path=resolved_path,
            filename_search_terms=name_contains,
            file_extension=file_extensions,
            file_types=workspace_types,
            sort_by=sort_by,
            sort_order=sort_order,
            num_results=num_results,
            tool_name="workspace_browse",
        )
        return result

    except Exception as e:
        return {
            "error": f"Workspace browse failed: {type(e).__name__}: {str(e)}",
            "errorType": "API_ERROR",
            "path": resolved_path,
            "source": "bvbrc-workspace",
        }


async def get_file_metadata(
    path: str,
    config: Optional[AgentConfig] = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Get detailed metadata for a single file or folder.

    Returns name, type, size, creation time, owner, permissions, etc.
    """
    cfg = config or AgentConfig()
    ws_fn = get_workspace_functions()
    api = _get_api(cfg)
    token = _extract_token(headers)
    user_id = _extract_user_id(headers)

    if not token:
        return {
            "error": "Authentication required for workspace operations.",
            "errorType": "AUTHENTICATION_ERROR",
            "source": "bvbrc-workspace",
        }

    resolved_path = _resolve_path(path, user_id)

    try:
        # Use workspace_get_object with metadata_only=True for detailed info
        result = await ws_fn.workspace_get_object(
            api=api,
            path=resolved_path,
            metadata_only=True,
            token=token,
        )
        return result

    except Exception as e:
        return {
            "error": f"File metadata retrieval failed: {type(e).__name__}: {str(e)}",
            "errorType": "API_ERROR",
            "path": resolved_path,
            "source": "bvbrc-workspace",
        }
