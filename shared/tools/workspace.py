"""
Shared workspace tools for browsing, inspecting, and previewing BV-BRC
workspace files.

Any agent can import ``workspace_browse``, ``get_file_metadata``, and
``read_file_preview`` to interact with the user's workspace.

These tools accept a generic *config* object (any object with
``bvbrc_workspace_url``, ``tool_timeout_seconds``, and ``mcp_server_path``
attributes) and a *headers* dict containing the ``Authorization`` token.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from shared.tools._mcp_imports import get_workspace_functions, get_json_rpc


# ---------------------------------------------------------------------------
# Shared helpers
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


def _extract_token(headers: Optional[Dict[str, str]]) -> Optional[str]:
    """Extract auth token from headers dict."""
    if headers and "Authorization" in headers:
        return headers["Authorization"]
    return None


def _extract_user_id(headers: Optional[Dict[str, str]]) -> Optional[str]:
    """Extract ``user_id`` from the BV-BRC auth token."""
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
    """Resolve a relative path to an absolute workspace path."""
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

    # Relative — resolve from home
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
    config: Any = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Browse and search files in the user's workspace.

    Uses ``workspace_functions.workspace_browse`` which handles both
    directory listing (no filters) and recursive search (with filters).

    Returns the full response including items and ``ui_grid`` payload.
    """
    ws_fn = get_workspace_functions(getattr(config, "mcp_server_path", None))
    api = _get_api(config)
    token = _extract_token(headers)
    user_id = _extract_user_id(headers)

    if not token:
        return {
            "error": "Authentication required for workspace operations.",
            "errorType": "AUTHENTICATION_ERROR",
            "source": "bvbrc-workspace",
        }

    # Clamp num_results: at least 1, at most 500
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

        # Attach a workspace browser URL so the LLM can include a
        # clickable markdown link in its response.
        if isinstance(result, dict) and not result.get("error"):
            from shared.tools.url_utils import build_workspace_url

            ws_url = build_workspace_url(resolved_path)
            if ws_url:
                result["workspace_browser_url"] = ws_url

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
    config: Any = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Get detailed metadata for a single file or folder.

    Returns name, type, size, creation time, owner, permissions, etc.
    """
    ws_fn = get_workspace_functions(getattr(config, "mcp_server_path", None))
    api = _get_api(config)
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
        result = await ws_fn.workspace_get_object(
            api=api,
            path=resolved_path,
            metadata_only=True,
            token=token,
        )

        # Attach a workspace browser URL for the file's parent directory.
        if isinstance(result, dict) and not result.get("error"):
            from shared.tools.url_utils import build_workspace_url

            # Link to the file's parent folder so the user can see it
            # in the workspace browser.
            parent_path = "/".join(resolved_path.rstrip("/").split("/")[:-1])
            ws_url = build_workspace_url(parent_path or resolved_path)
            if ws_url:
                result["workspace_browser_url"] = ws_url

        return result

    except Exception as e:
        return {
            "error": f"File metadata retrieval failed: {type(e).__name__}: {str(e)}",
            "errorType": "API_ERROR",
            "path": resolved_path,
            "source": "bvbrc-workspace",
        }


async def read_file_preview(
    path: str,
    max_bytes: int = 8192,
    config: Any = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Read the first portion of a workspace file.

    Returns text data for text files, base64-encoded data for binary files.
    Includes ``total_size`` and ``is_complete`` for paging awareness.

    Args:
        path: Workspace path to the file.
        max_bytes: Maximum bytes to read (default 8192, max 1 048 576).
        config: Agent configuration object.
        headers: HTTP headers with auth token.
    """
    ws_fn = get_workspace_functions(getattr(config, "mcp_server_path", None))
    api = _get_api(config)
    token = _extract_token(headers)
    user_id = _extract_user_id(headers)

    if not token:
        return {
            "error": "Authentication required for workspace operations.",
            "errorType": "AUTHENTICATION_ERROR",
            "source": "bvbrc-workspace",
        }

    resolved_path = _resolve_path(path, user_id)

    # Clamp max_bytes to 1 MB
    max_bytes = min(max(max_bytes, 1), 1024 * 1024)

    try:
        result = await ws_fn.workspace_read_range(
            api=api,
            path=resolved_path,
            token=token,
            start_byte=0,
            max_bytes=max_bytes,
        )

        if isinstance(result, dict) and not result.get("error"):
            result["workspace_path"] = resolved_path
            result["source_type"] = "workspace"

        return result

    except Exception as e:
        return {
            "error": f"File read failed: {type(e).__name__}: {str(e)}",
            "errorType": "API_ERROR",
            "path": resolved_path,
            "source": "bvbrc-workspace",
        }
