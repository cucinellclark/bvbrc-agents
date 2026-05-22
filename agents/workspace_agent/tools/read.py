"""
Tool wrapper for reading/previewing workspace file contents.

Thin async wrapper around workspace_read_range for byte-range reads.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from workspace_agent.models import AgentConfig
from workspace_agent.tools._mcp_imports import get_workspace_functions, get_json_rpc
from workspace_agent.tools.browse import _get_api, _extract_token, _extract_user_id, _resolve_path


async def read_file_preview(
    path: str,
    max_bytes: int = 8192,
    config: Optional[AgentConfig] = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Read the first portion of a workspace file.

    Uses workspace_read_range with start_byte=0 to preview file contents.
    Returns text data for text files, base64-encoded data for binary files.
    Includes total_size and is_complete for paging awareness.

    Args:
        path: Workspace path to the file.
        max_bytes: Maximum bytes to read (default 8192, max 1048576).
        config: Agent configuration.
        headers: HTTP headers with auth token.

    Returns:
        Dict with data, bytes_read, total_size, is_complete, etc.
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

        # Add workspace_path to result for reference
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
