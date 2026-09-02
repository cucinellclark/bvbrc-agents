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

    # Treat "." and "./" as home — these are invalid workspace paths
    if path in (".", "./"):
        return home

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


def _is_object_not_found(value: Any) -> bool:
    """Check whether an error string or result indicates an object-not-found."""
    text = str(value).lower()
    return "object not found" in text or "_error_object not found" in text


def _path_not_found_result(resolved_path: str) -> Dict[str, Any]:
    """Return a structured PATH_NOT_FOUND error with actionable hints."""
    return {
        "error": (
            f"The path '{resolved_path}' was not found in the workspace. "
            "This means the folder or file does not exist at that location "
            "(it is NOT a workspace service outage)."
        ),
        "errorType": "PATH_NOT_FOUND",
        "path": resolved_path,
        "hint": (
            "Try browsing the home directory (empty path) or searching "
            "with name_contains to find the correct path."
        ),
        "source": "bvbrc-workspace",
    }


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

        # Detect object-not-found buried inside the result dict
        if isinstance(result, dict):
            err_val = result.get("error") or result.get("data")
            if err_val and _is_object_not_found(err_val):
                return _path_not_found_result(resolved_path)

        # Attach a workspace browser URL so the LLM can include a
        # clickable markdown link in its response.
        if isinstance(result, dict) and not result.get("error"):
            from shared.tools.url_utils import build_workspace_url

            ws_url = build_workspace_url(resolved_path)
            if ws_url:
                result["workspace_browser_url"] = ws_url

        return result

    except Exception as e:
        if _is_object_not_found(e):
            return _path_not_found_result(resolved_path)
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

        # Detect object-not-found buried inside the result dict
        if isinstance(result, dict):
            err_val = result.get("error") or result.get("data")
            if err_val and _is_object_not_found(err_val):
                return _path_not_found_result(resolved_path)

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
        if _is_object_not_found(e):
            return _path_not_found_result(resolved_path)
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

    # --- PDF branch ---
    if resolved_path.lower().endswith(".pdf"):
        return await _read_pdf_preview(
            resolved_path, max_bytes, config, headers, ws_fn, api, token,
        )

    try:
        result = await ws_fn.workspace_read_range(
            api=api,
            path=resolved_path,
            token=token,
            start_byte=0,
            max_bytes=max_bytes,
        )

        # Detect object-not-found buried inside the result dict
        if isinstance(result, dict):
            err_val = result.get("error") or result.get("data")
            if err_val and _is_object_not_found(err_val):
                return _path_not_found_result(resolved_path)

        if isinstance(result, dict) and not result.get("error"):
            result["workspace_path"] = resolved_path
            result["source_type"] = "workspace"

        return result

    except Exception as e:
        if _is_object_not_found(e):
            return _path_not_found_result(resolved_path)
        return {
            "error": f"File read failed: {type(e).__name__}: {str(e)}",
            "errorType": "API_ERROR",
            "path": resolved_path,
            "source": "bvbrc-workspace",
        }


async def _read_pdf_preview(
    resolved_path: str,
    max_bytes: int,
    config: Any,
    headers: Optional[Dict[str, str]],
    ws_fn: Any,
    api: Any,
    token: str,
) -> Dict[str, Any]:
    """Handle ``read_file_preview`` for ``.pdf`` files.

    Downloads the full PDF (after a size check), extracts text via PyMuPDF,
    optionally persists a ``.txt`` to the session workspace, and returns
    the extracted text honoring *max_bytes*.
    """
    import asyncio

    from shared.tools.pdf import (
        decode_workspace_download,
        extract_text_from_pdf,
        persist_extracted_text,
    )

    MAX_PDF_BYTES = 10 * 1024 * 1024  # 10 MB

    try:
        # 1. Size check via metadata FIRST (before downloading)
        meta_result = await ws_fn.workspace_get_object(
            api=api, path=resolved_path, metadata_only=True, token=token,
        )
        if isinstance(meta_result, dict):
            err_val = meta_result.get("error") or meta_result.get("data")
            if err_val and _is_object_not_found(err_val):
                return _path_not_found_result(resolved_path)
            total_size = meta_result.get("total_size", 0) or meta_result.get("size", 0) or 0
            if total_size > MAX_PDF_BYTES:
                return {
                    "error": f"PDF too large for extraction ({total_size} bytes, limit {MAX_PDF_BYTES})",
                    "total_size": total_size,
                    "workspace_path": resolved_path,
                }

        # 2. Download the full file
        dl_result = await ws_fn.workspace_download_file(
            api=api, path=resolved_path, token=token, return_data=True,
        )
        if isinstance(dl_result, dict) and dl_result.get("error"):
            if _is_object_not_found(dl_result.get("error", "")):
                return _path_not_found_result(resolved_path)
            return {
                "error": f"PDF download failed: {dl_result['error']}",
                "workspace_path": resolved_path,
            }

        pdf_bytes = decode_workspace_download(dl_result)
        if len(pdf_bytes) > MAX_PDF_BYTES:
            return {
                "error": f"PDF too large for extraction ({len(pdf_bytes)} bytes, limit {MAX_PDF_BYTES})",
                "workspace_path": resolved_path,
            }

        # 3. Extract text (CPU-bound — offload to thread)
        text, page_count = await asyncio.to_thread(extract_text_from_pdf, pdf_bytes)

        if not text.strip():
            return {
                "error": "PDF has no selectable text (likely scanned). OCR is not supported.",
                "workspace_path": resolved_path,
                "page_count": page_count,
            }

        # 4. Persist .txt to session workspace (best-effort)
        import os

        persist_path: str | None = None
        session_id = getattr(config, "session_id", None)
        workspace_path = getattr(config, "workspace_path", None)
        if session_id and workspace_path and token:
            stem = os.path.splitext(os.path.basename(resolved_path))[0] or "document"
            try:
                persist_path = await persist_extracted_text(
                    text=text,
                    stem=stem,
                    session_id=session_id,
                    workspace_path=workspace_path,
                    token=token,
                    config=config,
                )
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(
                    "read_file_preview: PDF persist failed for %s: %s",
                    resolved_path,
                    e,
                )
        else:
            import logging

            logging.getLogger(__name__).warning(
                "read_file_preview: skipping PDF persist (no session context)"
            )

        # 5. Return extracted text, honoring max_bytes
        extracted_len = len(text)
        return {
            "data": text[:max_bytes],
            "start_byte": 0,
            "bytes_read": min(extracted_len, max_bytes),
            "total_size": extracted_len,
            "is_complete": extracted_len <= max_bytes,
            "workspace_path": resolved_path,
            "source_type": "pdf_extraction",
            "page_count": page_count,
            "parsed_txt_path": persist_path,
        }

    except Exception as e:
        if _is_object_not_found(e):
            return _path_not_found_result(resolved_path)
        return {
            "error": f"PDF extraction failed: {type(e).__name__}: {str(e)}",
            "errorType": "API_ERROR",
            "path": resolved_path,
            "source": "bvbrc-workspace",
        }
