"""Pre-routing document extraction and workspace persistence.

Called from ``orchestrate()`` **before** ``route()`` so that:
  - The router sees document names (not content) via the ``has_documents`` flag
  - Agents receive excerpts in ``context_data["parsed_documents"]``
  - PDFs are extracted and written to ``parsed_pdfs/<stem>.txt``
  - Text uploads are written verbatim to ``uploaded_files/<original_name>``
  - Both are available for later ``read_file_preview`` or GoWe input paths.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Ensure the bvbrc-agents repo root is on sys.path so that
# ``shared.tools.pdf`` can be imported.  The orchestrator's own
# import chain runs before ``agent_dispatch.py`` adds the repo root.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from shared.tools.pdf import (  # noqa: E402
    decode_workspace_download,
    extract_text_from_pdf,
    persist_extracted_text,
    persist_session_file,
)

# Maximum document size (decoded bytes)
MAX_DOC_BYTES = 10 * 1024 * 1024  # 10 MB

# Maximum excerpt size forwarded to the agent prompt
EXCERPT_BYTES = 25 * 1024  # 25 KB


async def prepare_attached_documents(request: Any) -> None:
    """Decode, extract, persist, and populate ``request.parsed_documents``.

    Mutates *request* in place.  Never raises — individual document
    failures are captured as ``{name, error}`` entries.

    Handles three sources:
      1. ``request.pdfs`` — browser-uploaded PDFs (base64 data URIs)
      2. ``request.attached_files`` — browser-uploaded text files (raw text)
      3. ``request.selected_items`` with ``.pdf`` paths — workspace items
    """
    parsed: list[dict[str, Any]] = []

    # Track names already used (for dedup)
    used_names: set[str] = set()

    # --- 1. Uploaded PDFs ---
    for pdf_entry in request.pdfs or []:
        name = pdf_entry.get("name", "document.pdf") if isinstance(pdf_entry, dict) else "document.pdf"
        try:
            content = pdf_entry.get("content", "") if isinstance(pdf_entry, dict) else str(pdf_entry)
            pdf_bytes = _decode_upload(content)

            if len(pdf_bytes) > MAX_DOC_BYTES:
                parsed.append({"name": name, "error": f"PDF exceeds the {MAX_DOC_BYTES // (1024*1024)} MB limit ({len(pdf_bytes)} bytes)."})
                continue

            if not pdf_bytes[:5] == b"%PDF-":
                parsed.append({"name": name, "error": "File does not appear to be a valid PDF."})
                continue

            text, page_count = await asyncio.to_thread(extract_text_from_pdf, pdf_bytes)

            if not text.strip():
                parsed.append({
                    "name": name,
                    "error": "This PDF has no selectable text (likely a scanned document). OCR is not supported.",
                    "page_count": page_count,
                })
                continue

            # Persist .txt to session workspace.  ``used_names`` holds the
            # full filenames persist_session_file actually writes, so the
            # reservation is "<stem>.txt", not "<stem>" — otherwise dedup
            # never fires and a second same-named PDF overwrites the first.
            stem = _stem_from_name(name)
            txt_path = await _try_persist(
                text, stem, request, used_names,
            )
            used_names.add(f"{stem}.txt")

            parsed.append({
                "name": name,
                "workspace_path": txt_path,
                "workspace_txt_path": txt_path,  # backward compat — drop next revision
                "page_count": page_count,
                "char_count": len(text),
                "excerpt": _excerpt(text),
                "source": "pdf",
            })

        except Exception as e:
            logger.warning("PDF extraction failed for %s: %s", name, e, exc_info=True)
            parsed.append({"name": name, "error": f"PDF extraction failed: {e}"})

    # --- 2. Uploaded text files ---
    for file_entry in getattr(request, "attached_files", None) or []:
        name = file_entry.get("name", "unnamed_file") if isinstance(file_entry, dict) else "unnamed_file"
        try:
            content = file_entry.get("content", "") if isinstance(file_entry, dict) else ""

            # Reject binary (null bytes)
            if "\x00" in content:
                parsed.append({
                    "name": name,
                    "error": "File appears to be a binary file; only text uploads are supported.",
                    "source": "upload",
                })
                continue

            # Check UTF-8 byte size
            byte_len = len(content.encode("utf-8"))
            if byte_len > MAX_DOC_BYTES:
                parsed.append({
                    "name": name,
                    "error": f"File exceeds the {MAX_DOC_BYTES // (1024*1024)} MB limit ({byte_len} bytes).",
                    "source": "upload",
                })
                continue

            # Persist original file to uploaded_files/.  persist_session_file
            # reserves the name it writes in ``used_names``; reserve the
            # original too in case the write was skipped entirely.
            ws_path = await _try_persist_file(
                content, name, request, used_names,
            )
            used_names.add(name)

            mime_type = file_entry.get("mime_type", "text/plain") if isinstance(file_entry, dict) else "text/plain"
            entry: dict[str, Any] = {
                "name": name,
                "workspace_path": ws_path,
                "char_count": len(content),
                "excerpt": _excerpt(content),
                "source": "upload",
                "mime_type": mime_type,
            }
            # ``name`` stays the filename the user attached (it matches the
            # chip in the UI).  If another attachment in this message had
            # the same name, the workspace object was written under a
            # different basename — say so rather than letting the header
            # and the path disagree silently.
            saved_name = os.path.basename(ws_path) if ws_path else name
            if saved_name != name:
                entry["saved_as"] = saved_name
            parsed.append(entry)

        except Exception as e:
            logger.warning("Text file processing failed for %s: %s", name, e, exc_info=True)
            parsed.append({"name": name, "error": f"Text file processing failed: {e}", "source": "upload"})

    # --- 3. Workspace PDF items from selected_items ---
    if request.selected_items:
        for item in request.selected_items:
            path = item.get("path", "") if isinstance(item, dict) else ""
            if not path.lower().endswith(".pdf"):
                continue

            name = os.path.basename(path)
            try:
                result = await _workspace_pdf(path, name, request, used_names)
                parsed.append(result)

                # If successful, rewrite the item's path to point at the .txt
                ws_path = result.get("workspace_path") or result.get("workspace_txt_path")
                if ws_path:
                    item["path"] = ws_path
                    used_names.add(f"{_stem_from_name(name)}.txt")

            except Exception as e:
                logger.warning("Workspace PDF extraction failed for %s: %s", path, e, exc_info=True)
                parsed.append({"name": name, "error": f"Workspace PDF extraction failed: {e}"})

    request.parsed_documents = parsed


async def _workspace_pdf(
    path: str, name: str, request: Any, used_stems: set[str]
) -> dict[str, Any]:
    """Download, size-check, extract, and persist a workspace PDF."""
    from shared.tools._mcp_imports import get_workspace_functions, get_json_rpc

    # Build API client
    mcp_path = None
    json_rpc_mod = get_json_rpc(mcp_path)
    ws_fn = get_workspace_functions(mcp_path)
    ws_url = "https://p3.theseed.org/services/Workspace"
    api = json_rpc_mod.JsonRpcCaller(service_url=ws_url, timeout=30)
    token = request.auth_token or ""

    # Size check first
    meta_result = await ws_fn.workspace_get_object(
        api=api, path=path, metadata_only=True, token=token,
    )
    if isinstance(meta_result, dict) and meta_result.get("error"):
        return {"name": name, "error": f"Cannot access workspace PDF: {meta_result['error']}"}

    total_size = 0
    if isinstance(meta_result, dict):
        total_size = meta_result.get("total_size", 0) or meta_result.get("size", 0) or 0
    if total_size > MAX_DOC_BYTES:
        return {"name": name, "error": f"PDF exceeds the {MAX_DOC_BYTES // (1024*1024)} MB limit ({total_size} bytes)."}

    # Download
    dl_result = await ws_fn.workspace_download_file(
        api=api, path=path, token=token, return_data=True,
    )
    if isinstance(dl_result, dict) and dl_result.get("error"):
        return {"name": name, "error": f"Download failed: {dl_result['error']}"}

    pdf_bytes = decode_workspace_download(dl_result)

    if len(pdf_bytes) > MAX_DOC_BYTES:
        return {"name": name, "error": f"PDF exceeds the {MAX_DOC_BYTES // (1024*1024)} MB limit ({len(pdf_bytes)} bytes)."}

    text, page_count = await asyncio.to_thread(extract_text_from_pdf, pdf_bytes)

    if not text.strip():
        return {
            "name": name,
            "error": "This PDF has no selectable text (likely a scanned document). OCR is not supported.",
            "page_count": page_count,
        }

    stem = _stem_from_name(name)
    txt_path = await _try_persist(text, stem, request, used_stems)

    return {
        "name": name,
        "workspace_path": txt_path,
        "workspace_txt_path": txt_path,  # backward compat — drop next revision
        "page_count": page_count,
        "char_count": len(text),
        "excerpt": _excerpt(text),
        "source": "pdf",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode_upload(content: str) -> bytes:
    """Strip optional data URI prefix and base64-decode."""
    if content.startswith("data:"):
        # Strip "data:application/pdf;base64,"
        _, _, after = content.partition(",")
        return base64.b64decode(after)
    return base64.b64decode(content)


def _excerpt(text: str) -> str:
    """First ``EXCERPT_BYTES`` of *text*, cut on a UTF-8 character boundary.

    Slicing by characters would let a multi-byte document ship up to 4x
    the intended prompt budget.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= EXCERPT_BYTES:
        return text
    return encoded[:EXCERPT_BYTES].decode("utf-8", errors="ignore")


def _stem_from_name(name: str) -> str:
    """Extract filename stem (no extension)."""
    base = os.path.basename(name)
    stem, _ = os.path.splitext(base)
    return stem or "document"


async def _try_persist_file(
    content: str,
    filename: str,
    request: Any,
    used_names: set[str],
) -> str | None:
    """Persist an uploaded text file to the session workspace. Returns path or None."""
    session_id = getattr(request, "session_id", None)
    workspace_path = getattr(request, "workspace_path", None)
    auth_token = getattr(request, "auth_token", None)

    if not session_id or not workspace_path or not auth_token:
        logger.warning(
            "persist_session_file: missing session context "
            "(session_id=%s, workspace_path=%s), skipping write",
            session_id,
            bool(workspace_path),
        )
        return None

    class _MinConfig:
        mcp_server_path = None
        bvbrc_workspace_url = "https://p3.theseed.org/services/Workspace"
        tool_timeout_seconds = 30

    try:
        return await persist_session_file(
            content=content,
            filename=filename,
            subfolder="uploaded_files",
            session_id=session_id,
            workspace_path=workspace_path,
            token=auth_token,
            config=_MinConfig(),
            existing_names=used_names,
        )
    except Exception as e:
        logger.warning("persist_session_file failed for %s: %s", filename, e)
        return None


async def _try_persist(
    text: str,
    stem: str,
    request: Any,
    used_names: set[str],
) -> str | None:
    """Persist extracted PDF text to the session workspace. Returns path or None."""
    session_id = getattr(request, "session_id", None)
    workspace_path = getattr(request, "workspace_path", None)
    auth_token = getattr(request, "auth_token", None)

    if not session_id or not workspace_path or not auth_token:
        logger.warning(
            "persist_extracted_text: missing session context "
            "(session_id=%s, workspace_path=%s), skipping write",
            session_id,
            bool(workspace_path),
        )
        return None

    # Build a minimal config-like object for persist_extracted_text
    class _MinConfig:
        mcp_server_path = None
        bvbrc_workspace_url = "https://p3.theseed.org/services/Workspace"
        tool_timeout_seconds = 30

    try:
        return await persist_extracted_text(
            text=text,
            stem=stem,
            session_id=session_id,
            workspace_path=workspace_path,
            token=auth_token,
            config=_MinConfig(),
            existing_names=used_names,
        )
    except Exception as e:
        logger.warning("persist_extracted_text failed: %s", e)
        return None
