"""
PDF extraction and workspace persistence helpers.

Shared by the orchestrator preprocess (``documents.py``) and by
``read_file_preview`` for ad-hoc workspace PDFs.

Key functions:
  - ``extract_text_from_pdf`` — CPU-bound, call via ``asyncio.to_thread``
  - ``decode_workspace_download`` — unwrap ``workspace_download_file``
  - ``persist_session_file`` — write any file to a session workspace subfolder
  - ``persist_extracted_text`` — thin wrapper: write ``.txt`` to ``parsed_pdfs/``
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import os
import re
import tempfile
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_text_from_pdf(pdf_bytes: bytes) -> tuple[str, int]:
    """Extract embedded text from PDF bytes.

    Returns ``(text, page_count)``.  *text* is the concatenation of all
    pages separated by double newlines.

    **CPU-bound** — always call via ``asyncio.to_thread()``.

    Raises on corrupt or unreadable PDFs.  An empty *text* (with
    ``page_count > 0``) means the PDF is scanned / image-only.
    """
    import fitz  # PyMuPDF — lazy import; only installed in orchestrator_env

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pages = [page.get_text() for page in doc]
    finally:
        doc.close()
    return "\n\n".join(pages), len(pages)


# ---------------------------------------------------------------------------
# Workspace download decoding
# ---------------------------------------------------------------------------


def decode_workspace_download(data: Any) -> bytes:
    """Unwrap ``workspace_download_file(return_data=True)`` into raw bytes.

    ``workspace_download_file`` (``workspace_functions.py:762``) tries
    ``content.decode('utf-8')`` first and, on failure, wraps the base64
    output in ``<base64_encoded_data>...</base64_encoded_data>``.  This
    function handles both shapes.
    """
    if isinstance(data, bytes):
        return data

    if isinstance(data, str):
        # Check for the sentinel wrapper
        m = re.match(
            r"<base64_encoded_data>(.*)</base64_encoded_data>",
            data,
            re.DOTALL,
        )
        if m:
            return base64.b64decode(m.group(1))
        # Plain text (successful utf-8 decode of the content) — re-encode
        return data.encode("utf-8")

    # dict with "data" key (the result envelope)
    if isinstance(data, dict) and "data" in data:
        return decode_workspace_download(data["data"])

    raise ValueError(f"Cannot decode workspace download: unexpected type {type(data)}")


# ---------------------------------------------------------------------------
# Workspace persistence
# ---------------------------------------------------------------------------


def _filename_with_dedup(filename: str, existing_names: set[str]) -> str:
    """Return a name for *filename* that is free in *existing_names*.

    On collision a short hash of the original name is inserted before the
    extension (``sample.fasta`` -> ``sample_a1b2c3.fasta``).  The hash is
    deterministic, so a *third* file with the same original name would
    produce the same candidate; a counter is appended until the name is
    actually free.
    """
    if filename not in existing_names:
        return filename
    stem, ext = os.path.splitext(filename)
    h = hashlib.sha256(filename.encode()).hexdigest()[:6]
    candidate = f"{stem}_{h}{ext}"
    n = 2
    while candidate in existing_names:
        candidate = f"{stem}_{h}_{n}{ext}"
        n += 1
    return candidate


async def persist_session_file(
    content: str,
    filename: str,
    subfolder: str,
    session_id: str,
    workspace_path: str,
    token: str,
    config: Any,
    existing_names: set[str] | None = None,
) -> str | None:
    """Write *filename* under ``/chats/<session_id>/<subfolder>/``.

    Returns the full workspace path on success, or ``None`` on failure
    (logged at WARNING).

    *filename* keeps its original extension (e.g. ``Ecoli_K12.fasta``).
    For PDF extracts, callers should pass ``filename="<stem>.txt"`` and
    ``subfolder="parsed_pdfs"``.

    *existing_names*, when given, is both read and **mutated**: the name
    this call writes is added to it so subsequent calls in the same batch
    dedup against it.

    Uses ``workspace_upload()`` from MCP ``workspace_functions`` via a
    temp file, wrapped in ``asyncio.to_thread()`` for the synchronous
    upload step.  The temp file is written inside a temp directory using
    the desired basename so that ``workspace_upload`` — which derives the
    workspace object name from ``os.path.basename(filename)`` — creates
    the correctly named object.
    """
    from shared.tools._mcp_imports import get_workspace_functions, get_json_rpc

    ws_fn = get_workspace_functions(getattr(config, "mcp_server_path", None))
    json_rpc_mod = get_json_rpc(getattr(config, "mcp_server_path", None))
    ws_url = (
        getattr(config, "bvbrc_workspace_url", None)
        or "https://p3.theseed.org/services/Workspace"
    )
    timeout = getattr(config, "tool_timeout_seconds", 30)
    api = json_rpc_mod.JsonRpcCaller(service_url=ws_url, timeout=timeout)

    # Resolve deduplication and *reserve* the resolved name so a later
    # call in the same batch cannot pick it again.  Reserve even if the
    # upload below fails — the caller learns the final name from the
    # returned path (or from the reservation).
    if existing_names is not None:
        filename = _filename_with_dedup(filename, existing_names)
        existing_names.add(filename)

    # Build the destination folder and file path
    # workspace_path is like "/user@domain/home" — strip trailing slash
    base = workspace_path.rstrip("/")
    folder_path = f"{base}/chats/{session_id}/{subfolder}"
    file_path = f"{folder_path}/{filename}"

    tmp_dir: str | None = None
    try:
        # Ensure the subfolder exists
        try:
            await ws_fn._workspace_create(
                api,
                [[folder_path, "folder", {}, ""]],
                token,
                create_upload_nodes=False,
                overwrite=None,
            )
        except Exception:
            # Folder may already exist — that's fine
            pass

        # Write content to a temp file with the correct basename.
        # workspace_upload uses os.path.basename(filename) as the
        # workspace object name, so we must use the real filename.
        tmp_dir = tempfile.mkdtemp(prefix="ws_persist_")
        tmp_path = os.path.join(tmp_dir, os.path.basename(filename))
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Upload — workspace_upload is sync-heavy (_upload_file_to_url uses
        # requests), so wrap in to_thread
        result = await asyncio.to_thread(
            _sync_upload, ws_fn, api, tmp_path, folder_path, token
        )
        if isinstance(result, dict) and result.get("error"):
            logger.warning(
                "persist_session_file: upload failed for %s: %s",
                file_path,
                result["error"],
            )
            return None

        logger.info("persist_session_file: wrote %s (%d bytes)", file_path, len(content))
        return file_path

    except Exception as e:
        logger.warning("persist_session_file failed for %s: %s", file_path, e)
        return None
    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            import shutil
            try:
                shutil.rmtree(tmp_dir)
            except OSError:
                pass


async def persist_extracted_text(
    text: str,
    stem: str,
    session_id: str,
    workspace_path: str,
    token: str,
    config: Any,
    existing_names: set[str] | None = None,
) -> str | None:
    """Write ``<stem>.txt`` under ``/chats/<session_id>/parsed_pdfs/``.

    Thin wrapper around :func:`persist_session_file`.
    """
    return await persist_session_file(
        content=text,
        filename=f"{stem}.txt",
        subfolder="parsed_pdfs",
        session_id=session_id,
        workspace_path=workspace_path,
        token=token,
        config=config,
        existing_names=existing_names,
    )


def _sync_upload(ws_fn: Any, api: Any, filename: str, upload_dir: str, token: str) -> Any:
    """Synchronous wrapper around the async workspace_upload.

    ``workspace_upload`` is an async function, but its heavy I/O is in
    ``_upload_file_to_url`` (synchronous ``requests.put``).  We need to
    call it from ``asyncio.to_thread``, so we spin up a small event loop
    for the async preamble.
    """
    import asyncio as _asyncio

    loop = _asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            ws_fn.workspace_upload(api, filename, upload_dir, token)
        )
    finally:
        loop.close()
