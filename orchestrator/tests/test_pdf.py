"""Tests for the PDF extraction pipeline.

Covers:
  - extract_text_from_pdf (bvbrc_mcp.tools.pdf)
  - decode_workspace_download (bvbrc_mcp.tools.pdf)
  - _filename_with_dedup / persist_session_file (bvbrc_mcp.tools.pdf)
  - prepare_attached_documents text uploads (orchestrator.documents)
  - format_attached_documents (shared.agent_utils)
"""

import base64
import sys
from pathlib import Path

# Ensure the bvbrc-agents repo root is importable so that
# ``bvbrc_mcp.tools.pdf`` and ``shared.agent_utils`` resolve.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import fitz  # PyMuPDF
import pytest

from bvbrc_mcp.tools.pdf import (
    _filename_with_dedup,
    decode_workspace_download,
    extract_text_from_pdf,
    persist_session_file,
)
from shared.agent_utils import format_attached_documents


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _make_pdf_bytes(*page_texts: str) -> bytes:
    """Build a minimal in-memory PDF with one page per *page_texts* entry.

    Each string is inserted as text on a new page.  If *page_texts* is
    empty, a single blank page (no text layer) is created.
    """
    doc = fitz.open()
    if not page_texts:
        # One blank page — simulates a scanned/image-only PDF
        doc.new_page()
    else:
        for text in page_texts:
            page = doc.new_page()
            page.insert_text((72, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# -----------------------------------------------------------------------
# extract_text_from_pdf
# -----------------------------------------------------------------------








# -----------------------------------------------------------------------
# decode_workspace_download
# -----------------------------------------------------------------------














# -----------------------------------------------------------------------
# _filename_with_dedup
# -----------------------------------------------------------------------














# -----------------------------------------------------------------------
# persist_session_file
# -----------------------------------------------------------------------


class _FakeConfig:
    bvbrc_workspace_url = "https://example.invalid/Workspace"
    tool_timeout_seconds = 5


def _install_fake_workspace(monkeypatch, uploads):
    """Stub the MCP workspace layer; record (local_path, upload_dir)."""
    import bvbrc_mcp.tools._clients as mcp_imports
    import bvbrc_mcp.tools.pdf as pdf_mod

    class _FakeWsFn:
        @staticmethod
        async def _workspace_create(api, objects, token, **kwargs):
            return {"ok": True}

    class _FakeJsonRpc:
        class JsonRpcCaller:
            def __init__(self, service_url=None, timeout=None):
                pass

    monkeypatch.setattr(
        mcp_imports, "get_workspace_functions", lambda path: _FakeWsFn, raising=False
    )
    monkeypatch.setattr(
        mcp_imports, "get_json_rpc", lambda path: _FakeJsonRpc, raising=False
    )

    def _fake_upload(ws_fn, api, filename, upload_dir, token):
        with open(filename, encoding="utf-8") as fh:
            uploads.append((filename, upload_dir, fh.read()))
        return {"ok": True}

    monkeypatch.setattr(pdf_mod, "_sync_upload", _fake_upload)








# -----------------------------------------------------------------------
# prepare_attached_documents — text uploads
# -----------------------------------------------------------------------


def _upload_request(files):
    from types import SimpleNamespace

    return SimpleNamespace(
        pdfs=[],
        attached_files=files,
        selected_items=[],
        session_id="sess-1",
        workspace_path="/alice@patricbrc.org/home",
        auth_token="tok",
        parsed_documents=[],
    )


@pytest.mark.asyncio
async def test_upload_persisted_with_original_extension(monkeypatch):
    """A text upload lands in uploaded_files/ under its original name."""
    from orchestrator.documents import prepare_attached_documents

    uploads = []
    _install_fake_workspace(monkeypatch, uploads)

    request = _upload_request([
        {"name": "Ecoli_K12.fasta", "content": ">seq1\nACGT\n",
         "mime_type": "text/plain", "size": 12},
    ])
    await prepare_attached_documents(request)

    (doc,) = request.parsed_documents
    assert doc["name"] == "Ecoli_K12.fasta"
    assert doc["workspace_path"] == (
        "/alice@patricbrc.org/home/chats/sess-1/uploaded_files/Ecoli_K12.fasta"
    )
    assert doc["source"] == "upload"
    assert "page_count" not in doc
    assert "saved_as" not in doc
    # The workspace holds the full original file, not the excerpt.
    assert uploads[0][2] == ">seq1\nACGT\n"


@pytest.mark.asyncio
async def test_two_uploads_with_the_same_name_do_not_overwrite(monkeypatch):
    """Same-named attachments get distinct objects, and we say so."""
    from orchestrator.documents import prepare_attached_documents

    uploads = []
    _install_fake_workspace(monkeypatch, uploads)

    request = _upload_request([
        {"name": "sample.fasta", "content": "first"},
        {"name": "sample.fasta", "content": "second"},
    ])
    await prepare_attached_documents(request)

    first, second = request.parsed_documents
    assert first["workspace_path"] != second["workspace_path"]
    assert second["workspace_path"].endswith(".fasta")
    # name stays what the user attached; saved_as reports the real object
    assert second["name"] == "sample.fasta"
    assert second["saved_as"] != "sample.fasta"
    assert second["workspace_path"].endswith(second["saved_as"])
    assert [body for _, _, body in uploads] == ["first", "second"]


@pytest.mark.asyncio
async def test_upload_excerpt_capped_on_utf8_boundary(monkeypatch):
    """The excerpt honours the 25 KB budget in bytes, not characters."""
    from orchestrator.documents import EXCERPT_BYTES, prepare_attached_documents

    _install_fake_workspace(monkeypatch, [])

    content = "\u00e9" * (EXCERPT_BYTES)  # 2 bytes per char
    request = _upload_request([{"name": "notes.txt", "content": content}])
    await prepare_attached_documents(request)

    (doc,) = request.parsed_documents
    excerpt = doc["excerpt"]
    assert len(excerpt.encode("utf-8")) <= EXCERPT_BYTES
    assert excerpt == content[: EXCERPT_BYTES // 2]
    # char_count still reports the whole file
    assert doc["char_count"] == EXCERPT_BYTES


@pytest.mark.asyncio
async def test_binary_upload_reports_an_error_and_does_not_fail_the_turn(monkeypatch):
    """A null byte yields an {name, error} entry, not an exception."""
    from orchestrator.documents import prepare_attached_documents

    uploads = []
    _install_fake_workspace(monkeypatch, uploads)

    request = _upload_request([
        {"name": "blob.bin", "content": "abc\x00def"},
        {"name": "ok.txt", "content": "fine"},
    ])
    await prepare_attached_documents(request)

    bad, good = request.parsed_documents
    assert bad["name"] == "blob.bin"
    assert "binary" in bad["error"]
    assert "workspace_path" not in bad
    assert good["workspace_path"] is not None
    # Only the good file was written.
    assert len(uploads) == 1


@pytest.mark.asyncio
async def test_upload_survives_a_failed_workspace_write(monkeypatch):
    """A write failure keeps the excerpt; only the path is lost."""
    import bvbrc_mcp.tools.pdf as pdf_mod
    from orchestrator.documents import prepare_attached_documents

    _install_fake_workspace(monkeypatch, [])
    monkeypatch.setattr(
        pdf_mod, "_sync_upload", lambda *a, **k: {"error": "workspace down"}
    )

    request = _upload_request([{"name": "notes.csv", "content": "a,b\n1,2\n"}])
    await prepare_attached_documents(request)

    (doc,) = request.parsed_documents
    assert doc["workspace_path"] is None
    assert doc["excerpt"] == "a,b\n1,2\n"
    assert "error" not in doc


# -----------------------------------------------------------------------
# format_attached_documents
# -----------------------------------------------------------------------


def test_format_successful_document():
    """A successful document renders name, page count, char count, excerpt, txt path."""
    docs = [
        {
            "name": "paper.pdf",
            "page_count": 5,
            "char_count": 12000,
            "workspace_txt_path": "/user@host/home/chats/abc/parsed_pdfs/paper.txt",
            "excerpt": "This is the beginning of the paper...",
        }
    ]

    result = format_attached_documents(docs)

    assert "ATTACHED DOCUMENTS" in result
    assert "paper.pdf" in result
    assert "5 pages" in result
    assert "12000 chars" in result
    assert "/user@host/home/chats/abc/parsed_pdfs/paper.txt" in result
    assert "This is the beginning of the paper..." in result
    assert "read_file_preview" in result


def test_format_failed_document():
    """A failed document renders the error and the 'MUST tell the user' instruction."""
    docs = [
        {
            "name": "corrupt.pdf",
            "error": "PDF is password-protected",
        }
    ]

    result = format_attached_documents(docs)

    assert "corrupt.pdf" in result
    assert "PDF is password-protected" in result
    assert "MUST tell the user" in result


def test_format_empty_input():
    """Empty or None input returns empty string."""
    assert format_attached_documents(None) == ""
    assert format_attached_documents([]) == ""


def test_format_mixed_success_and_failure():
    """Mix of successful and failed documents."""
    docs = [
        {
            "name": "good.pdf",
            "page_count": 3,
            "char_count": 8000,
            "workspace_txt_path": "/u/home/chats/s1/parsed_pdfs/good.txt",
            "excerpt": "Abstract: ...",
        },
        {
            "name": "bad.pdf",
            "error": "Encrypted PDF",
        },
    ]

    result = format_attached_documents(docs)

    # Both documents should appear
    assert "Document 1: good.pdf" in result
    assert "Document 2: bad.pdf" in result

    # Success details
    assert "3 pages" in result
    assert "8000 chars" in result
    assert "Abstract: ..." in result

    # Failure details
    assert "Encrypted PDF" in result
    assert "MUST tell the user" in result


def test_format_document_without_txt_path():
    """A successful document with no workspace_txt_path omits the file reference."""
    docs = [
        {
            "name": "inline.pdf",
            "page_count": 1,
            "char_count": 500,
            "excerpt": "Short doc",
        }
    ]

    result = format_attached_documents(docs)

    assert "inline.pdf" in result
    assert "1 pages" in result
    # Should NOT mention read_file_preview since there's no txt_path
    assert "read_file_preview" not in result


def test_format_document_without_excerpt():
    """A successful document with no excerpt still renders the header."""
    docs = [
        {
            "name": "empty_extract.pdf",
            "page_count": 2,
            "char_count": 0,
            "workspace_txt_path": "/u/home/chats/s/parsed_pdfs/empty_extract.txt",
        }
    ]

    result = format_attached_documents(docs)

    assert "empty_extract.pdf" in result
    assert "2 pages" in result
    assert "0 chars" in result
    # No excerpt section
    assert "Excerpt" not in result


def test_format_upload_omits_pages_and_flags_the_original_file():
    """A text upload has no page_count and is usable as a GoWe input."""
    out = format_attached_documents([
        {
            "name": "Ecoli_K12.fasta",
            "workspace_path": "/alice/home/chats/s1/uploaded_files/Ecoli_K12.fasta",
            "char_count": 4637821,
            "excerpt": ">seq1",
            "source": "upload",
            "mime_type": "text/plain",
        }
    ])

    assert "Ecoli_K12.fasta (4637821 chars)" in out
    assert "pages" not in out
    assert "read_file_preview" in out
    assert "GoWe workflow input" in out


def test_format_reports_a_renamed_upload():
    """saved_as is surfaced so the header and the path do not disagree."""
    out = format_attached_documents([
        {
            "name": "sample.fasta",
            "saved_as": "sample_a1b2c3.fasta",
            "workspace_path": "/alice/home/chats/s1/uploaded_files/sample_a1b2c3.fasta",
            "char_count": 10,
            "excerpt": ">s",
            "source": "upload",
        }
    ])

    assert "sample_a1b2c3.fasta" in out
    assert "same filename" in out
