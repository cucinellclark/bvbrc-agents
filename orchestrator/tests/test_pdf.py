"""Tests for the PDF extraction pipeline.

Covers:
  - extract_text_from_pdf (shared.tools.pdf)
  - decode_workspace_download (shared.tools.pdf)
  - _filename_with_dedup / persist_session_file (shared.tools.pdf)
  - prepare_attached_documents text uploads (orchestrator.documents)
  - format_attached_documents (shared.agent_utils)
"""

import base64
import sys
from pathlib import Path

# Ensure the bvbrc-agents repo root is importable so that
# ``shared.tools.pdf`` and ``shared.agent_utils`` resolve.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import fitz  # PyMuPDF
import pytest

from shared.tools.pdf import (
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


def test_extract_text_single_page():
    """Single-page PDF with known text."""
    pdf = _make_pdf_bytes("Hello World")
    text, page_count = extract_text_from_pdf(pdf)

    assert page_count == 1
    assert "Hello World" in text


def test_extract_text_multi_page():
    """Multi-page PDF returns concatenated text and correct page count."""
    pdf = _make_pdf_bytes("Page one content", "Page two content")
    text, page_count = extract_text_from_pdf(pdf)

    assert page_count == 2
    assert "Page one content" in text
    assert "Page two content" in text


def test_extract_text_scanned_pdf():
    """A page with no text layer yields empty text but page_count > 0."""
    pdf = _make_pdf_bytes()  # blank page, no text inserted
    text, page_count = extract_text_from_pdf(pdf)

    assert page_count == 1
    assert text.strip() == ""


# -----------------------------------------------------------------------
# decode_workspace_download
# -----------------------------------------------------------------------


def test_decode_base64_sentinel():
    """Unwrap <base64_encoded_data>...</base64_encoded_data> sentinel."""
    raw = b"binary PDF content \x00\xff"
    encoded = base64.b64encode(raw).decode("ascii")
    wrapped = f"<base64_encoded_data>{encoded}</base64_encoded_data>"

    assert decode_workspace_download(wrapped) == raw


def test_decode_plain_text_passthrough():
    """Plain text string (no sentinel) is re-encoded to UTF-8 bytes."""
    plain = "just some text"
    result = decode_workspace_download(plain)

    assert result == b"just some text"


def test_decode_dict_with_data_key():
    """Dict envelope with ``"data"`` key is recursively unwrapped."""
    raw = b"\x89PNG"
    encoded = base64.b64encode(raw).decode("ascii")
    envelope = {"data": f"<base64_encoded_data>{encoded}</base64_encoded_data>"}

    assert decode_workspace_download(envelope) == raw


def test_decode_dict_plain_text_data():
    """Dict envelope whose ``"data"`` is plain text."""
    envelope = {"data": "hello from workspace"}
    result = decode_workspace_download(envelope)

    assert result == b"hello from workspace"


def test_decode_raw_bytes_passthrough():
    """Raw bytes are returned unchanged."""
    raw = b"\x00\x01\x02"
    assert decode_workspace_download(raw) is raw


def test_decode_unexpected_type_raises():
    """Unexpected type raises ValueError."""
    with pytest.raises(ValueError, match="unexpected type"):
        decode_workspace_download(12345)


# -----------------------------------------------------------------------
# _filename_with_dedup
# -----------------------------------------------------------------------


def test_filename_no_collision():
    """When the filename is not in existing_names, it is returned as-is."""
    assert _filename_with_dedup("report.txt", {"other.txt"}) == "report.txt"


def test_filename_collision_appends_hash_before_extension():
    """On collision a 6-hex-char suffix goes before the extension."""
    result = _filename_with_dedup("report.txt", {"report.txt"})

    assert result != "report.txt"
    assert result.startswith("report_")
    assert result.endswith(".txt")
    suffix = result[len("report_"):-len(".txt")]
    assert len(suffix) == 6
    assert all(c in "0123456789abcdef" for c in suffix)


def test_filename_collision_deterministic():
    """The dedup suffix is deterministic for the same filename."""
    existing = {"myfile.fasta"}
    assert _filename_with_dedup("myfile.fasta", existing) == _filename_with_dedup(
        "myfile.fasta", existing
    )


def test_filename_third_collision_gets_a_counter():
    """A third same-named file must not land on the second one's name."""
    existing = {"sample.fasta"}
    second = _filename_with_dedup("sample.fasta", existing)
    existing.add(second)
    third = _filename_with_dedup("sample.fasta", existing)

    assert third not in ("sample.fasta", second)
    assert third.endswith(".fasta")


def test_filename_extensionless_still_dedups():
    """A name with no extension still gets a suffix."""
    result = _filename_with_dedup("README", {"README"})
    assert result != "README"
    assert result.startswith("README_")


def test_filename_empty_existing():
    """Empty existing_names set means no collision."""
    assert _filename_with_dedup("anything.csv", set()) == "anything.csv"


# -----------------------------------------------------------------------
# persist_session_file
# -----------------------------------------------------------------------


class _FakeConfig:
    mcp_server_path = None
    bvbrc_workspace_url = "https://example.invalid/Workspace"
    tool_timeout_seconds = 5


def _install_fake_workspace(monkeypatch, uploads):
    """Stub the MCP workspace layer; record (local_path, upload_dir)."""
    import shared.tools._mcp_imports as mcp_imports
    import shared.tools.pdf as pdf_mod

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


@pytest.mark.asyncio
async def test_persist_uses_the_intended_basename(monkeypatch):
    """The temp file handed to workspace_upload must carry the real name.

    workspace_upload derives the workspace object name from
    os.path.basename(), so a mkstemp-style name would create an object
    whose name does not match the path reported to the agent.
    """
    uploads = []
    _install_fake_workspace(monkeypatch, uploads)

    path = await persist_session_file(
        content=">seq1\nACGT\n",
        filename="Ecoli_K12.fasta",
        subfolder="uploaded_files",
        session_id="sess-1",
        workspace_path="/alice@patricbrc.org/home",
        token="tok",
        config=_FakeConfig(),
    )

    assert path == (
        "/alice@patricbrc.org/home/.chats/sess-1/uploaded_files/Ecoli_K12.fasta"
    )
    local_path, upload_dir, body = uploads[0]
    import os as _os

    assert _os.path.basename(local_path) == "Ecoli_K12.fasta"
    assert upload_dir == "/alice@patricbrc.org/home/.chats/sess-1/uploaded_files"
    assert body == ">seq1\nACGT\n"


@pytest.mark.asyncio
async def test_persist_reserves_the_name_it_writes(monkeypatch):
    """existing_names is mutated so the next call cannot reuse the name."""
    uploads = []
    _install_fake_workspace(monkeypatch, uploads)
    used: set[str] = set()

    kwargs = dict(
        subfolder="uploaded_files",
        session_id="sess-1",
        workspace_path="/alice@patricbrc.org/home",
        token="tok",
        config=_FakeConfig(),
        existing_names=used,
    )

    first = await persist_session_file(content="a", filename="s.fasta", **kwargs)
    second = await persist_session_file(content="b", filename="s.fasta", **kwargs)
    third = await persist_session_file(content="c", filename="s.fasta", **kwargs)

    assert first.endswith("/s.fasta")
    # Each write lands on a distinct object, and every name is reserved.
    assert len({first, second, third}) == 3
    assert len(used) == 3
    assert [body for _, _, body in uploads] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_persist_returns_none_when_upload_fails(monkeypatch):
    """An upload error yields None rather than a path that 404s."""
    import shared.tools.pdf as pdf_mod

    _install_fake_workspace(monkeypatch, [])
    monkeypatch.setattr(
        pdf_mod,
        "_sync_upload",
        lambda *a, **k: {"error": "permission denied"},
    )

    path = await persist_session_file(
        content="x",
        filename="s.txt",
        subfolder="uploaded_files",
        session_id="sess-1",
        workspace_path="/alice@patricbrc.org/home",
        token="tok",
        config=_FakeConfig(),
    )
    assert path is None


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
        "/alice@patricbrc.org/home/.chats/sess-1/uploaded_files/Ecoli_K12.fasta"
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
    import shared.tools.pdf as pdf_mod
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
            "workspace_txt_path": "/user@host/home/.chats/abc/parsed_pdfs/paper.txt",
            "excerpt": "This is the beginning of the paper...",
        }
    ]

    result = format_attached_documents(docs)

    assert "ATTACHED DOCUMENTS" in result
    assert "paper.pdf" in result
    assert "5 pages" in result
    assert "12000 chars" in result
    assert "/user@host/home/.chats/abc/parsed_pdfs/paper.txt" in result
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
            "workspace_txt_path": "/u/home/.chats/s1/parsed_pdfs/good.txt",
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
            "workspace_txt_path": "/u/home/.chats/s/parsed_pdfs/empty_extract.txt",
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
            "workspace_path": "/alice/home/.chats/s1/uploaded_files/Ecoli_K12.fasta",
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
            "workspace_path": "/alice/home/.chats/s1/uploaded_files/sample_a1b2c3.fasta",
            "char_count": 10,
            "excerpt": ">s",
            "source": "upload",
        }
    ])

    assert "sample_a1b2c3.fasta" in out
    assert "same filename" in out
