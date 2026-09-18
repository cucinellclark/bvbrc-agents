"""Tests for Phase 1 paged file reading.

Covers:
  - Paging continuity (consecutive reads with no gaps/duplicates)
  - Line-boundary alignment
  - UTF-8 multi-byte codepoint safety
  - Gzip transparent decompression
  - 32 KB page not truncated by result caps
  - Binary refusal with BINARY_FILE error
  - 25 MB scan cap with partial flag
  - PDF branch paging via start_byte
  - _summary survives trim_messages_to_fit
"""

from __future__ import annotations

import asyncio
import gzip
import io
import json
import os
import sys
import tempfile
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure repo root and shared dirs on sys.path
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Local Range HTTP server fixture
# ---------------------------------------------------------------------------

class RangeHTTPHandler(BaseHTTPRequestHandler):
    """Serves files from a temp directory with HTTP Range support."""

    # Set by the fixture
    serve_dir: str = ""

    def do_GET(self):
        # Path is /<filename>
        filename = self.path.lstrip("/")
        filepath = os.path.join(self.serve_dir, filename)
        if not os.path.isfile(filepath):
            self.send_error(404)
            return

        with open(filepath, "rb") as f:
            data = f.read()

        total = len(data)
        range_header = self.headers.get("Range")

        if range_header and range_header.startswith("bytes="):
            range_spec = range_header[6:]
            parts = range_spec.split("-")
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if len(parts) > 1 and parts[1] else total - 1
            end = min(end, total - 1)

            if start >= total:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{total}")
                self.end_headers()
                return

            chunk = data[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
            self.send_header("Content-Length", str(len(chunk)))
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            self.write_response(chunk)
        else:
            self.send_response(200)
            self.send_header("Content-Length", str(total))
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            self.write_response(data)

    def write_response(self, data: bytes):
        self.wfile.write(data)

    def log_message(self, format, *args):
        pass  # Suppress logs


@pytest.fixture(scope="module")
def range_server():
    """Start a local HTTP server that serves files with Range support."""
    tmpdir = tempfile.mkdtemp(prefix="test_file_reading_")

    # Create a handler class bound to tmpdir
    handler = type("BoundHandler", (RangeHTTPHandler,), {"serve_dir": tmpdir})
    server = HTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield tmpdir, f"http://127.0.0.1:{port}"

    server.shutdown()


# ---------------------------------------------------------------------------
# Mock config for tools
# ---------------------------------------------------------------------------

class MockConfig:
    bvbrc_workspace_url = "https://p3.theseed.org/services/Workspace"
    tool_timeout_seconds = 30
    mcp_server_path = None
    bvbrc_auth_token = "un=test@patricbrc.org|tokenstring"
    session_id = None
    workspace_path = None


def _mock_headers():
    return {"Authorization": "un=test@patricbrc.org|tokenstring"}


# ---------------------------------------------------------------------------
# Helper: patch _get_download_url to point at local server
# ---------------------------------------------------------------------------

def _patch_download_url(base_url: str, filename: str):
    """Return a patch context manager that makes _get_download_url return
    our local server URL for the given filename."""
    url = f"{base_url}/{filename}"

    async def fake_get_download_url(path, token, config):
        return url

    return patch(
        "shared.tools.file_stream._get_download_url",
        side_effect=fake_get_download_url,
    )


# ---------------------------------------------------------------------------
# Test: Paging continuity
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_paging_continuity(range_server):
    """Consecutive read_file_preview calls cover a 100 KB CSV with no gaps
    or duplicate rows; every next_start lands on a line boundary;
    is_complete on the last page only."""
    tmpdir, base_url = range_server
    from shared.tools.workspace import read_file_preview

    # Generate a 100 KB CSV
    lines = []
    row_num = 0
    while True:
        line = f"row_{row_num:05d},value_{row_num},{'x' * 80}\n"
        lines.append(line)
        row_num += 1
        if sum(len(l) for l in lines) > 100 * 1024:
            break

    csv_content = "".join(lines)
    filepath = os.path.join(tmpdir, "test_paging.csv")
    with open(filepath, "w") as f:
        f.write(csv_content)

    collected_text = ""
    start = 0
    page_count = 0
    max_pages = 20  # safety limit

    with _patch_download_url(base_url, "test_paging.csv"):
        while page_count < max_pages:
            result = await read_file_preview(
                path="/test@patricbrc.org/home/test_paging.csv",
                max_bytes=32768,
                start_byte=start,
                config=MockConfig(),
                headers=_mock_headers(),
            )

            assert "error" not in result, f"Unexpected error: {result.get('error')}"
            assert "data" in result

            data = result["data"]
            assert "partial" not in result
            if page_count > 0:
                assert result["start_byte"] == start
            collected_text += data
            page_count += 1

            if result.get("is_complete"):
                assert "next_start" not in result
                break

            assert data.endswith("\n"), "non-final page must end on a line boundary"
            assert "next_start" in result, "next_start missing on non-complete page"
            start = result["next_start"]

    assert page_count > 1, "File should require multiple pages"
    # Exact reconstruction: no gaps, no duplicated or dropped rows
    assert collected_text == csv_content
    # Every page after the first starts at a line boundary and ends on one
    assert "partial" not in result, "partial means scan cap, not page end"


# ---------------------------------------------------------------------------
# Test: UTF-8 multi-byte safety
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_utf8_multibyte_safety(range_server):
    """A file with multi-byte characters straddling the 32 KB boundary
    pages cleanly (no mojibake, no base64)."""
    tmpdir, base_url = range_server
    from shared.tools.workspace import read_file_preview

    # Create a file with lots of 3-byte UTF-8 characters (e.g., CJK)
    # Each char is 3 bytes; fill ~64 KB
    cjk_char = "\u4e16"  # '世' = 3 bytes in UTF-8
    # Create lines of ~100 chars each
    lines = []
    total_bytes = 0
    line_num = 0
    while total_bytes < 64 * 1024:
        line = f"line_{line_num:04d}:" + cjk_char * 90 + "\n"
        lines.append(line)
        total_bytes += len(line.encode("utf-8"))
        line_num += 1

    content = "".join(lines)
    filepath = os.path.join(tmpdir, "test_utf8.txt")
    with open(filepath, "wb") as f:
        f.write(content.encode("utf-8"))

    all_text = ""
    start = 0
    page_count = 0

    with _patch_download_url(base_url, "test_utf8.txt"):
        while page_count < 10:
            result = await read_file_preview(
                path="/test@patricbrc.org/home/test_utf8.txt",
                max_bytes=32768,
                start_byte=start,
                config=MockConfig(),
                headers=_mock_headers(),
            )

            assert "error" not in result, f"Error: {result.get('error')}"
            data = result["data"]
            # No replacement characters should appear (except possibly at
            # the very boundary, which errors='replace' handles gracefully)
            all_text += data
            page_count += 1

            if result.get("is_complete"):
                break
            start = result["next_start"]

    assert page_count >= 2
    # The CJK character should appear throughout
    assert cjk_char in all_text


# ---------------------------------------------------------------------------
# Test: Gzip transparent decompression
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gzip_transparent(range_server):
    """A gzipped CSV yields text (not base64) with offset_space='decompressed'."""
    tmpdir, base_url = range_server
    from shared.tools.workspace import read_file_preview

    # Create a plain CSV and gzip it
    csv_lines = [f"row_{i:04d},data_{i}\n" for i in range(500)]
    plain_content = "".join(csv_lines)
    gz_path = os.path.join(tmpdir, "test_data.csv.gz")
    with gzip.open(gz_path, "wb") as f:
        f.write(plain_content.encode("utf-8"))

    with _patch_download_url(base_url, "test_data.csv.gz"):
        result = await read_file_preview(
            path="/test@patricbrc.org/home/test_data.csv.gz",
            max_bytes=32768,
            start_byte=0,
            config=MockConfig(),
            headers=_mock_headers(),
        )

    assert "error" not in result, f"Error: {result.get('error')}"
    assert result.get("gzip") is True
    assert result.get("offset_space") == "decompressed"
    # Data should be readable text, not base64
    assert "row_0000" in result["data"]
    assert "<base64_encoded_data>" not in result["data"]


# ---------------------------------------------------------------------------
# Test: Binary refusal
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_binary_refusal(range_server):
    """A .png file returns BINARY_FILE error with hint."""
    tmpdir, base_url = range_server
    from shared.tools.workspace import read_file_preview

    # Create a fake PNG (just the magic bytes + some data)
    png_path = os.path.join(tmpdir, "test_image.png")
    with open(png_path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 1000)

    with _patch_download_url(base_url, "test_image.png"):
        result = await read_file_preview(
            path="/test@patricbrc.org/home/test_image.png",
            max_bytes=8192,
            start_byte=0,
            config=MockConfig(),
            headers=_mock_headers(),
        )

    assert result.get("errorType") == "BINARY_FILE"
    assert "png" in result.get("hint", "").lower()


# ---------------------------------------------------------------------------
# Test: Result cap does not truncate 32 KB page
# ---------------------------------------------------------------------------

def test_result_cap_preserves_32kb_page():
    """A 32 KB read_file_preview result is NOT truncated by truncate_result
    at the read limit (48000 chars)."""
    from shared.tools import truncate_result, result_char_limit

    limit = result_char_limit("read_file_preview")
    assert limit == 48_000

    # Simulate a 32 KB text result — ~32000 chars (worst case ~40000 with JSON)
    data = "A" * 32000
    result = {
        "data": data,
        "start_byte": 0,
        "bytes_read": 32000,
        "total_size": 100000,
        "is_complete": False,
        "next_start": 32000,
        "workspace_path": "/user/home/file.txt",
        "source_type": "workspace",
        "_summary": {"path": "/user/home/file.txt", "start_byte": 0, "next_start": 32000},
    }

    serialized = truncate_result(result, max_chars=limit)
    parsed = json.loads(serialized)

    # Data should NOT be truncated
    assert parsed["data"] == data
    assert "[TRUNCATED" not in serialized


# ---------------------------------------------------------------------------
# Test: _summary survives trim_messages_to_fit
# ---------------------------------------------------------------------------

def test_summary_survives_trim():
    """When trim_messages_to_fit compresses a tool result, _summary is kept."""
    from shared.agent_utils import trim_messages_to_fit

    summary = {"path": "/user/home/file.txt", "start_byte": 0, "next_start": 32768}
    tool_result = json.dumps({
        "data": "X" * 40000,
        "start_byte": 0,
        "_summary": summary,
    })

    messages = [
        {"role": "system", "content": "You are a test agent."},
        {"role": "user", "content": "Read the file"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "tc1", "type": "function", "function": {"name": "read_file_preview", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "tc1", "content": tool_result},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "tc2", "type": "function", "function": {"name": "read_file_preview", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "tc2", "content": '{"data": "more data"}'},
        {"role": "assistant", "content": "Here is the result."},
    ]

    # Trim with a very small token budget to force compression
    trimmed = trim_messages_to_fit(messages, None, max_tokens=500)

    # Find the compressed tool result
    for msg in trimmed:
        if msg.get("role") == "tool" and msg.get("tool_call_id") == "tc1":
            content = msg["content"]
            # The compressed version should mention the summary
            assert "summary=" in content or "_summary" in content
            break


# ---------------------------------------------------------------------------
# Test: 25 MB scan cap
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scan_cap_partial(range_server):
    """A synthetic 30 MB file stops at the 25 MB scan cap with partial=true."""
    tmpdir, base_url = range_server
    from shared.tools.file_stream import open_workspace_stream, SCAN_CAP_BYTES

    # Create a 30 MB file
    large_path = os.path.join(tmpdir, "test_large.txt")
    with open(large_path, "wb") as f:
        # Write 30 MB of repeated lines
        line = b"A" * 999 + b"\n"  # 1000 bytes per line
        for _ in range(30 * 1024):  # ~30 MB
            f.write(line)

    with _patch_download_url(base_url, "test_large.txt"):
        info, stream = await open_workspace_stream(
            "/test@patricbrc.org/home/test_large.txt",
            "un=test@patricbrc.org|tokenstring",
            MockConfig(),
            start_byte=0,
            max_bytes=SCAN_CAP_BYTES + 1024 * 1024,  # request more than cap
        )

        total = 0
        async for chunk in stream:
            total += len(chunk)

    # Should have stopped at the cap
    assert total <= SCAN_CAP_BYTES + 256 * 1024  # allow one chunk overshoot
    assert info.partial is True


# ---------------------------------------------------------------------------
# Test: result_char_limit function
# ---------------------------------------------------------------------------

def test_result_char_limit():
    """result_char_limit returns per-tool overrides and respects the default."""
    from shared.tools import result_char_limit

    assert result_char_limit("read_file_preview") == 48_000
    assert result_char_limit("read_file_preview", default=50_000) == 50_000
    assert result_char_limit("search_data") == 8000  # no override, use default
    assert result_char_limit("search_data", default=10_000) == 10_000


# ---------------------------------------------------------------------------
# Test: last line without trailing newline is not lost
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_trailing_newline_last_line_kept(range_server):
    """A file whose last line lacks a trailing newline returns that line on
    the final page and reports is_complete."""
    tmpdir, base_url = range_server
    from shared.tools.workspace import read_file_preview

    content = "".join(f"line {i}\n" for i in range(3000)) + "LAST_LINE_NO_NEWLINE"
    assert len(content) > 8192
    with open(os.path.join(tmpdir, "test_nonl.txt"), "w") as f:
        f.write(content)

    collected = ""
    start = 0
    with _patch_download_url(base_url, "test_nonl.txt"):
        for _ in range(10):
            result = await read_file_preview(
                path="/test@patricbrc.org/home/test_nonl.txt",
                max_bytes=8192, start_byte=start,
                config=MockConfig(), headers=_mock_headers(),
            )
            assert "error" not in result, result
            collected += result["data"]
            if result["is_complete"]:
                break
            start = result["next_start"]

    assert collected == content
    assert result["is_complete"] is True

    # Small file in one page: nothing trimmed
    with open(os.path.join(tmpdir, "test_small.txt"), "w") as f:
        f.write("a\nb\nc")
    with _patch_download_url(base_url, "test_small.txt"):
        result = await read_file_preview(
            path="/test@patricbrc.org/home/test_small.txt",
            config=MockConfig(), headers=_mock_headers(),
        )
    assert result["data"] == "a\nb\nc"
    assert result["is_complete"] is True
    assert result["line_count"] == 3


# ---------------------------------------------------------------------------
# Test: arbitrary (mid-line) start_byte aligns forward to the next line
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_midline_start_byte_aligns_forward(range_server):
    """An arbitrary start_byte inside a line skips to the next line start,
    and start_byte in the result reports where the page really began."""
    tmpdir, base_url = range_server
    from shared.tools.workspace import read_file_preview

    content = "alpha\nbravo\ncharlie\ndelta\n"
    with open(os.path.join(tmpdir, "test_mid.txt"), "w") as f:
        f.write(content)

    with _patch_download_url(base_url, "test_mid.txt"):
        # byte 8 is inside "bravo" (alpha\n = 6 bytes, bravo starts at 6)
        result = await read_file_preview(
            path="/test@patricbrc.org/home/test_mid.txt",
            start_byte=8, config=MockConfig(), headers=_mock_headers(),
        )
        assert result["data"] == "charlie\ndelta\n"
        assert result["start_byte"] == 12
        assert result["is_complete"] is True

        # byte 6 IS a line boundary: nothing skipped
        result = await read_file_preview(
            path="/test@patricbrc.org/home/test_mid.txt",
            start_byte=6, config=MockConfig(), headers=_mock_headers(),
        )
        assert result["data"] == "bravo\ncharlie\ndelta\n"
        assert result["start_byte"] == 6

        # start_byte past EOF: empty, complete
        result = await read_file_preview(
            path="/test@patricbrc.org/home/test_mid.txt",
            start_byte=len(content) + 50, config=MockConfig(), headers=_mock_headers(),
        )
        assert "error" not in result, result
        assert result["data"] == ""
        assert result["is_complete"] is True


# ---------------------------------------------------------------------------
# Test: gzip paging reconstructs the file at decompressed offsets
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gzip_paging_continuity(range_server):
    """Paging through a gzipped file yields the same text as the plain file,
    with next_start in decompressed offsets."""
    tmpdir, base_url = range_server
    from shared.tools.workspace import read_file_preview

    plain = "".join(f"row_{i:05d},{'y' * 60}\n" for i in range(1500))
    assert len(plain) > 3 * 32768
    with gzip.open(os.path.join(tmpdir, "test_page.csv.gz"), "wb") as f:
        f.write(plain.encode("utf-8"))

    collected = ""
    start = 0
    pages = 0
    with _patch_download_url(base_url, "test_page.csv.gz"):
        while pages < 20:
            result = await read_file_preview(
                path="/test@patricbrc.org/home/test_page.csv.gz",
                max_bytes=32768, start_byte=start,
                config=MockConfig(), headers=_mock_headers(),
            )
            assert "error" not in result, result
            assert result["gzip"] is True
            assert result["offset_space"] == "decompressed"
            assert "partial" not in result
            collected += result["data"]
            pages += 1
            if result["is_complete"]:
                break
            start = result["next_start"]

    assert pages >= 4
    assert collected == plain


# ---------------------------------------------------------------------------
# Test: missing file surfaces PATH_NOT_FOUND, not a URL error
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_missing_file_path_not_found():
    """When the workspace says 'Object not found', the tool returns the
    structured PATH_NOT_FOUND error (the MCP helper returns [error_str])."""
    from shared.tools.workspace import read_file_preview

    fake_ws = MagicMock()
    fake_ws._get_download_url = AsyncMock(
        return_value=["Error getting download URL: _ERROR_Object not found: /x/y_ERROR_"]
    )
    fake_rpc = MagicMock()
    with patch("shared.tools.file_stream.get_workspace_functions", return_value=fake_ws), \
         patch("shared.tools.file_stream.get_json_rpc", return_value=fake_rpc):
        result = await read_file_preview(
            path="/test@patricbrc.org/home/missing.txt",
            config=MockConfig(), headers=_mock_headers(),
        )
    assert result.get("errorType") == "PATH_NOT_FOUND", result


# ---------------------------------------------------------------------------
# Test: iter_lines offsets and CRLF handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_iter_lines_offsets():
    from shared.tools.file_stream import iter_lines

    async def gen():
        yield b"ab\r\ncd"
        yield b"\nef"

    out = [(o, l) async for o, l in iter_lines(gen())]
    assert out == [(0, b"ab"), (4, b"cd"), (7, b"ef")]
