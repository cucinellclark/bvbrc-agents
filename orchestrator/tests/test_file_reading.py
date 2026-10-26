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

import gzip
import json
import os
import sys
import tempfile
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
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
        "bvbrc_mcp.tools.file_stream._get_download_url",
        side_effect=fake_get_download_url,
    )


# ---------------------------------------------------------------------------
# Test: Paging continuity
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Test: UTF-8 multi-byte safety
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Test: Gzip transparent decompression
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Test: Binary refusal
# ---------------------------------------------------------------------------



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



# ---------------------------------------------------------------------------
# Test: arbitrary (mid-line) start_byte aligns forward to the next line
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Test: gzip paging reconstructs the file at decompressed offsets
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Test: missing file surfaces PATH_NOT_FOUND, not a URL error
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Test: iter_lines offsets and CRLF handling
# ---------------------------------------------------------------------------



# ===========================================================================
# Phase 2: search_file
# ===========================================================================

def _write(tmpdir, name, text, gz=False):
    path = os.path.join(tmpdir, name)
    if gz:
        with gzip.open(path, "wb") as f:
            f.write(text.encode("utf-8"))
    else:
        with open(path, "wb") as f:
            f.write(text.encode("utf-8"))
    return path


async def _search(base_url, filename, **kw):
    from bvbrc_mcp.tools.file_search import search_file
    kw.setdefault("config", MockConfig())
    kw.setdefault("headers", _mock_headers())
    with _patch_download_url(base_url, filename):
        return await search_file(path=f"/test@patricbrc.org/home/{filename}", **kw)














def test_search_result_fits_loop_cap():
    """A worst-case result (200 matches x 300-char lines) stays under the
    search_file result cap so truncate_result never hard-cuts it."""
    from shared.tools import truncate_result, result_char_limit
    from bvbrc_mcp.tools.file_search import RESULT_CHAR_BUDGET
    limit = result_char_limit("search_file")
    assert RESULT_CHAR_BUDGET < limit
    # envelope on top of the matches budget must still fit
    envelope = {"matches": [], "match_count": 0, "truncated": True, "scanned_bytes": 26214400,
                "partial": True, "gzip": True, "offset_space": "decompressed", "total_size": 10**9,
                "workspace_path": "/x" * 100, "pattern": "p" * 2000, "regex": True,
                "line_numbers_relative_to": 10**9, "next_start": 10**9,
                "next_step": "x" * 200, "_summary": {"first_offsets": list(range(10))}}
    assert len(json.dumps(envelope, indent=2)) + RESULT_CHAR_BUDGET <= limit
    assert "[TRUNCATED" not in truncate_result(envelope, max_chars=limit)




