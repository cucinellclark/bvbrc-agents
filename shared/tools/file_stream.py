"""
Streaming workspace file reader with Range GET, gzip auto-detect,
line iteration, and a 25 MB scan cap.

Every workspace file has a download URL that honors HTTP ``Range``.
This module streams the file through ``httpx`` a chunk at a time,
decompresses gzip on the fly when the magic bytes say so, and
yields raw bytes or lines for callers to consume.

Usage::

    info, stream = await open_workspace_stream(path, token, config)
    async for offset, line in iter_lines(stream):
        ...

Nothing is written to disk; only the compact result reaches the LLM.
"""

from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx

from shared.tools._mcp_imports import get_workspace_functions, get_json_rpc

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCAN_CAP_BYTES: int = 25 * 1024 * 1024  # 25 MB decompressed; scans stop here
CHUNK_BYTES: int = 256 * 1024  # 256 KB per HTTP chunk
GZIP_MAGIC: bytes = b"\x1f\x8b"
MAX_LINE_BYTES: int = 1_000_000  # 1 MB; lines longer than this are yielded truncated
HTTP_TIMEOUT_SECONDS: float = 90.0

# Binary detection: known magic bytes for non-text formats
_BINARY_MAGICS: list[tuple[bytes, str]] = [
    (b"\x89PNG", "png"),
    (b"\xff\xd8\xff", "jpg"),
    (b"GIF8", "gif"),
    (b"PK\x03\x04", "zip"),
    (b"PK\x05\x06", "zip"),
    (b"%PDF", "pdf"),
    (GZIP_MAGIC, "gzip"),  # gzip is "binary" but we decompress transparently
    (b"\x42\x5a\x68", "bzip2"),
    (b"\xfd\x37\x7a\x58\x5a\x00", "xz"),
    (b"Rar!", "rar"),
    (b"\x7fELF", "elf"),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class StreamInfo:
    """Metadata about an open workspace file stream.

    ``bytes_consumed`` / ``partial`` / ``eof`` are updated as the byte
    iterator returned by :func:`open_workspace_stream` is consumed.
    """

    path: str
    total_size: int | None = None  # on-disk (compressed) size from Content-Range
    gzip: bool = False
    bytes_consumed: int = 0  # decompressed bytes yielded so far (from start_byte)
    partial: bool = False  # hit SCAN_CAP_BYTES before EOF
    detected_format: str | None = None  # e.g. "gzip", "png", None for plain text
    eof: bool = False  # true when the underlying stream was read to its end


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _looks_binary(data: bytes) -> tuple[bool, str | None]:
    """Check whether *data* looks like a binary file.

    Returns ``(is_binary, detected_format)``.
    Checks magic bytes first, then scans for null bytes in the first 8 KB.
    gzip is detected as binary but flagged specially so the caller can
    decompress transparently.
    """
    for magic, fmt in _BINARY_MAGICS:
        if data[: len(magic)] == magic:
            return True, fmt
    if b"\x00" in data[:8192]:
        return True, "binary"
    return False, None


async def _get_download_url(path: str, token: str, config: Any) -> str:
    """Resolve the workspace download URL for *path*.

    Raises ``RuntimeError`` with the workspace error text when the lookup
    fails (the MCP helper swallows exceptions and returns ``[error_str]``),
    so callers can map "Object not found" to PATH_NOT_FOUND.
    """
    ws_fn = get_workspace_functions(getattr(config, "mcp_server_path", None))
    json_rpc_mod = get_json_rpc(getattr(config, "mcp_server_path", None))

    ws_url = (
        getattr(config, "bvbrc_workspace_url", None)
        or "https://p3.theseed.org/services/Workspace"
    )
    timeout = getattr(config, "tool_timeout_seconds", 30)

    api = json_rpc_mod.JsonRpcCaller(service_url=ws_url, timeout=timeout)
    result = await ws_fn._get_download_url(api, path, token)
    # Success is [[url]]; failure is ["Error getting download URL: ..."]
    try:
        url = result[0][0]
    except (IndexError, TypeError, KeyError):
        url = None
    if not isinstance(url, str) or not url.startswith("http"):
        raise RuntimeError(f"Could not resolve download URL for {path}: {result}")
    return url


def _parse_total_size(resp: httpx.Response) -> int | None:
    """Total on-disk size from Content-Range (206) or Content-Length (200)."""
    content_range = resp.headers.get("content-range", "")
    if "/" in content_range:
        try:
            return int(content_range.rsplit("/", 1)[1])
        except ValueError:
            return None
    if resp.status_code == 200:
        cl = resp.headers.get("content-length")
        if cl:
            try:
                return int(cl)
            except ValueError:
                return None
    return None


# ---------------------------------------------------------------------------
# Core streaming reader
# ---------------------------------------------------------------------------


async def open_workspace_stream(
    path: str,
    token: str,
    config: Any,
    *,
    start_byte: int = 0,
    max_bytes: int | None = None,
) -> tuple[StreamInfo, AsyncIterator[bytes]]:
    """Open a streaming reader for a workspace file.

    For plain files with ``start_byte > 0``, an HTTP ``Range`` header skips
    to the offset directly — no wasted download.

    For gzip files with ``start_byte > 0``, the compressed stream is
    decompressed from byte 0 and bytes before ``start_byte`` are discarded.
    This costs O(start_byte) but is bounded by ``SCAN_CAP_BYTES``.

    Returns ``(StreamInfo, async_byte_iterator)``.  The iterator yields
    decompressed chunks and stops after ``max_bytes`` (if given) or
    ``SCAN_CAP_BYTES``, whichever is smaller.  ``StreamInfo.partial`` is set
    only when the scan cap was hit; stopping at ``max_bytes`` is not
    "partial".  ``StreamInfo.eof`` is set when the underlying stream ended.

    Binary detection is performed eagerly (before returning) so callers can
    check ``info.detected_format`` immediately; for non-gzip binaries the
    returned iterator is empty.
    """
    download_url = await _get_download_url(path, token, config)
    info = StreamInfo(path=path)
    auth_headers: dict[str, str] = {"Authorization": token}

    # --- Eager probe: detect binary / gzip and total size ---
    async with httpx.AsyncClient(timeout=httpx.Timeout(HTTP_TIMEOUT_SECONDS)) as probe_client:
        probe_headers = dict(auth_headers)
        probe_headers["Range"] = "bytes=0-7"
        # Stream so a server that ignores Range does not download everything
        async with probe_client.stream("GET", download_url, headers=probe_headers) as probe_resp:
            if probe_resp.status_code not in (200, 206):
                probe_resp.raise_for_status()
            info.total_size = _parse_total_size(probe_resp)
            probe_bytes = b""
            async for piece in probe_resp.aiter_bytes(64):
                probe_bytes += piece
                if len(probe_bytes) >= 8:
                    break
            probe_bytes = probe_bytes[:8]

    is_binary, fmt = _looks_binary(probe_bytes)

    if fmt == "gzip":
        info.gzip = True
        info.detected_format = "gzip"
    elif is_binary:
        info.detected_format = fmt
        info.eof = True

        async def _empty() -> AsyncIterator[bytes]:
            return
            yield  # pragma: no cover - makes this an async generator

        return info, _empty()

    limit = SCAN_CAP_BYTES if max_bytes is None else min(max_bytes, SCAN_CAP_BYTES)

    async def _generate() -> AsyncIterator[bytes]:
        yielded = 0  # bytes yielded to the caller

        async with httpx.AsyncClient(timeout=httpx.Timeout(HTTP_TIMEOUT_SECONDS)) as client:
            stream_headers = dict(auth_headers)
            if info.gzip:
                decompressor: zlib._Decompress | None = zlib.decompressobj(16 + zlib.MAX_WBITS)
                to_skip = start_byte  # gzip cannot seek: decompress and discard
            else:
                decompressor = None
                to_skip = 0
                if start_byte > 0:
                    stream_headers["Range"] = f"bytes={start_byte}-"

            async with client.stream("GET", download_url, headers=stream_headers) as resp:
                if resp.status_code == 416:
                    # Range starts past EOF: nothing to read
                    info.eof = True
                    return
                if resp.status_code not in (200, 206):
                    resp.raise_for_status()
                if info.total_size is None:
                    info.total_size = _parse_total_size(resp)

                async def _pieces() -> AsyncIterator[bytes]:
                    async for raw in resp.aiter_bytes(CHUNK_BYTES):
                        if decompressor is None:
                            yield raw
                            continue
                        try:
                            out = decompressor.decompress(raw)
                        except zlib.error as e:
                            logger.warning("gzip decompression error for %s: %s", path, e)
                            return
                        if out:
                            yield out
                    if decompressor is not None:
                        try:
                            tail = decompressor.flush()
                        except zlib.error:
                            tail = b""
                        if tail:
                            yield tail

                async for chunk in _pieces():
                    # Discard bytes before start_byte (gzip only)
                    if to_skip:
                        if len(chunk) <= to_skip:
                            to_skip -= len(chunk)
                            scanned_skip = start_byte - to_skip
                            if scanned_skip >= SCAN_CAP_BYTES:
                                info.partial = True
                                return
                            continue
                        chunk = chunk[to_skip:]
                        to_skip = 0

                    remaining = limit - yielded
                    if len(chunk) > remaining:
                        chunk = chunk[:remaining]
                    if chunk:
                        yielded += len(chunk)
                        info.bytes_consumed = yielded
                        yield chunk
                    if yielded >= limit:
                        if limit >= SCAN_CAP_BYTES:
                            info.partial = True
                        return

                info.eof = True

    return info, _generate()


async def iter_lines(
    stream: AsyncIterator[bytes],
    *,
    max_line_bytes: int = MAX_LINE_BYTES,
) -> AsyncIterator[tuple[int, bytes]]:
    """Yield ``(byte_offset_of_line_start, line_without_newline)`` tuples.

    Offsets are relative to the first byte the stream yields (i.e. to the
    ``start_byte`` the stream was opened with).  A single line longer than
    *max_line_bytes* is yielded in ``max_line_bytes`` pieces.  Handles both
    ``\\n`` and ``\\r\\n`` line endings.
    """
    buffer = b""
    offset = 0  # byte offset of the start of the current buffer content

    async for chunk in stream:
        buffer += chunk
        while True:
            nl = buffer.find(b"\n")
            if nl == -1:
                if len(buffer) > max_line_bytes:
                    piece = buffer[:max_line_bytes]
                    yield offset, piece
                    buffer = buffer[max_line_bytes:]
                    offset += max_line_bytes
                break

            line = buffer[:nl]
            if line.endswith(b"\r"):
                line = line[:-1]
            yield offset, line
            consumed = nl + 1
            offset += consumed
            buffer = buffer[consumed:]

    if buffer:
        line = buffer[:-1] if buffer.endswith(b"\r") else buffer
        yield offset, line
