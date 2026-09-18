"""
``search_file`` — grep-style search within a single workspace file.

Streams the file through :mod:`shared.tools.file_stream` (Range GET, gzip
transparent, 25 MB scan cap) and returns matching lines with the byte
offset of each match so the LLM can jump there with
``read_file_preview(start_byte=...)``.  Only the compact match list reaches
the LLM; the file itself never does.
"""

from __future__ import annotations

import json
import re
from collections import deque
from typing import Any, Dict, List, Optional

from shared.tools.file_stream import SCAN_CAP_BYTES, iter_lines, open_workspace_stream
from shared.tools.workspace import (
    _extract_token,
    _extract_user_id,
    _is_object_not_found,
    _path_not_found_result,
    _resolve_path,
)

MAX_PATTERN_CHARS = 2000
MAX_MATCHES_CAP = 200
MAX_CONTEXT_LINES = 5
MAX_LINE_CHARS = 300          # each returned line is cut to this many chars
RESULT_CHAR_BUDGET = 12_000   # matches JSON (indent=2, as truncate_result serializes)
                              # + ~3 000 envelope stays under the loop's 16 000 cap
_FASTA_EXTS = (".fa", ".fasta", ".fna", ".faa", ".ffn", ".frn")


def _decode(line: bytes) -> str:
    return line.decode("utf-8", errors="replace")


def _clip(text: str) -> tuple[str, bool]:
    if len(text) > MAX_LINE_CHARS:
        return text[:MAX_LINE_CHARS], True
    return text, False


def _strip_gz(name: str) -> str:
    return name[:-3] if name.lower().endswith(".gz") else name


async def search_file(
    path: str,
    pattern: str,
    regex: bool = False,
    case_sensitive: bool = False,
    max_matches: int = 50,
    context_lines: int = 0,
    start_byte: int = 0,
    config: Any = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Search a workspace file for *pattern*, line by line.

    Literal (substring) match by default; ``regex=True`` compiles *pattern*
    as a Python regular expression.  Case-insensitive unless
    ``case_sensitive=True``.

    Every match carries ``byte_offset`` — the offset of the start of the
    matching line (decompressed offset for gzip) — for a follow-up
    ``read_file_preview(path, start_byte=byte_offset)``.  When the file is
    FASTA and the match is a header line, ``record_length`` gives the number
    of sequence characters in that record.

    Stops after ``max_matches`` (≤200) or the 25 MB scan cap; in either
    case ``next_start`` says where to resume with ``start_byte``.
    """
    token = _extract_token(headers)
    user_id = _extract_user_id(headers)
    if not token:
        return {
            "error": "Authentication required for workspace operations.",
            "errorType": "AUTHENTICATION_ERROR",
            "source": "bvbrc-workspace",
        }

    resolved_path = _resolve_path(path, user_id)

    if not pattern:
        return {"error": "pattern must not be empty", "errorType": "INVALID_PARAMETERS"}
    if len(pattern) > MAX_PATTERN_CHARS:
        return {
            "error": f"pattern is longer than {MAX_PATTERN_CHARS} characters",
            "errorType": "INVALID_PARAMETERS",
        }
    max_matches = min(max(int(max_matches), 1), MAX_MATCHES_CAP)
    context_lines = min(max(int(context_lines), 0), MAX_CONTEXT_LINES)
    start_byte = max(int(start_byte), 0)

    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        compiled = re.compile(pattern if regex else re.escape(pattern), flags)
    except re.error as e:
        return {
            "error": f"Invalid regular expression: {e}",
            "errorType": "INVALID_PARAMETERS",
            "pattern": pattern,
        }

    if resolved_path.lower().endswith(".pdf"):
        return {
            "error": "search_file does not search PDFs directly.",
            "errorType": "UNSUPPORTED_FILE",
            "hint": (
                "Call read_file_preview on the PDF once — it extracts the text to a "
                ".txt under the session's parsed_pdfs/ folder (parsed_txt_path) — "
                "then search_file that .txt."
            ),
            "workspace_path": resolved_path,
        }

    looks_fasta = _strip_gz(resolved_path.lower()).endswith(_FASTA_EXTS)

    # Fetch one byte before start_byte so a mid-line start can be skipped
    # to the next line (same trick as read_file_preview).
    fetch_from = start_byte - 1 if start_byte > 0 else 0

    try:
        info, stream = await open_workspace_stream(
            resolved_path, token, config, start_byte=fetch_from, max_bytes=None,
        )
    except Exception as e:
        if _is_object_not_found(e):
            return _path_not_found_result(resolved_path)
        return {
            "error": f"File open failed: {type(e).__name__}: {e}",
            "errorType": "API_ERROR",
            "path": resolved_path,
            "source": "bvbrc-workspace",
        }

    if info.detected_format and info.detected_format != "gzip":
        return {
            "error": "binary file",
            "errorType": "BINARY_FILE",
            "hint": f"Detected format: {info.detected_format}. search_file only searches text.",
            "workspace_path": resolved_path,
            "total_size": info.total_size,
        }

    matches: List[Dict[str, Any]] = []
    before: deque = deque(maxlen=context_lines) if context_lines else deque(maxlen=0)
    awaiting_after: List[tuple[Dict[str, Any], int]] = []  # (match, lines still wanted)
    # FASTA: header matches whose record_length is still accumulating
    open_record: Optional[Dict[str, Any]] = None
    in_fasta_record_len = 0
    line_no = 0                 # counted from start_byte (exact when start_byte == 0)
    hit_cap = False
    resume_at: Optional[int] = None
    last_line_end = start_byte
    skip_first = start_byte > 0

    try:
        async for rel_off, raw in iter_lines(stream):
            abs_off = fetch_from + rel_off
            if skip_first:
                # The line containing the probe byte: either the tail of the
                # previous line (probe byte was '\n' → empty line) or a
                # partial line that began before start_byte.  Skip it.
                skip_first = False
                last_line_end = abs_off + len(raw) + 1
                continue

            line_no += 1
            text = _decode(raw)
            line_end = abs_off + len(raw) + 1
            last_line_end = line_end

            # Close an open FASTA record on the next header
            if looks_fasta:
                if text.startswith(">"):
                    if open_record is not None:
                        open_record["record_length"] = in_fasta_record_len
                        open_record = None
                    in_fasta_record_len = 0
                elif open_record is not None:
                    in_fasta_record_len += len(text.strip())

            # Feed "after" context to earlier matches
            if awaiting_after:
                still: List[tuple[Dict[str, Any], int]] = []
                for m, want in awaiting_after:
                    m["after"].append(_clip(text)[0])
                    if want - 1 > 0:
                        still.append((m, want - 1))
                awaiting_after = still

            if hit_cap:
                # Only draining after-context / FASTA record length now
                if not awaiting_after and open_record is None:
                    break
                before.append(text)
                continue

            if compiled.search(text):
                clipped, was_clipped = _clip(text)
                m: Dict[str, Any] = {
                    "line_number": line_no,
                    "byte_offset": abs_off,
                    "line": clipped,
                }
                if was_clipped:
                    m["line_clipped"] = True
                if context_lines:
                    m["before"] = list(before)
                    m["after"] = []
                    awaiting_after.append((m, context_lines))
                if looks_fasta and text.startswith(">"):
                    open_record = m
                    in_fasta_record_len = 0
                matches.append(m)
                if len(matches) >= max_matches:
                    hit_cap = True
                    resume_at = line_end
            before.append(text)
    except Exception as e:
        if _is_object_not_found(e):
            return _path_not_found_result(resolved_path)
        return {
            "error": f"Search failed: {type(e).__name__}: {e}",
            "errorType": "API_ERROR",
            "path": resolved_path,
            "source": "bvbrc-workspace",
        }
    finally:
        # Stopping early leaves the generator suspended on an open HTTP
        # response; close it now rather than waiting for GC.
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            await aclose()

    if open_record is not None:
        open_record["record_length"] = in_fasta_record_len
        if not info.eof:
            open_record["record_length_partial"] = True

    truncated = hit_cap
    scanned_bytes = info.bytes_consumed
    if info.partial and not hit_cap:
        resume_at = last_line_end

    # Keep the serialized result inside the loop's char cap by dropping
    # trailing matches (and pointing next_start at the first one dropped).
    def _size() -> int:
        return len(json.dumps(matches, indent=2, default=str))

    while matches and _size() > RESULT_CHAR_BUDGET:
        dropped = matches.pop()
        truncated = True
        resume_at = dropped["byte_offset"]

    result: Dict[str, Any] = {
        "matches": matches,
        "match_count": len(matches),
        "truncated": truncated,
        "scanned_bytes": scanned_bytes,
        "partial": bool(info.partial),
        "gzip": info.gzip,
        "offset_space": "decompressed" if info.gzip else "bytes",
        "total_size": info.total_size,
        "workspace_path": resolved_path,
        "pattern": pattern,
        "regex": regex,
    }
    if start_byte > 0:
        result["line_numbers_relative_to"] = start_byte
    if truncated or info.partial:
        result["next_start"] = resume_at
        if info.partial and not hit_cap:
            result["next_step"] = (
                f"Scanned {SCAN_CAP_BYTES // (1024 * 1024)} MB without reaching the end; "
                f"call search_file again with start_byte={resume_at} to continue."
            )
        else:
            result["next_step"] = (
                f"More matches may follow; call search_file again with start_byte={resume_at}."
            )
    elif matches:
        result["next_step"] = (
            "Use read_file_preview(path, start_byte=<byte_offset>) to read from a match."
        )
    else:
        result["next_step"] = "No matches. Check spelling/case or try regex=true."

    result["_summary"] = {
        "path": resolved_path,
        "pattern": pattern,
        "match_count": len(matches),
        "truncated": truncated,
        "partial": bool(info.partial),
        "first_offsets": [m["byte_offset"] for m in matches[:10]],
    }
    return result
