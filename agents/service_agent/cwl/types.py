"""CWL type mapping helpers for BV-BRC service parameters.

Pure functions for inferring CWL types from Python values and converting
Python values to CWL-typed input objects. Handles BV-BRC workspace path
wrapping (bare paths -> ws:/// URIs with CWL File/Directory objects).
"""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Workspace path detection and wrapping
# ---------------------------------------------------------------------------

# BV-BRC workspace paths start with / and contain @ (user identifier)
_WORKSPACE_PATH_RE = re.compile(r"^/[^/]*@(bvbrc|patricbrc\.org)/.+")

# Parameter names that are always output basenames (strings, not paths)
_OUTPUT_BASENAME_PARAMS = frozenset({"output_file"})

# Parameter names that are output destination paths — these are passed as
# plain strings to the BV-BRC app service (which writes results there).
# They must NOT be typed as CWL Directory objects, because that would cause
# GoWe's workspace stager to try to download them as inputs.
_OUTPUT_PATH_PARAMS = frozenset({"output_path"})

# Parameter names that are always directories (actual input directories
# that need to be staged in from the workspace)
_DIRECTORY_PARAMS = frozenset({"workspace_path"})


def is_workspace_path(value: str) -> bool:
    """Detect whether a string value is a BV-BRC workspace path.

    Workspace paths look like: /user@bvbrc/home/folder/file.fasta
    """
    return bool(_WORKSPACE_PATH_RE.match(value))


def wrap_workspace_path(path: str, as_type: str = "Directory") -> dict[str, str]:
    """Wrap a bare BV-BRC workspace path into a CWL typed object.

    Bare path:  /user@bvbrc/home/folder
    CWL object: {"class": "Directory", "location": "ws:///user@bvbrc/home/folder"}

    The ws:// prefix is "ws://" + bare_path (bare path already has leading /,
    producing ws:///).

    Args:
        path: Bare BV-BRC workspace path (e.g., "/user@bvbrc/home/folder").
        as_type: CWL class — "Directory" or "File".

    Returns:
        CWL typed object dict.
    """
    if as_type not in ("File", "Directory"):
        raise ValueError(f"as_type must be 'File' or 'Directory', got {as_type!r}")

    location = f"ws://{path}"
    return {"class": as_type, "location": location}


# ---------------------------------------------------------------------------
# CWL type inference
# ---------------------------------------------------------------------------


def infer_cwl_type(value: Any) -> str:
    """Infer a CWL type string from a Python value.

    Mapping:
        str   -> "string"  (or "File"/"Directory" if workspace path)
        int   -> "int"
        float -> "float"
        bool  -> "boolean"
        list  -> "<element_type>[]" (e.g., "string[]")
        dict  -> "File" or "Directory" if it has a "class" key, else "string"
        None  -> "null"

    Note: This performs best-effort inference from concrete values.
    For workspace paths, this returns "string" — use param_name context
    in python_to_cwl_value() for accurate File/Directory distinction.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        # bool check must come before int (bool is subclass of int)
        return "boolean"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        cls = value.get("class")
        if cls in ("File", "Directory"):
            return cls
        return "string"
    if isinstance(value, list):
        if not value:
            return "string[]"
        # Infer element type from first element
        elem_type = infer_cwl_type(value[0])
        return f"{elem_type}[]"

    return "string"


def infer_cwl_input_type(value: Any, param_name: str | None = None) -> str:
    """Infer CWL type for a tool input definition, using param name context.

    Unlike infer_cwl_type(), this uses parameter naming conventions:
    - output_path -> "Directory"
    - output_file -> "string"
    - Workspace paths in non-output params -> "File" or "Directory"
    """
    if param_name in _OUTPUT_BASENAME_PARAMS:
        return "string"
    if param_name in _OUTPUT_PATH_PARAMS:
        return "string"
    if param_name in _DIRECTORY_PARAMS:
        return "Directory"
    if isinstance(value, str) and is_workspace_path(value):
        # Heuristic: paths ending with a file extension are Files
        if _looks_like_file_path(value):
            return "File"
        return "Directory"
    return infer_cwl_type(value)


def _looks_like_file_path(path: str) -> bool:
    """Heuristic: does a workspace path look like a file (vs. directory)?

    Files typically have extensions like .fasta, .fastq, .gff, .genome, etc.
    Paths without extensions or ending with / are treated as directories.
    """
    # Strip trailing slash
    path = path.rstrip("/")
    # Get the last component
    last_part = path.rsplit("/", 1)[-1]
    # Check for a file extension (at least 2 chars after the dot)
    return bool(re.search(r"\.\w{2,}$", last_part))


# ---------------------------------------------------------------------------
# Python -> CWL value conversion
# ---------------------------------------------------------------------------


def python_to_cwl_value(value: Any, param_name: str | None = None) -> Any:
    """Convert a Python value to CWL input format.

    Handles workspace path wrapping:
    - output_path params -> Directory object with ws:/// URI
    - output_file params -> plain string (basename, not a path)
    - Other workspace paths -> File or Directory based on heuristic

    Scalar values (str, int, float, bool) pass through unchanged.
    Lists are recursively converted.
    Dicts with "class" key are assumed to already be CWL objects.

    Args:
        value: The Python value to convert.
        param_name: Parameter name for context-aware conversion.

    Returns:
        CWL-typed value suitable for submission inputs.
    """
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value

    if isinstance(value, str):
        # output_file is always a plain string basename
        if param_name in _OUTPUT_BASENAME_PARAMS:
            return value

        # output_path is a destination string — not a Directory to stage in
        if param_name in _OUTPUT_PATH_PARAMS:
            return value

        # Actual input directories that need staging
        if param_name in _DIRECTORY_PARAMS:
            if is_workspace_path(value):
                return wrap_workspace_path(value, as_type="Directory")
            return value

        # Other workspace paths: auto-detect File vs Directory
        if is_workspace_path(value):
            if _looks_like_file_path(value):
                return wrap_workspace_path(value, as_type="File")
            else:
                return wrap_workspace_path(value, as_type="Directory")

        return value

    if isinstance(value, dict):
        # Already a CWL object (has "class" key)
        if "class" in value:
            return value
        # Nested dict — recursively convert
        return {k: python_to_cwl_value(v, k) for k, v in value.items()}

    if isinstance(value, list):
        return [python_to_cwl_value(item, param_name) for item in value]

    # Fallback: convert to string
    return str(value)
