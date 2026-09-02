"""
Canonical lazy-import helper for MCP server modules.

Centralizes sys.path manipulation and module caching so every agent uses
the same bridge to the MCP server's ``functions/`` and ``common/`` packages.

Usage::

    from shared.tools._mcp_imports import get_workspace_functions, get_json_rpc

All getters accept an optional ``mcp_server_path`` string.  When omitted
the path is resolved from the standard layout::

    <agents-root>/mcp_server/

The very first call that provides (or resolves) a path wins; subsequent
calls reuse the cached modules regardless of the path argument.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

_path_added: bool = False

# Cached module references
_workspace_functions: Optional[ModuleType] = None
_data_functions: Optional[ModuleType] = None
_group_functions: Optional[ModuleType] = None
_sra_functions: Optional[ModuleType] = None
_json_rpc: Optional[ModuleType] = None
_gowe_client: Optional[ModuleType] = None
_service_functions: Optional[ModuleType] = None


def _default_mcp_path() -> str:
    """Derive the MCP server path from the standard repo layout."""
    return str(Path(__file__).resolve().parent.parent.parent / "mcp_server")


def _ensure_path(mcp_server_path: Optional[str] = None) -> None:
    """Add the MCP server root to ``sys.path`` if not already present."""
    global _path_added
    if _path_added:
        return

    mcp_path = mcp_server_path or _default_mcp_path()
    if mcp_path and mcp_path not in sys.path:
        sys.path.insert(0, mcp_path)

    # bvbrc_solr_api is an editable install in mcp_env but needs to be
    # importable when running in the orchestrator venv too.
    solr_api_path = str(Path(mcp_path) / "bvbrc-python-api")
    if os.path.isdir(solr_api_path) and solr_api_path not in sys.path:
        sys.path.insert(0, solr_api_path)

    _path_added = True


# ---------------------------------------------------------------------------
# Public getters — one per MCP server module
# ---------------------------------------------------------------------------


def get_workspace_functions(mcp_server_path: Optional[str] = None) -> ModuleType:
    """Import and return ``mcp_server.functions.workspace_functions``."""
    global _workspace_functions
    if _workspace_functions is None:
        _ensure_path(mcp_server_path)
        from functions import workspace_functions  # type: ignore[import-untyped]

        _workspace_functions = workspace_functions
    return _workspace_functions


def get_data_functions(mcp_server_path: Optional[str] = None) -> ModuleType:
    """Import and return ``mcp_server.functions.data_functions``."""
    global _data_functions
    if _data_functions is None:
        _ensure_path(mcp_server_path)
        from functions import data_functions  # type: ignore[import-untyped]

        _data_functions = data_functions
    return _data_functions


def get_group_functions(mcp_server_path: Optional[str] = None) -> ModuleType:
    """Import and return ``mcp_server.functions.group_functions``."""
    global _group_functions
    if _group_functions is None:
        _ensure_path(mcp_server_path)
        from functions import group_functions  # type: ignore[import-untyped]

        _group_functions = group_functions
    return _group_functions


def get_sra_functions(mcp_server_path: Optional[str] = None) -> ModuleType:
    """Import and return ``mcp_server.functions.sra_functions``."""
    global _sra_functions
    if _sra_functions is None:
        _ensure_path(mcp_server_path)
        from functions import sra_functions  # type: ignore[import-untyped]

        _sra_functions = sra_functions
    return _sra_functions


def get_json_rpc(mcp_server_path: Optional[str] = None) -> ModuleType:
    """Import and return ``mcp_server.common.json_rpc``."""
    global _json_rpc
    if _json_rpc is None:
        _ensure_path(mcp_server_path)
        from common import json_rpc  # type: ignore[import-untyped]

        _json_rpc = json_rpc
    return _json_rpc


def get_gowe_client(mcp_server_path: Optional[str] = None) -> ModuleType:
    """Import and return ``mcp_server.common.gowe_client``."""
    global _gowe_client
    if _gowe_client is None:
        _ensure_path(mcp_server_path)
        from common import gowe_client  # type: ignore[import-untyped]

        _gowe_client = gowe_client
    return _gowe_client


def get_service_functions(mcp_server_path: Optional[str] = None) -> ModuleType:
    """Import and return ``mcp_server.functions.service_functions``."""
    global _service_functions
    if _service_functions is None:
        _ensure_path(mcp_server_path)
        from functions import service_functions  # type: ignore[import-untyped]

        _service_functions = service_functions
    return _service_functions
