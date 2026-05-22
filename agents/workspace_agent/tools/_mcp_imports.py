"""
Lazy import helper for MCP server modules.

Adds the MCP server path to sys.path on first use and caches the imported
modules. This keeps the sys.path manipulation in one place and avoids
import-time side effects.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Optional

from workspace_agent.models import AgentConfig

_workspace_functions: Optional[ModuleType] = None
_json_rpc: Optional[ModuleType] = None
_path_added: bool = False


def _ensure_path() -> None:
    """Add the MCP server root to sys.path if not already present."""
    global _path_added
    if _path_added:
        return

    config = AgentConfig()
    mcp_path = config.mcp_server_path

    if mcp_path and mcp_path not in sys.path:
        sys.path.insert(0, mcp_path)
        _path_added = True


def get_workspace_functions() -> ModuleType:
    """Import and return the MCP server's workspace_functions module."""
    global _workspace_functions
    if _workspace_functions is None:
        _ensure_path()
        from functions import workspace_functions
        _workspace_functions = workspace_functions
    return _workspace_functions


def get_json_rpc() -> ModuleType:
    """Import and return the MCP server's json_rpc module."""
    global _json_rpc
    if _json_rpc is None:
        _ensure_path()
        from common import json_rpc
        _json_rpc = json_rpc
    return _json_rpc
