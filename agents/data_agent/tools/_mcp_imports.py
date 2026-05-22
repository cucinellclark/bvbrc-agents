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

from data_agent.models import AgentConfig

_data_functions: Optional[ModuleType] = None
_group_functions: Optional[ModuleType] = None
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


def get_data_functions() -> ModuleType:
    """Import and return the MCP server's data_functions module."""
    global _data_functions
    if _data_functions is None:
        _ensure_path()
        from functions import data_functions
        _data_functions = data_functions
    return _data_functions


def get_group_functions() -> ModuleType:
    """Import and return the MCP server's group_functions module."""
    global _group_functions
    if _group_functions is None:
        _ensure_path()
        from functions import group_functions
        _group_functions = group_functions
    return _group_functions
