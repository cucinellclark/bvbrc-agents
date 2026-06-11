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

from helpdesk_agent.models import AgentConfig

_rag_functions: Optional[ModuleType] = None
_service_validation_functions: Optional[ModuleType] = None
_path_added: bool = False


def _ensure_path(config: AgentConfig | None = None) -> None:
    """Add the MCP server root to sys.path if not already present."""
    global _path_added
    if _path_added:
        return

    cfg = config or AgentConfig()
    mcp_path = cfg.mcp_server_path

    if mcp_path and mcp_path not in sys.path:
        sys.path.insert(0, mcp_path)
        _path_added = True


def get_rag_functions(config: AgentConfig | None = None) -> ModuleType:
    """Import and return the MCP server's rag_database_functions module."""
    global _rag_functions
    if _rag_functions is None:
        _ensure_path(config)
        from functions import rag_database_functions
        _rag_functions = rag_database_functions
    return _rag_functions


def get_service_validation_functions(config: AgentConfig | None = None) -> ModuleType:
    """Import and return the MCP server's service_validation_functions module."""
    global _service_validation_functions
    if _service_validation_functions is None:
        _ensure_path(config)
        from functions import service_validation_functions
        _service_validation_functions = service_validation_functions
    return _service_validation_functions
