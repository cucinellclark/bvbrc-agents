"""
Lazy import helper for MCP server modules.

Adds the Service MCP server path to sys.path on first use and caches the
imported modules. This keeps the sys.path manipulation in one place and
avoids import-time side effects.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Optional

from service_agent.models import AgentConfig

_service_validation_functions: Optional[ModuleType] = None
_workflow_composition_functions: Optional[ModuleType] = None
_workspace_functions: Optional[ModuleType] = None
_data_functions: Optional[ModuleType] = None
_group_functions: Optional[ModuleType] = None
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


def get_service_validation_functions(config: AgentConfig | None = None) -> ModuleType:
    """Import and return the MCP server's service_validation_functions module."""
    global _service_validation_functions
    if _service_validation_functions is None:
        _ensure_path(config)
        from functions import service_validation_functions
        _service_validation_functions = service_validation_functions
    return _service_validation_functions


def get_workspace_functions(config: AgentConfig | None = None) -> ModuleType:
    """Import and return the MCP server's workspace_functions module."""
    global _workspace_functions
    if _workspace_functions is None:
        _ensure_path(config)
        from functions import workspace_functions
        _workspace_functions = workspace_functions
    return _workspace_functions


def get_data_functions(config: AgentConfig | None = None) -> ModuleType:
    """Import and return the MCP server's data_functions module."""
    global _data_functions
    if _data_functions is None:
        _ensure_path(config)
        from functions import data_functions
        _data_functions = data_functions
    return _data_functions


def get_group_functions(config: AgentConfig | None = None) -> ModuleType:
    """Import and return the MCP server's group_functions module."""
    global _group_functions
    if _group_functions is None:
        _ensure_path(config)
        from functions import group_functions
        _group_functions = group_functions
    return _group_functions


def get_workflow_composition_functions(config: AgentConfig | None = None) -> ModuleType:
    """Import and return the MCP server's workflow_composition_functions module."""
    global _workflow_composition_functions
    if _workflow_composition_functions is None:
        _ensure_path(config)
        from functions import workflow_composition_functions
        _workflow_composition_functions = workflow_composition_functions
    return _workflow_composition_functions
