"""
Agent-specific service tool implementations.

These functions bridge the agent's tool-call interface to the MCP server's
service_validation_functions. Identical to v1 -- the MCP layer is unchanged.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional
from types import ModuleType

from service_agent.models import AgentConfig


# ---------------------------------------------------------------------------
# MCP server import helper (duplicated to avoid circular imports)
# ---------------------------------------------------------------------------

_validation_functions: Optional[ModuleType] = None
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


def _get_validation_functions(config: AgentConfig | None = None) -> ModuleType:
    """Import and return the service_validation_functions module."""
    global _validation_functions
    if _validation_functions is None:
        _ensure_path(config)
        from functions import service_validation_functions
        _validation_functions = service_validation_functions
    return _validation_functions


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def list_services_impl(
    config: AgentConfig | None = None,
) -> Dict[str, Any]:
    """List all available BV-BRC services."""
    vf = _get_validation_functions(config)
    return vf.list_services_fn()


async def get_service_schema_impl(
    service_name: str,
    config: AgentConfig | None = None,
) -> Dict[str, Any]:
    """Get the full parameter schema for a specific service."""
    vf = _get_validation_functions(config)
    return vf.get_service_schema_fn(service_name)


async def plan_service_impl(
    service_name: str,
    params: dict,
    user_id: str = "anonymous",
    config: AgentConfig | None = None,
) -> Dict[str, Any]:
    """
    Validate and plan a single service job.

    This is the core tool: validates parameters against the service schema,
    applies defaults, fuzzy-matches enums, and returns the validated parameter
    set ready for submission.
    """
    vf = _get_validation_functions(config)
    return vf.validate_service_params(service_name, params, user_id)
