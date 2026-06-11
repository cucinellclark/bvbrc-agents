"""
Agent-specific helpdesk tool implementations.

These functions bridge the agent's tool-call interface to the MCP server's
rag_database_functions and service_validation_functions modules.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, Dict, Optional

from helpdesk_agent.models import AgentConfig


# ---------------------------------------------------------------------------
# MCP server import helpers
# ---------------------------------------------------------------------------

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


def _get_rag_functions(config: AgentConfig | None = None) -> ModuleType:
    """Import and return the rag_database_functions module."""
    global _rag_functions
    if _rag_functions is None:
        _ensure_path(config)
        from functions import rag_database_functions
        _rag_functions = rag_database_functions
    return _rag_functions


def _get_service_validation_functions(config: AgentConfig | None = None) -> ModuleType:
    """Import and return the service_validation_functions module."""
    global _service_validation_functions
    if _service_validation_functions is None:
        _ensure_path(config)
        from functions import service_validation_functions
        _service_validation_functions = service_validation_functions
    return _service_validation_functions


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def query_helpdesk_impl(
    query: str,
    top_k: int = 5,
    config: AgentConfig | None = None,
) -> Dict[str, Any]:
    """Query the helpdesk RAG database and return raw documents (no summarization).

    The agent's own LLM will synthesize the answer from the raw documents.
    """
    rag = _get_rag_functions(config)

    # Read the RAG config from the MCP server's config if available
    rag_config = {}
    try:
        cfg = config or AgentConfig()
        mcp_config_path = cfg.mcp_server_path + "/config/config.json"
        import json
        with open(mcp_config_path, "r") as f:
            mcp_config = json.load(f)
        rag_config = mcp_config.get("rag_database", {})
    except Exception:
        pass

    return rag.query_rag_helpdesk_func(
        query=query,
        top_k=top_k,
        config=rag_config,
        summarize=False,
    )


async def list_services_impl(
    config: AgentConfig | None = None,
) -> Dict[str, Any]:
    """List all available BV-BRC services."""
    vf = _get_service_validation_functions(config)
    return vf.list_services_fn()


async def get_service_schema_impl(
    service_name: str,
    config: AgentConfig | None = None,
) -> Dict[str, Any]:
    """Get the full parameter schema for a specific service."""
    vf = _get_service_validation_functions(config)
    return vf.get_service_schema_fn(service_name)
