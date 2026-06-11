"""
Tool implementations for service introspection (read-only).

Thin async wrappers that translate agent tool-call arguments into the
MCP server's service_validation_functions module.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from helpdesk_agent.mcp_tools.agent_helpdesk_tools import (
    list_services_impl,
    get_service_schema_impl,
)
from helpdesk_agent.models import AgentConfig


async def list_services(
    config: Optional[AgentConfig] = None,
    **kwargs,
) -> Dict[str, Any]:
    """List all available BV-BRC services with descriptions."""
    return await list_services_impl(config=config)


async def get_service_schema(
    service_name: str,
    config: Optional[AgentConfig] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Get the full parameter schema for a specific service."""
    return await get_service_schema_impl(
        service_name=service_name,
        config=config,
    )
