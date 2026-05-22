"""
Tool wrappers for service planning operations.

Thin async wrappers that translate agent tool-call arguments into the
MCP tool functions in service_agent.mcp_tools.agent_service_tools.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from service_agent.mcp_tools.agent_service_tools import (
    list_services_impl,
    get_service_schema_impl,
    plan_service_impl,
)
from service_agent.models import AgentConfig


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
    return await get_service_schema_impl(service_name=service_name, config=config)


async def plan_service(
    service_name: str,
    params: Optional[dict] = None,
    config: Optional[AgentConfig] = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Validate and plan a single BV-BRC service job.

    Extracts user_id from the auth token if available.

    Handles a common LLM mistake: when the model flattens the ``params``
    dict into top-level keyword arguments (e.g. ``plan_service(service_name=
    "genome_assembly", srr_ids=[...])`` instead of wrapping them in a
    ``params`` dict).  If *params* is not provided but extra kwargs are
    present, they are collected into *params* automatically.
    """
    # --- Auto-fix: collect flattened service params into params dict ---
    _meta_keys = {"config", "headers"}
    extra = {k: v for k, v in kwargs.items() if k not in _meta_keys}
    if params is None:
        # params was entirely missing -- collect all extra kwargs
        params = extra
    elif extra:
        # params exists, but LLM also passed some params at the top level;
        # merge them in without overwriting explicit params values.
        for k, v in extra.items():
            params.setdefault(k, v)

    if not params:
        return {
            "error": "No service parameters provided. Pass a 'params' dict "
                     "with the required service parameters.",
            "tool": "plan_service",
        }

    # Extract user_id from auth token
    user_id = "anonymous"
    if headers and "Authorization" in headers:
        token = headers["Authorization"]
        try:
            for part in token.split("|"):
                if part.startswith("un="):
                    user_id = part[3:]
                    break
        except Exception:
            pass

    return await plan_service_impl(
        service_name=service_name,
        params=params,
        user_id=user_id,
        config=config,
    )
