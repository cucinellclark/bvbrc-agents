"""Unified agent chat tool -- MCP entry point for all agent LLM loops.

Exposes a single ``agent_chat`` MCP tool that dispatches to one of three
agent back-ends (data, service, workspace) based on the ``agent_type``
parameter.  Each agent's ``run_agent()`` function is imported lazily so
the MCP server can start even if an individual agent has unmet deps.

This bridges the MCP server (external interface) to the agent modules
(internal LLM loops with tool calling).
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from fastmcp import Context, FastMCP

# ---------------------------------------------------------------------------
# sys.path setup -- make the three agent packages importable.
#
# Layout relative to this file:
#   bvbrc-agents/
#     ├── mcp_server/tools/agent_chat_tool.py  <-- HERE
#     ├── agents/data_agent/
#     ├── agents/service_agent/
#     ├── agents/workspace_agent/
#     └── config/llm_config.py
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

for _subdir in ("agents", "config"):
    _path = str(_REPO_ROOT / _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _resolve_auth_token(token_provider, token: Optional[str]) -> Optional[str]:
    """Resolve auth token: HTTP header > provided param > none."""
    if token_provider:
        return token_provider.get_token(token)
    return token or None


def _parse_context(context: Optional[str]) -> dict[str, Any]:
    """Parse context JSON string into a dict."""
    if not context:
        return {}
    try:
        return json.loads(context)
    except (json.JSONDecodeError, TypeError):
        return {"raw_context": context}


def _build_config_kwargs(
    auth_token: Optional[str],
    ctx: dict[str, Any],
) -> dict[str, Any]:
    """Build config kwargs from auth token and LLM override in context."""
    config_kwargs: dict[str, Any] = {"bvbrc_auth_token": auth_token}
    llm_override = ctx.pop("llm_override", None)
    if llm_override and isinstance(llm_override, dict):
        if llm_override.get("base_url"):
            config_kwargs["llm_base_url"] = llm_override["base_url"]
        if llm_override.get("api_key"):
            config_kwargs["llm_api_key"] = llm_override["api_key"]
        if llm_override.get("model"):
            config_kwargs["llm_model"] = llm_override["model"]
    return config_kwargs


def _build_tool_trace(result: Any) -> list[dict[str, Any]]:
    """Extract a serializable tool trace from an agent result."""
    return [
        {
            "tool": ex.tool_call.name,
            "arguments": ex.tool_call.arguments,
            "error": ex.error,
            "duration_ms": ex.duration_ms,
        }
        for ex in result.tool_trace
    ]


def _error_response(message: str) -> Dict[str, Any]:
    """Build a standard error response dict."""
    return {
        "answer": message,
        "status": "error",
        "sources": [],
        "iterations_used": 0,
        "elapsed_seconds": 0.0,
        "tool_trace": [],
    }


# ---------------------------------------------------------------------------
# Per-agent dispatch helpers
# ---------------------------------------------------------------------------

async def _run_data_agent(
    query: str,
    config_kwargs: dict[str, Any],
    ctx: dict[str, Any],
    progress_callback,
) -> Dict[str, Any]:
    """Import and run the data agent, returning a response dict."""
    from data_agent.agent import run_agent
    from data_agent.models import AgentConfig

    config = AgentConfig(**config_kwargs)
    result = await run_agent(
        query=query, config=config, context=ctx,
        progress_callback=progress_callback,
    )

    return {
        "answer": result.answer,
        "status": result.status,
        "sources": result.sources,
        "iterations_used": result.iterations_used,
        "elapsed_seconds": result.elapsed_seconds,
        "tool_trace": _build_tool_trace(result),
    }


async def _run_service_agent(
    query: str,
    config_kwargs: dict[str, Any],
    ctx: dict[str, Any],
    progress_callback,
) -> Dict[str, Any]:
    """Import and run the service agent, returning a response dict."""
    from service_agent.agent import run_agent
    from service_agent.models import AgentConfig

    config = AgentConfig(**config_kwargs)
    result = await run_agent(
        query=query, config=config, context=ctx,
        progress_callback=progress_callback,
    )

    tool_trace = _build_tool_trace(result)

    # Service2: build answer from structured result
    answer = result.pretty()
    if result.status == "needs_input" and result.question:
        answer = result.question
    elif result.status == "error" and result.error_message:
        answer = result.error_message

    response: Dict[str, Any] = {
        "answer": answer,
        "status": result.status,
        "sources": result.sources,
        "elapsed_seconds": result.elapsed_seconds,
        "tool_trace": tool_trace,
    }
    if result.manifest:
        response["manifest"] = result.manifest
    if result.workflow_plan:
        response["workflow_plan"] = result.workflow_plan
    if result.question:
        response["question"] = result.question
    # Workflow engine persistence metadata
    if result.workflow_id:
        response["workflow_id"] = result.workflow_id
    response["persisted"] = result.persisted
    return response


async def _run_workspace_agent(
    query: str,
    config_kwargs: dict[str, Any],
    ctx: dict[str, Any],
    progress_callback,
) -> Dict[str, Any]:
    """Import and run the workspace agent, returning a response dict."""
    from workspace_agent.agent import run_agent
    from workspace_agent.models import AgentConfig

    config = AgentConfig(**config_kwargs)
    result = await run_agent(
        query=query, config=config, context=ctx,
        progress_callback=progress_callback,
    )

    tool_trace = _build_tool_trace(result)

    return {
        "answer": result.answer,
        "status": result.status,
        # Keep fields aligned with other agent_chat responses
        "sources": [],
        "iterations_used": result.iterations_used,
        "elapsed_seconds": result.elapsed_seconds,
        "tool_trace": tool_trace,
        # Workspace agent structured payload for rich UI rendering
        "items": getattr(result, "items", []),
        "metadata": getattr(result, "metadata", []),
        "ui_grids": getattr(result, "ui_grids", []),
        "previews": getattr(result, "previews", []),
        "paths_explored": getattr(result, "paths_explored", []),
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_agent_chat_tool(
    mcp: FastMCP,
    token_provider=None,
) -> None:
    """Register the unified ``agent_chat`` tool on the MCP server.

    Args:
        mcp: FastMCP server instance.
        token_provider: TokenProvider for extracting auth from HTTP headers.
    """

    @mcp.tool()
    async def agent_chat(
        query: str,
        agent_type: str = "data",
        context: Optional[str] = None,
        token: Optional[str] = None,
        mcp_ctx: Context = None,
    ) -> Dict[str, Any]:
        """Chat with a BV-BRC agent using natural language.

        Runs the specified agent's full LLM loop: the agent's internal LLM
        will analyze the query, call whatever internal tools it needs (data
        searches, workspace operations, service planning, etc.), and return
        a complete answer.

        This is the primary entry point for the orchestrator to delegate
        a user's question to an agent.

        Args:
            query: Natural language question or request for the agent.
            agent_type: Which agent to use: "data", "service", or "workspace".
            context: Optional JSON string with additional context
                     (conversation history, workspace items, etc.).
            token: Optional auth token override.

        Returns:
            Dict with keys:
              - answer: The agent's natural language response.
              - status: "completed", "max_iterations", "needs_input", or "error".
              - sources: List of BV-BRC collections/services referenced.
              - iterations_used: Number of LLM iterations used.
              - elapsed_seconds: Wall-clock time for the agent run.
              - tool_trace: List of tool calls the agent made internally.

            Additional keys by agent_type:
              - service: manifest, workflow_plan, question, workflow_id, persisted
              - workspace: items, metadata, ui_grids, previews, paths_explored
        """
        auth_token = _resolve_auth_token(token_provider, token)
        ctx = _parse_context(context)
        config_kwargs = _build_config_kwargs(auth_token, ctx)

        # Build progress callback that sends MCP progress notifications
        async def progress_callback(progress, total, message):
            if mcp_ctx is not None:
                await mcp_ctx.report_progress(progress, total, message)

        try:
            if agent_type == "data":
                return await _run_data_agent(query, config_kwargs, ctx, progress_callback)
            elif agent_type == "service":
                return await _run_service_agent(query, config_kwargs, ctx, progress_callback)
            elif agent_type == "workspace":
                return await _run_workspace_agent(query, config_kwargs, ctx, progress_callback)
            else:
                return _error_response(f"Unknown agent type: {agent_type}")

        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            return _error_response(f"Agent error: {e}")
