"""FastAPI HTTP server exposing the orchestrator to the Copilot API gateway.

Phase 3 of the orchestrator implementation. This module provides:
  - POST /orchestrate       — Run orchestration, return final OrchestratorResponse
  - POST /orchestrate/stream — Run orchestration, stream Event objects as SSE
  - GET  /health            — Health check with agent status
  - GET  /agents            — Agent catalog for inspection

The existing Node.js Copilot API calls this Python service instead of
calling agents directly.

Usage:
    python -m orchestrator.server
    python -m orchestrator.server --port 9000 --config config/agents.yaml
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from orchestrator.config import OrchestratorConfig
from orchestrator.events.events import Event, EventType
from orchestrator.events.stream import collect_events
from orchestrator.llm.client import LLMClient
from orchestrator.llm.config import LLMConfig
from orchestrator.models import OrchestratorRequest, OrchestratorResponse
from orchestrator.orchestrate import orchestrate, orchestrate_to_response
from orchestrator.registry.agent_registry import AgentRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application state (set during lifespan)
# ---------------------------------------------------------------------------


class AppState:
    """Holds shared application state initialized at startup."""

    registry: AgentRegistry | None = None
    llm: LLMClient | None = None
    routing_llm: LLMClient | None = None  # Separate (faster) client for routing
    config: OrchestratorConfig | None = None
    ready: bool = False
    startup_time: float = 0.0

    # Cache of LLMClient instances keyed by (base_url, api_key, model).
    # Avoids creating a new AsyncOpenAI client (and TLS handshake) for
    # every request that uses an llm_override.
    llm_cache: dict[tuple[str, str, str], LLMClient] | None = None
    LLM_CACHE_MAX: int = 16


_state = AppState()


# ---------------------------------------------------------------------------
# Auth token resolution
# ---------------------------------------------------------------------------


def _load_default_token() -> str | None:
    """Load the default auth token from auth_token.txt or environment."""
    token = os.environ.get("BV_BRC_AUTH_TOKEN")
    if token:
        return token

    token_file = Path(__file__).parent.parent / "auth_token.txt"
    if token_file.exists():
        return token_file.read_text().strip() or None

    return None


def _resolve_auth_token(
    body_token: str | None,
    authorization: str | None,
) -> str | None:
    """Resolve auth token from multiple sources.

    Priority: request body > Authorization header > default token.
    """
    # 1. Explicit token in request body
    if body_token:
        return body_token

    # 2. Authorization header (Bearer <token>)
    if authorization:
        if authorization.startswith("Bearer "):
            return authorization[7:]
        return authorization

    # 3. Default token from env or file
    return _load_default_token()


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


def _build_lifespan(
    config_path: str | None = None,
):
    """Build a lifespan context manager with the given config path."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """FastAPI lifespan: initialize registry and LLM on startup, clean up on shutdown."""
        startup_start = time.monotonic()

        # --- Load configuration ---
        if config_path:
            config = OrchestratorConfig.from_yaml(config_path)
        else:
            config = OrchestratorConfig.from_defaults()
        _state.config = config

        # --- Apply default auth token to agents that lack one ---
        default_token = _load_default_token()
        if default_token:
            for agent_config in config.agents.values():
                if not agent_config.auth_token:
                    agent_config.auth_token = default_token

        # --- Initialize LLM client cache ---
        _state.llm_cache = {}

        # --- Initialize LLM client ---
        llm_config = LLMConfig(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            timeout_seconds=config.llm_timeout_seconds,
        )
        _state.llm = LLMClient(llm_config)
        logger.info(f"LLM client initialized: {llm_config.model} @ {llm_config.base_url}")

        # --- Initialize routing LLM client (faster model) ---
        if config.routing_model and config.routing_model != config.llm_model:
            routing_config = LLMConfig(
                base_url=config.llm_base_url,
                api_key=config.llm_api_key,
                model=config.routing_model,
                temperature=0.0,
                max_tokens=512,  # Routing responses are small JSON
                timeout_seconds=config.llm_timeout_seconds,
            )
            _state.routing_llm = LLMClient(routing_config)
            logger.info(
                f"Routing LLM client initialized: {routing_config.model} "
                f"@ {routing_config.base_url}"
            )
        else:
            _state.routing_llm = None

        # --- Initialize registry and discover agents ---
        _state.registry = AgentRegistry(config)

        logger.info("Starting agent discovery...")
        discovery_results: list[str] = []
        async for event in _state.registry.discover_all():
            if event.type == EventType.DISCOVERY_AGENT:
                agent = event.data.get("agent", "?")
                tools = event.data.get("tool_count", 0)
                discovery_results.append(f"{agent}({tools} tools)")
                logger.info(f"Discovered agent: {agent} with {tools} tools")
            elif event.type == EventType.ORCHESTRATOR_ERROR:
                logger.error(f"Discovery error: {event.data.get('error', '?')}")
            elif event.type == EventType.DISCOVERY_DONE:
                healthy = event.data.get("healthy", 0)
                total = event.data.get("total", 0)
                logger.info(f"Discovery complete: {healthy}/{total} agents healthy")

        # Start periodic health checks
        await _state.registry.start_health_checks()

        elapsed = round((time.monotonic() - startup_start) * 1000, 1)
        _state.startup_time = time.time()
        _state.ready = True
        logger.info(
            f"Orchestrator ready in {elapsed}ms. "
            f"Agents: {', '.join(discovery_results) or 'none'}"
        )

        yield  # --- Server is running ---

        # --- Shutdown ---
        logger.info("Shutting down orchestrator...")
        _state.ready = False

        if _state.registry:
            await _state.registry.shutdown()
        # Close all cached LLM clients
        if _state.llm_cache:
            for cached_client in _state.llm_cache.values():
                await cached_client.close()
            _state.llm_cache.clear()
        if _state.routing_llm:
            await _state.routing_llm.close()
        if _state.llm:
            await _state.llm.close()

        logger.info("Orchestrator shut down.")

    return lifespan


# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------


def create_app(config_path: str | None = None) -> FastAPI:
    """Create the FastAPI application.

    Args:
        config_path: Path to agents.yaml. If None, uses default location.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="BV-BRC Copilot Orchestrator",
        description=(
            "Multi-agent orchestrator for the BV-BRC Copilot. "
            "Routes user requests to specialized agents (Data, Service), "
            "executes them via MCP, and synthesizes unified responses."
        ),
        version="0.1.0",
        lifespan=_build_lifespan(config_path),
    )

    # --- Register routes ---

    def _resolve_llm(request: OrchestratorRequest) -> LLMClient:
        """Return an LLM client for this request.

        Uses the default singleton when no override is provided.
        For overrides, returns a cached client keyed by
        ``(base_url, api_key, model)`` to enable HTTP connection reuse
        and avoid repeated TLS handshakes with remote endpoints.
        """
        if request.llm_override:
            override = request.llm_override
            default_cfg = _state.llm.config
            needs_override = (
                (override.base_url and override.base_url != default_cfg.base_url)
                or (override.model and override.model != default_cfg.model)
                or (override.api_key and override.api_key != default_cfg.api_key)
            )
            if needs_override:
                cache_key = (
                    override.base_url or default_cfg.base_url,
                    override.api_key or default_cfg.api_key,
                    override.model or default_cfg.model,
                )
                cache = _state.llm_cache
                if cache is not None:
                    cached = cache.get(cache_key)
                    if cached is not None:
                        return cached

                override_config = LLMConfig(
                    base_url=cache_key[0],
                    api_key=cache_key[1],
                    model=cache_key[2],
                    temperature=default_cfg.temperature,
                    max_tokens=default_cfg.max_tokens,
                    timeout_seconds=default_cfg.timeout_seconds,
                )
                logger.info(
                    f"Creating cached LLM client: model={override_config.model!r} "
                    f"base_url={override_config.base_url!r}"
                )
                client = LLMClient(override_config)

                if cache is not None:
                    # Evict oldest entry if cache is full
                    if len(cache) >= _state.LLM_CACHE_MAX:
                        oldest_key = next(iter(cache))
                        cache.pop(oldest_key, None)
                    cache[cache_key] = client

                return client
        return _state.llm

    @app.post("/orchestrate", response_model=OrchestratorResponse)
    async def post_orchestrate(
        request: OrchestratorRequest,
        authorization: str | None = Header(default=None),
    ) -> OrchestratorResponse:
        """Run the orchestration loop and return the final response.

        Accepts an OrchestratorRequest JSON body. Auth token can be provided
        in the request body (auth_token field) or via Authorization header.
        """
        _require_ready()

        # Resolve auth token
        token = _resolve_auth_token(request.auth_token, authorization)
        request.auth_token = token

        llm = _resolve_llm(request)
        # When the request specifies an LLM override, use that model for
        # everything — including routing.  Only fall back to the dedicated
        # (cheaper/faster) routing model when no override is provided.
        routing_llm = None if request.llm_override else _state.routing_llm
        try:
            response = await orchestrate_to_response(
                request, _state.registry, llm,
                routing_llm=routing_llm,
            )
            return response
        except Exception as e:
            logger.error(f"Orchestration failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Orchestration error: {e}",
            )
        finally:
            pass  # Cached clients are reused; closed at shutdown

    @app.post("/orchestrate/stream")
    async def post_orchestrate_stream(
        request: OrchestratorRequest,
        authorization: str | None = Header(default=None),
    ) -> EventSourceResponse:
        """Run the orchestration loop and stream events as SSE.

        Each SSE event has:
          - event: <EventType value>  (e.g., "routing_decision", "agent_result")
          - data: <JSON payload>

        The final event is "orchestrator_done" with the complete response.
        """
        _require_ready()

        # Resolve auth token
        token = _resolve_auth_token(request.auth_token, authorization)
        request.auth_token = token

        llm = _resolve_llm(request)
        routing_llm = None if request.llm_override else _state.routing_llm

        async def event_generator() -> AsyncGenerator[dict[str, str], None]:
            try:
                async for event in orchestrate(
                    request, _state.registry, llm,
                    routing_llm=routing_llm,
                ):
                    yield {
                        "event": event.type.value,
                        "data": json.dumps(
                            {
                                "id": event.id,
                                "type": event.type.value,
                                "data": event.data,
                                "agent_name": event.agent_name,
                                "step_index": event.step_index,
                                "timestamp": event.timestamp,
                            },
                            default=str,
                        ),
                    }
            except Exception as e:
                logger.error(f"SSE stream error: {e}", exc_info=True)
                yield {
                    "event": "orchestrator_error",
                    "data": json.dumps({"error": str(e)}),
                }
            finally:
                pass  # Cached clients are reused; closed at shutdown

        return EventSourceResponse(event_generator())

    @app.get("/health")
    async def get_health() -> JSONResponse:
        """Health check endpoint.

        Returns server status, agent health, and readiness information.
        """
        agents_status: dict[str, Any] = {}

        if _state.registry:
            for key, agent in _state.registry.agents.items():
                agents_status[key] = {
                    "name": agent.name,
                    "endpoint": agent.endpoint,
                    "healthy": agent.is_healthy,
                    "connected": agent.is_connected,
                    "tool_count": len(agent.tools),
                    "latency_ms": round(agent._last_latency_ms, 1),
                }

        healthy_count = sum(
            1 for a in agents_status.values() if a.get("healthy")
        )
        total_count = len(agents_status)

        status_code = 200 if _state.ready and healthy_count > 0 else 503

        return JSONResponse(
            status_code=status_code,
            content={
                "status": "ok" if status_code == 200 else "degraded",
                "ready": _state.ready,
                "uptime_seconds": round(time.time() - _state.startup_time, 1)
                if _state.startup_time
                else 0,
                "agents": {
                    "total": total_count,
                    "healthy": healthy_count,
                    "details": agents_status,
                },
                "llm": {
                    "model": _state.config.llm_model if _state.config else None,
                    "base_url": _state.config.llm_base_url
                    if _state.config
                    else None,
                },
            },
        )

    @app.get("/agents")
    async def get_agents() -> JSONResponse:
        """Return the agent catalog.

        Includes agent descriptions, capabilities, tools, and health status.
        """
        _require_ready()

        catalog = _state.registry.catalog_structured()
        return JSONResponse(content={"agents": catalog})

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_ready() -> None:
    """Raise 503 if the orchestrator is not ready."""
    if not _state.ready or not _state.registry or not _state.llm:
        raise HTTPException(
            status_code=503,
            detail="Orchestrator is not ready. Agent discovery may still be in progress.",
        )


# ---------------------------------------------------------------------------
# Default app instance (for uvicorn orchestrator.server:app)
# ---------------------------------------------------------------------------


app = create_app()
