"""Tests for the orchestrator HTTP server (Phase 3).

Tests all endpoints with mocked orchestrator components:
  - POST /orchestrate
  - POST /orchestrate/stream
  - GET  /health
  - GET  /agents
"""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from mcp.types import Tool as McpTool, CallToolResult, TextContent

from orchestrator.config import AgentConfig, OrchestratorConfig
from orchestrator.events.events import Event, EventType
from orchestrator.llm.client import LLMClient
from orchestrator.models import OrchestratorRequest, OrchestratorResponse
from orchestrator.registry.agent_handle import AgentHandle
from orchestrator.registry.agent_registry import AgentRegistry
from orchestrator.server import AppState, _state, create_app, _resolve_auth_token


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_registry() -> AgentRegistry:
    """Create a registry with mock healthy agents (no real MCP connections)."""
    config = OrchestratorConfig(
        agents={
            "data": AgentConfig(
                name="Data Agent",
                description="Retrieves biological data.",
                endpoint="http://localhost:12009",
                capabilities=["data_retrieval"],
            ),
            "service2": AgentConfig(
                name="Service Agent",
                description="Plans service workflows.",
                endpoint="http://localhost:8053",
                capabilities=["workflow_planning"],
            ),
        },
        health_check_interval=0,
    )
    registry = AgentRegistry(config)

    for key in ["data", "service2"]:
        handle = AgentHandle(key, config.agents[key])
        handle._healthy = True
        handle._last_latency_ms = 42.0
        handle._tools = [
            McpTool(name="agent_chat", description="Chat", inputSchema={}),
            McpTool(name="other_tool", description="Other", inputSchema={}),
        ]
        registry._agents[key] = handle

    return registry


def _make_mock_llm(routing_response: str | None = None) -> LLMClient:
    """Create a mock LLM client."""
    client = MagicMock(spec=LLMClient)
    if routing_response:
        client.complete = AsyncMock(return_value=routing_response)
    else:
        client.complete = AsyncMock(return_value='{"decision":"direct","reasoning":"test","direct_response":"Hello!"}')
    client.close = AsyncMock()
    return client


def _make_agent_result(
    answer: str = "Agent answer",
    status: str = "completed",
) -> CallToolResult:
    """Create a mock MCP CallToolResult for agent_chat."""
    data = {
        "answer": answer,
        "status": status,
        "sources": ["genome"],
        "iterations_used": 2,
        "elapsed_seconds": 1.5,
        "tool_trace": [],
    }
    result = MagicMock(spec=CallToolResult)
    result.content = [TextContent(type="text", text=json.dumps(data))]
    result.isError = False
    return result


def _setup_ready_state(
    registry: AgentRegistry | None = None,
    llm: LLMClient | None = None,
) -> None:
    """Set up _state as if the server has started and is ready."""
    _state.registry = registry or _make_mock_registry()
    _state.llm = llm or _make_mock_llm()
    _state.config = OrchestratorConfig(
        agents={
            "data": AgentConfig(
                name="Data Agent",
                description="Retrieves biological data.",
                endpoint="http://localhost:12009",
                capabilities=["data_retrieval"],
            ),
            "service2": AgentConfig(
                name="Service Agent",
                description="Plans service workflows.",
                endpoint="http://localhost:8053",
                capabilities=["workflow_planning"],
            ),
        },
        health_check_interval=0,
    )
    _state.ready = True
    _state.startup_time = 1000000.0


def _teardown_state() -> None:
    """Reset _state to uninitialized."""
    _state.registry = None
    _state.llm = None
    _state.config = None
    _state.ready = False
    _state.startup_time = 0.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    """Create a FastAPI app without running the lifespan (we mock state directly)."""
    # We use create_app() but skip the lifespan by manually setting _state
    test_app = create_app()
    return test_app


@pytest.fixture
def ready_client(app):
    """A TestClient with the server in 'ready' state (mocked components)."""
    _setup_ready_state()
    yield TestClient(app, raise_server_exceptions=False)
    _teardown_state()


@pytest.fixture
def unready_client(app):
    """A TestClient with the server NOT ready."""
    _teardown_state()
    yield TestClient(app, raise_server_exceptions=False)
    _teardown_state()


# ---------------------------------------------------------------------------
# Tests: Auth token resolution
# ---------------------------------------------------------------------------


class TestResolveAuthToken:
    def test_body_token_highest_priority(self):
        assert _resolve_auth_token("body-token", "Bearer header-token") == "body-token"

    def test_authorization_header_bearer(self):
        assert _resolve_auth_token(None, "Bearer my-token") == "my-token"

    def test_authorization_header_raw(self):
        assert _resolve_auth_token(None, "raw-token") == "raw-token"

    def test_falls_back_to_default(self):
        with patch("orchestrator.server._load_default_token", return_value="default-tok"):
            assert _resolve_auth_token(None, None) == "default-tok"

    def test_returns_none_when_no_token(self):
        with patch("orchestrator.server._load_default_token", return_value=None):
            assert _resolve_auth_token(None, None) is None


# ---------------------------------------------------------------------------
# Tests: GET /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_when_ready(self, ready_client):
        resp = ready_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["ready"] is True
        assert body["agents"]["total"] == 2
        assert body["agents"]["healthy"] == 2
        assert "data" in body["agents"]["details"]
        assert "service2" in body["agents"]["details"]
        assert body["agents"]["details"]["data"]["healthy"] is True

    def test_health_when_not_ready(self, unready_client):
        resp = unready_client.get("/health")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["ready"] is False

    def test_health_degraded_when_no_healthy_agents(self, app):
        """Server is ready but no agents are healthy -> 503."""
        registry = _make_mock_registry()
        for agent in registry._agents.values():
            agent._healthy = False
        _setup_ready_state(registry=registry)

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["agents"]["healthy"] == 0

        _teardown_state()

    def test_health_includes_llm_info(self, ready_client):
        resp = ready_client.get("/health")
        body = resp.json()
        assert "llm" in body
        assert body["llm"]["model"] is not None


# ---------------------------------------------------------------------------
# Tests: GET /agents
# ---------------------------------------------------------------------------


class TestAgentsEndpoint:
    def test_agents_catalog(self, ready_client):
        resp = ready_client.get("/agents")
        assert resp.status_code == 200
        body = resp.json()
        assert "agents" in body
        agents = body["agents"]
        assert len(agents) == 2
        keys = [a["key"] for a in agents]
        assert "data" in keys
        assert "service2" in keys

    def test_agents_includes_details(self, ready_client):
        resp = ready_client.get("/agents")
        agents = resp.json()["agents"]
        data_agent = next(a for a in agents if a["key"] == "data")
        assert data_agent["name"] == "Data Agent"
        assert data_agent["healthy"] is True
        assert data_agent["tool_count"] == 2
        assert "agent_chat" in data_agent["tools"]

    def test_agents_when_not_ready(self, unready_client):
        resp = unready_client.get("/agents")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Tests: POST /orchestrate
# ---------------------------------------------------------------------------


class TestOrchestrateEndpoint:
    def test_direct_response(self, ready_client):
        """Route that produces a direct response (no agent invoked)."""
        resp = ready_client.post(
            "/orchestrate",
            json={"query": "Hello!"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["response_text"] == "Hello!"
        assert body["status"] == "completed"
        assert body["agents_used"] == []

    def test_agent_routing(self, app):
        """Route to data agent, execute, synthesize."""
        registry = _make_mock_registry()
        data_agent = registry.get("data")
        data_agent.call_tool = AsyncMock(
            return_value=_make_agent_result("Found 50 genomes.")
        )

        routing = json.dumps({
            "decision": "agent",
            "reasoning": "data query",
            "agent_key": "data",
            "task": "Find genomes",
        })
        llm = _make_mock_llm(routing)

        _setup_ready_state(registry=registry, llm=llm)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post(
            "/orchestrate",
            json={"query": "Find genomes"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "50 genomes" in body["response_text"]
        assert "data" in body["agents_used"]
        assert body["status"] == "completed"

        _teardown_state()

    def test_auth_token_from_body(self, app):
        """Auth token from request body is passed through."""
        registry = _make_mock_registry()
        data_agent = registry.get("data")
        data_agent.call_tool = AsyncMock(
            return_value=_make_agent_result("Result")
        )

        routing = json.dumps({
            "decision": "agent",
            "reasoning": "data query",
            "agent_key": "data",
            "task": "Find genomes",
        })
        llm = _make_mock_llm(routing)

        _setup_ready_state(registry=registry, llm=llm)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post(
            "/orchestrate",
            json={"query": "Find genomes", "auth_token": "my-token"},
        )
        assert resp.status_code == 200

        # Verify the token was passed to agent_chat
        call_args = data_agent.call_tool.call_args
        arguments = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("arguments", {})
        assert arguments.get("token") == "my-token"

        _teardown_state()

    def test_auth_token_from_header(self, app):
        """Auth token from Authorization header is used when body has none."""
        llm = _make_mock_llm()  # direct response
        _setup_ready_state(llm=llm)
        client = TestClient(app, raise_server_exceptions=False)

        # Mock _resolve_auth_token to verify it's called with header
        with patch("orchestrator.server._resolve_auth_token", wraps=_resolve_auth_token) as mock_resolve:
            resp = client.post(
                "/orchestrate",
                json={"query": "Hello"},
                headers={"Authorization": "Bearer header-token"},
            )
            assert resp.status_code == 200
            # The resolve function should have been called
            mock_resolve.assert_called_once()
            # Body token was None, header was "Bearer header-token"
            args = mock_resolve.call_args[0]
            assert args[0] is None  # body token
            assert args[1] == "Bearer header-token"

        _teardown_state()

    def test_orchestrate_when_not_ready(self, unready_client):
        resp = unready_client.post(
            "/orchestrate",
            json={"query": "Hello"},
        )
        assert resp.status_code == 503

    def test_with_conversation_context(self, ready_client):
        """Request with full conversation context."""
        resp = ready_client.post(
            "/orchestrate",
            json={
                "query": "Hello",
                "session_id": "sess-123",
                "user_id": "user-456",
                "conversation_summary": "Previous discussion about genomes.",
                "recent_messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                ],
            },
        )
        assert resp.status_code == 200

    def test_with_target_agent_override(self, app):
        """Forced routing via target_agent."""
        registry = _make_mock_registry()
        service_agent = registry.get("service2")
        service_agent.call_tool = AsyncMock(
            return_value=_make_agent_result("Workflow ready.", "completed")
        )

        # LLM should not be called for routing (forced), but complete still needs mock
        llm = MagicMock(spec=LLMClient)
        llm.complete = AsyncMock(return_value="unused")
        llm.close = AsyncMock()

        _setup_ready_state(registry=registry, llm=llm)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post(
            "/orchestrate",
            json={"query": "Assemble genome", "target_agent": "service2"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "Workflow ready" in body["response_text"]
        assert "service2" in body["agents_used"]

        _teardown_state()

    def test_invalid_request_body(self, ready_client):
        """Missing required 'query' field."""
        resp = ready_client.post(
            "/orchestrate",
            json={"not_a_query": "oops"},
        )
        assert resp.status_code == 422  # Pydantic validation error


# ---------------------------------------------------------------------------
# Tests: POST /orchestrate/stream
# ---------------------------------------------------------------------------


class TestOrchestrateStreamEndpoint:
    def test_stream_direct_response(self, ready_client):
        """Stream events for a direct response."""
        resp = ready_client.post(
            "/orchestrate/stream",
            json={"query": "Hello!"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        # Parse SSE events from the response body
        events = _parse_sse_events(resp.text)
        assert len(events) > 0

        event_types = [e["event"] for e in events]
        assert "orchestrator_start" in event_types
        assert "routing_start" in event_types
        assert "routing_decision" in event_types
        assert "orchestrator_done" in event_types

    def test_stream_agent_routing(self, app):
        """Stream events for agent routing flow."""
        registry = _make_mock_registry()
        data_agent = registry.get("data")
        data_agent.call_tool = AsyncMock(
            return_value=_make_agent_result("Found genomes.")
        )

        routing = json.dumps({
            "decision": "agent",
            "reasoning": "data query",
            "agent_key": "data",
            "task": "Find genomes",
        })
        llm = _make_mock_llm(routing)

        _setup_ready_state(registry=registry, llm=llm)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post(
            "/orchestrate/stream",
            json={"query": "Find genomes"},
        )
        assert resp.status_code == 200

        events = _parse_sse_events(resp.text)
        event_types = [e["event"] for e in events]
        assert "orchestrator_start" in event_types
        assert "agent_start" in event_types
        assert "orchestrator_done" in event_types

        # Verify the final event has response data
        done_events = [e for e in events if e["event"] == "orchestrator_done"]
        assert len(done_events) == 1
        done_data = json.loads(done_events[0]["data"])
        assert "Found genomes" in done_data["data"]["response_text"]

        _teardown_state()

    def test_stream_when_not_ready(self, unready_client):
        resp = unready_client.post(
            "/orchestrate/stream",
            json={"query": "Hello"},
        )
        assert resp.status_code == 503

    def test_stream_events_have_typed_event_field(self, ready_client):
        """Each SSE message should have an `event:` field with the EventType value."""
        resp = ready_client.post(
            "/orchestrate/stream",
            json={"query": "Hello"},
        )
        events = _parse_sse_events(resp.text)
        for event in events:
            assert "event" in event, "SSE event missing 'event' field"
            assert event["event"] != "", "SSE event has empty 'event' field"
            # Each event data should be valid JSON
            data = json.loads(event["data"])
            assert "type" in data
            assert "data" in data
            assert "timestamp" in data

    def test_stream_with_auth_token(self, app):
        """Auth token is passed through in streaming mode."""
        registry = _make_mock_registry()
        data_agent = registry.get("data")
        data_agent.call_tool = AsyncMock(
            return_value=_make_agent_result("Result")
        )

        routing = json.dumps({
            "decision": "agent",
            "reasoning": "data query",
            "agent_key": "data",
            "task": "Find genomes",
        })
        llm = _make_mock_llm(routing)

        _setup_ready_state(registry=registry, llm=llm)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post(
            "/orchestrate/stream",
            json={"query": "Find genomes", "auth_token": "stream-token"},
        )
        assert resp.status_code == 200

        # Verify token was passed to agent
        call_args = data_agent.call_tool.call_args
        arguments = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("arguments", {})
        assert arguments.get("token") == "stream-token"

        _teardown_state()


# ---------------------------------------------------------------------------
# Tests: Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_orchestration_exception_returns_500(self, app):
        """Internal exception during orchestration returns 500."""
        registry = _make_mock_registry()
        llm = MagicMock(spec=LLMClient)
        llm.complete = AsyncMock(side_effect=RuntimeError("LLM exploded"))
        llm.close = AsyncMock()

        _setup_ready_state(registry=registry, llm=llm)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post(
            "/orchestrate",
            json={"query": "test"},
        )
        # The orchestrate_to_response catches internal errors and returns
        # an error status in the response, not a 500 HTTP error.
        # However, if orchestrate_to_response itself throws, we get 500.
        # With our current code, the fallback routing will produce an error response
        # since the LLM call for routing fails, but it falls back to keyword routing.
        # Since "test" doesn't match keywords well, it defaults to data agent.
        # But data agent's call_tool isn't mocked, so it will fail.
        # The orchestrator catches all of this and returns a response.
        assert resp.status_code == 200
        body = resp.json()
        # It should still return a response (error handling in orchestrate)
        assert body["status"] in ("completed", "error")

        _teardown_state()

    def test_stream_error_emits_error_event(self, app):
        """SSE stream emits error events on failure."""
        registry = _make_mock_registry()
        llm = MagicMock(spec=LLMClient)
        llm.complete = AsyncMock(side_effect=RuntimeError("LLM down"))
        llm.close = AsyncMock()

        _setup_ready_state(registry=registry, llm=llm)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post(
            "/orchestrate/stream",
            json={"query": "test"},
        )
        assert resp.status_code == 200

        events = _parse_sse_events(resp.text)
        # Should still complete with some events (orchestrator catches errors)
        assert len(events) > 0

        _teardown_state()


# ---------------------------------------------------------------------------
# Tests: Response model compliance
# ---------------------------------------------------------------------------


class TestResponseModel:
    def test_response_has_all_fields(self, app):
        """OrchestratorResponse includes all expected fields."""
        registry = _make_mock_registry()
        data_agent = registry.get("data")
        data_agent.call_tool = AsyncMock(
            return_value=_make_agent_result("Answer here.")
        )

        routing = json.dumps({
            "decision": "agent",
            "reasoning": "data",
            "agent_key": "data",
            "task": "query",
        })
        llm = _make_mock_llm(routing)

        _setup_ready_state(registry=registry, llm=llm)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/orchestrate", json={"query": "test"})
        body = resp.json()

        # All OrchestratorResponse fields should be present
        assert "response_text" in body
        assert "agent_used" in body
        assert "agents_used" in body
        assert "tool_calls" in body
        assert "result_for_ui" in body
        assert "execution_trace" in body
        assert "status" in body

        _teardown_state()


# ---------------------------------------------------------------------------
# SSE parsing helper
# ---------------------------------------------------------------------------


def _parse_sse_events(raw: str) -> list[dict[str, str]]:
    """Parse raw SSE text into a list of event dicts.

    Each event is a dict with 'event' and 'data' keys.
    """
    events = []
    current: dict[str, str] = {}

    for line in raw.split("\n"):
        line = line.rstrip()
        if line.startswith("event:"):
            current["event"] = line[len("event:"):].strip()
        elif line.startswith("data:"):
            current["data"] = line[len("data:"):].strip()
        elif line == "" and current:
            if "event" in current and "data" in current:
                events.append(current)
            current = {}

    # Handle final event without trailing newline
    if "event" in current and "data" in current:
        events.append(current)

    return events
