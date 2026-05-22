"""Tests for the agent registry (unit tests with mocked MCP)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from mcp.types import Tool as McpTool

from orchestrator.config import AgentConfig, OrchestratorConfig
from orchestrator.registry.agent_handle import AgentHandle
from orchestrator.registry.agent_registry import AgentRegistry
from orchestrator.events.events import EventType
from orchestrator.events.stream import collect_events


def _make_config(agents: dict | None = None) -> OrchestratorConfig:
    """Create a test config."""
    if agents is None:
        agents = {
            "test": AgentConfig(
                name="Test Agent",
                description="A test agent for unit tests",
                endpoint="http://localhost:9999",
                capabilities=["testing"],
            )
        }
    return OrchestratorConfig(agents=agents, health_check_interval=0)


def _make_mcp_tool(name: str, description: str = "") -> McpTool:
    """Create a mock MCP tool."""
    return McpTool(
        name=name,
        description=description or f"Tool: {name}",
        inputSchema={"type": "object", "properties": {}},
    )


class TestAgentHandle:
    """Tests for AgentHandle."""

    def test_creation(self):
        config = AgentConfig(
            name="Test", description="desc", endpoint="http://localhost:1234",
            capabilities=["cap1", "cap2"],
        )
        handle = AgentHandle("test", config)
        assert handle.key == "test"
        assert handle.name == "Test"
        assert not handle.is_connected
        assert not handle.is_healthy
        assert handle.tools == []
        assert handle.tool_names == []

    def test_catalog_entry(self):
        config = AgentConfig(
            name="Test Agent", description="Does testing",
            endpoint="http://localhost:1234", capabilities=["testing"],
        )
        handle = AgentHandle("test", config)
        entry = handle.catalog_entry()
        assert "Test Agent" in entry
        assert "testing" in entry
        assert "test" in entry

    def test_summary(self):
        config = AgentConfig(
            name="Test", description="desc", endpoint="http://localhost:1234",
        )
        handle = AgentHandle("test", config)
        summary = handle.summary()
        assert summary["key"] == "test"
        assert summary["connected"] is False
        assert summary["tool_count"] == 0


class TestAgentRegistry:
    """Tests for AgentRegistry."""

    def test_creation(self):
        config = _make_config()
        registry = AgentRegistry(config)
        assert registry.agents == {}
        assert registry.agent_keys == []

    def test_get_missing_agent(self):
        config = _make_config()
        registry = AgentRegistry(config)
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_find_by_tool_empty(self):
        config = _make_config()
        registry = AgentRegistry(config)
        assert registry.find_by_tool("some_tool") is None

    def test_catalog_empty(self):
        config = _make_config()
        registry = AgentRegistry(config)
        assert "No agents registered" in registry.catalog()

    @pytest.mark.asyncio
    async def test_discover_all_with_mock(self):
        """Test discovery with a mocked MCP client."""
        config = _make_config()
        registry = AgentRegistry(config)

        mock_tools = [
            _make_mcp_tool("tool_a", "Tool A"),
            _make_mcp_tool("tool_b", "Tool B"),
        ]

        with patch.object(AgentHandle, "connect", new_callable=AsyncMock), \
             patch.object(AgentHandle, "discover", new_callable=AsyncMock, return_value=mock_tools), \
             patch.object(AgentHandle, "health_check", new_callable=AsyncMock) as mock_health:

            # Simulate healthy agent
            from orchestrator.events.events import health_event
            mock_health.return_value = health_event("test", True, 5.0)

            events = await collect_events(registry.discover_all())

        # Should have: discovery_start, discovery_agent, discovery_done
        event_types = [e.type for e in events]
        assert EventType.DISCOVERY_START in event_types
        assert EventType.DISCOVERY_DONE in event_types

        # Agent should be registered
        assert "test" in registry.agents

    @pytest.mark.asyncio
    async def test_discover_handles_connection_failure(self):
        """Test that discovery continues even when an agent fails."""
        agents = {
            "good": AgentConfig(
                name="Good", description="works", endpoint="http://localhost:1111",
            ),
            "bad": AgentConfig(
                name="Bad", description="broken", endpoint="http://localhost:2222",
            ),
        }
        config = _make_config(agents)
        registry = AgentRegistry(config)

        call_count = 0

        async def mock_connect(self):
            nonlocal call_count
            call_count += 1
            if self.key == "bad":
                raise ConnectionError("refused")

        async def mock_discover(self):
            self._tools = [_make_mcp_tool("tool_x")]
            return self._tools

        async def mock_health(self):
            self._healthy = True
            from orchestrator.events.events import health_event
            return health_event(self.key, True, 1.0)

        with patch.object(AgentHandle, "connect", mock_connect), \
             patch.object(AgentHandle, "discover", mock_discover), \
             patch.object(AgentHandle, "health_check", mock_health):

            events = await collect_events(registry.discover_all())

        # Both agents should be in the registry (bad one is unhealthy)
        assert "good" in registry.agents
        assert "bad" in registry.agents

        # Should have an error event for the bad agent
        error_events = [e for e in events if e.type == EventType.ORCHESTRATOR_ERROR]
        assert len(error_events) == 1
        assert "bad" in error_events[0].data["error"]

    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test clean shutdown."""
        config = _make_config()
        registry = AgentRegistry(config)

        # Manually add a mock agent
        handle = AgentHandle("test", config.agents["test"])
        handle._mcp = AsyncMock()
        registry._agents["test"] = handle

        await registry.shutdown()
        assert registry.agents == {}
