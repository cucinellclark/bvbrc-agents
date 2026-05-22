"""Agent Registry — discovers, manages, and provides access to agents.

The registry is the orchestrator's view of the agent ecosystem. It:
1. Loads agent configuration from agents.yaml
2. Connects to each agent's MCP server
3. Discovers available tools
4. Runs periodic health checks
5. Provides lookup by key, capability, or tool name
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

from orchestrator.config import AgentConfig, OrchestratorConfig
from orchestrator.registry.agent_handle import AgentHandle
from orchestrator.events.events import (
    Event,
    EventType,
    discovery_event,
    error_event,
)

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Central registry for all agents the orchestrator can delegate to.

    Usage:
        config = OrchestratorConfig.from_yaml("config/agents.yaml")
        registry = AgentRegistry(config)

        async for event in registry.discover_all():
            print(event)  # Stream discovery progress

        agent = registry.get("data")
        result = await agent.call_tool("bvbrc_search_data", {...})
    """

    def __init__(self, config: OrchestratorConfig):
        self._config = config
        self._agents: dict[str, AgentHandle] = {}
        self._health_task: asyncio.Task | None = None

    @property
    def agents(self) -> dict[str, AgentHandle]:
        """All registered agents (including unhealthy ones)."""
        return dict(self._agents)

    @property
    def healthy_agents(self) -> dict[str, AgentHandle]:
        """Only agents that passed their last health check."""
        return {k: a for k, a in self._agents.items() if a.is_healthy}

    @property
    def agent_keys(self) -> list[str]:
        return list(self._agents.keys())

    # --- Discovery ---

    async def discover_all(self) -> AsyncGenerator[Event, None]:
        """Connect to all configured agents, discover their tools.

        Yields events for each agent (success or failure). Agents that
        fail to connect are still registered but marked unhealthy.
        """
        yield Event(
            type=EventType.DISCOVERY_START,
            data={"agent_count": len(self._config.agents)},
        )

        for key, agent_config in self._config.agents.items():
            handle = AgentHandle(key, agent_config)
            self._agents[key] = handle

            try:
                await handle.connect()
                tools = await handle.discover()
                health = await handle.health_check()

                yield discovery_event(
                    agent_name=key,
                    tool_count=len(tools),
                )
                logger.info(
                    f"Agent '{key}' ready: {len(tools)} tools, "
                    f"healthy={handle.is_healthy}"
                )

            except Exception as e:
                logger.error(f"Agent '{key}' discovery failed: {e}")
                yield error_event(
                    message=f"Agent '{key}' discovery failed: {e}",
                    agent_name=key,
                    details={"endpoint": agent_config.endpoint},
                )

        healthy = len(self.healthy_agents)
        total = len(self._agents)
        yield Event(
            type=EventType.DISCOVERY_DONE,
            data={
                "total": total,
                "healthy": healthy,
                "unhealthy": total - healthy,
                "agents": {k: a.summary() for k, a in self._agents.items()},
            },
        )

    async def discover_agent(self, key: str) -> AgentHandle:
        """Discover a single agent (connect + tool discovery + health check).

        Useful for re-discovering an agent after a failure.
        """
        if key not in self._config.agents:
            raise KeyError(f"No agent configured with key '{key}'")

        config = self._config.agents[key]
        handle = AgentHandle(key, config)

        await handle.connect()
        await handle.discover()
        await handle.health_check()

        self._agents[key] = handle
        return handle

    # --- Lookup ---

    def get(self, key: str) -> AgentHandle:
        """Get an agent by its key. Raises KeyError if not found."""
        if key not in self._agents:
            available = ", ".join(self._agents.keys()) or "(none)"
            raise KeyError(
                f"Agent '{key}' not found. Available agents: {available}"
            )
        return self._agents[key]

    def find_by_capability(self, capability: str) -> list[AgentHandle]:
        """Find all agents that declare a given capability."""
        return [
            a
            for a in self._agents.values()
            if capability in a.capabilities and a.is_healthy
        ]

    def find_by_tool(self, tool_name: str) -> AgentHandle | None:
        """Find the agent that owns a given tool name.

        Returns None if no agent has this tool.
        If multiple agents have it, returns the first healthy one.
        """
        for agent in self._agents.values():
            if tool_name in agent.tool_names and agent.is_healthy:
                return agent
        # Fall back to unhealthy agents
        for agent in self._agents.values():
            if tool_name in agent.tool_names:
                return agent
        return None

    # --- Health Checks ---

    async def health_check_all(self) -> list[Event]:
        """Run health checks on all registered agents."""
        events = []
        for agent in self._agents.values():
            event = await agent.health_check()
            events.append(event)
        return events

    async def start_health_checks(self) -> None:
        """Start periodic background health checks."""
        interval = self._config.health_check_interval
        if interval <= 0:
            return

        async def _loop():
            while True:
                await asyncio.sleep(interval)
                try:
                    await self.health_check_all()
                except Exception as e:
                    logger.error(f"Health check cycle failed: {e}")

        self._health_task = asyncio.create_task(_loop())
        logger.info(f"Health checks started (interval={interval}s)")

    async def stop_health_checks(self) -> None:
        """Stop periodic health checks."""
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None

    # --- Catalog ---

    def catalog(self) -> str:
        """Build the full agent catalog string for the routing LLM.

        This is the information the router uses to decide which agent
        to invoke. It includes descriptions and tool names, but NOT
        full tool parameter schemas.
        """
        if not self._agents:
            return "[No agents registered]"

        entries = []
        for agent in self._agents.values():
            if agent.is_healthy:
                entries.append(agent.catalog_entry())

        if not entries:
            return "[No healthy agents available]"

        return "\n\n".join(entries)

    def catalog_structured(self) -> list[dict[str, Any]]:
        """Structured catalog for programmatic use."""
        return [a.summary() for a in self._agents.values()]

    # --- Lifecycle ---

    async def shutdown(self) -> None:
        """Disconnect all agents and stop health checks."""
        await self.stop_health_checks()
        for agent in self._agents.values():
            try:
                await agent.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting agent '{agent.key}': {e}")
        self._agents.clear()
        logger.info("Agent registry shut down")

    def __repr__(self) -> str:
        healthy = len(self.healthy_agents)
        total = len(self._agents)
        return f"AgentRegistry(agents={total}, healthy={healthy})"
