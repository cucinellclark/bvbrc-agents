"""AgentHandle — a registered agent wrapper.

Analogous to claude-code's AgentTool: encapsulates a single sub-agent
with its MCP connection, discovered tools, capabilities, and metadata.
The orchestrator interacts with agents exclusively through handles.
"""

from __future__ import annotations

import logging
from typing import Any

from mcp.types import Tool as McpTool, CallToolResult

from orchestrator.config import AgentConfig
from orchestrator.mcp.client import MCPAgentClient, MCPClientError
from orchestrator.events.events import (
    Event,
    EventType,
    agent_start_event,
    agent_result_event,
    error_event,
    health_event,
)

logger = logging.getLogger(__name__)


class AgentHandle:
    """A registered agent that can be discovered, health-checked, and invoked.

    Lifecycle:
        1. Created from AgentConfig
        2. connect() — establishes MCP connection
        3. discover() — discovers tools via MCP tools/list
        4. invoke() / call_tool() — executes agent tools
        5. disconnect() — closes MCP connection
    """

    def __init__(self, key: str, config: AgentConfig):
        self.key = key  # Short identifier, e.g. "data", "service2"
        self.config = config
        self.name = config.name
        self.description = config.description
        self.capabilities = config.capabilities
        self.endpoint = config.endpoint

        self._mcp = MCPAgentClient(
            endpoint=config.endpoint,
            name=key,
            timeout=config.timeout_seconds,
            auth_token=config.auth_token,
        )
        self._tools: list[McpTool] = []
        self._healthy: bool = False
        self._last_latency_ms: float = 0.0

    @property
    def is_connected(self) -> bool:
        return self._mcp.is_connected

    @property
    def is_healthy(self) -> bool:
        return self._healthy

    @property
    def tools(self) -> list[McpTool]:
        return self._tools

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self._tools]

    # --- Lifecycle ---

    async def connect(self) -> None:
        """Connect to the agent's MCP server."""
        await self._mcp.connect()

    async def disconnect(self) -> None:
        """Disconnect from the agent's MCP server."""
        await self._mcp.disconnect()
        self._healthy = False

    async def discover(self) -> list[McpTool]:
        """Discover tools from the agent's MCP server."""
        self._tools = await self._mcp.discover_tools(force=True)
        logger.info(
            f"Agent '{self.key}' discovered {len(self._tools)} tools: "
            f"{self.tool_names}"
        )
        return self._tools

    async def health_check(self) -> Event:
        """Ping the agent and return a health event."""
        try:
            self._healthy, self._last_latency_ms = await self._mcp.ping()
        except MCPClientError:
            self._healthy = False
            self._last_latency_ms = -1.0

        return health_event(
            agent_name=self.key,
            healthy=self._healthy,
            latency_ms=self._last_latency_ms,
        )

    # --- Tool invocation ---

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        progress_handler: Any | None = None,
    ) -> CallToolResult:
        """Call a specific tool on this agent.

        Args:
            tool_name: Tool name (must be one of this agent's discovered tools).
            arguments: Tool arguments.
            progress_handler: Optional async callback for progress notifications.

        Returns:
            MCP CallToolResult.

        Raises:
            MCPClientError: If the tool call fails.
            ValueError: If the tool is not found on this agent.
        """
        if tool_name not in self.tool_names:
            available = ", ".join(self.tool_names) or "(none)"
            raise ValueError(
                f"Tool '{tool_name}' not found on agent '{self.key}'. "
                f"Available tools: {available}"
            )
        return await self._mcp.call_tool(tool_name, arguments, progress_handler=progress_handler)

    # --- Formatting ---

    def catalog_entry(self) -> str:
        """Format this agent as a catalog entry for the routing LLM.

        Returns a concise description with capabilities and tool names
        (but NOT tool parameters — the router doesn't need them).
        """
        tools_str = ", ".join(self.tool_names) if self.tool_names else "(no tools)"
        caps_str = ", ".join(self.capabilities) if self.capabilities else "(none)"
        return (
            f"Agent: {self.name} (key={self.key})\n"
            f"  Description: {self.description.strip()}\n"
            f"  Capabilities: {caps_str}\n"
            f"  Tools: {tools_str}\n"
            f"  Status: {'healthy' if self._healthy else 'unhealthy'}"
        )

    def tools_detail(self) -> str:
        """Format full tool details (for agent execution, not routing)."""
        return self._mcp.tool_schemas_for_prompt()

    def summary(self) -> dict[str, Any]:
        """Structured summary for logging/debugging."""
        return {
            "key": self.key,
            "name": self.name,
            "endpoint": self.endpoint,
            "connected": self.is_connected,
            "healthy": self._healthy,
            "latency_ms": round(self._last_latency_ms, 1),
            "tool_count": len(self._tools),
            "tools": self.tool_names,
            "capabilities": self.capabilities,
        }

    def __repr__(self) -> str:
        status = "healthy" if self._healthy else "unhealthy"
        return (
            f"AgentHandle(key={self.key!r}, tools={len(self._tools)}, "
            f"status={status})"
        )
