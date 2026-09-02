"""Agent handles — registered agent wrappers.

The orchestrator interacts with agents exclusively through handles.
Two implementations share the same public interface:

- ``AgentHandle``: MCP HTTP client (rollback / optional remote agents)
- ``InProcessAgentHandle``: calls ``shared.agent_dispatch.dispatch_agent``
  in the same process (default Copilot path)

Both ``call_tool()`` methods return a plain ``dict``.  MCP type unwrapping
happens inside ``AgentHandle`` so the executor never imports MCP types.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from orchestrator.config import AgentConfig
from orchestrator.events.events import (
    Event,
    health_event,
)

if TYPE_CHECKING:
    from mcp.types import Tool as McpTool

logger = logging.getLogger(__name__)

# bvbrc-agents repo root (orchestrator/orchestrator/registry/this file)
_AGENTS_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)


def _ensure_agents_repo_on_path() -> None:
    """Make ``shared`` importable from the orchestrator process."""
    if _AGENTS_ROOT not in sys.path:
        sys.path.insert(0, _AGENTS_ROOT)


_ensure_agents_repo_on_path()


def _parse_mcp_result(mcp_result: Any) -> dict[str, Any]:
    """Parse an MCP CallToolResult into a dict.

    The agent_chat tool returns a JSON dict, but MCP wraps it in
    content blocks. Extract and parse the actual data.
    """
    if isinstance(mcp_result, dict):
        return mcp_result

    if not hasattr(mcp_result, "content") or not mcp_result.content:
        return {"answer": "(empty result from agent)", "status": "error"}

    parsed: dict[str, Any] | None = None
    for block in mcp_result.content:
        if hasattr(block, "text"):
            try:
                parsed = json.loads(block.text)
            except (json.JSONDecodeError, TypeError):
                parsed = {"answer": block.text, "status": "completed"}
            break

    if parsed is None:
        parsed = {"answer": str(mcp_result.content), "status": "completed"}

    if getattr(mcp_result, "isError", False) and parsed.get("status") != "error":
        parsed["status"] = "error"

    return parsed


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
        from orchestrator.mcp.client import MCPAgentClient

        self.key = key  # Short identifier, e.g. "data", "service"
        self.config = config
        self.name = config.name
        self.description = config.description
        self.capabilities = config.capabilities
        self.endpoint = config.endpoint

        if not config.endpoint:
            raise ValueError(
                f"MCP AgentHandle '{key}' requires endpoint "
                f"(protocol={config.protocol!r})"
            )

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
        from orchestrator.mcp.client import MCPClientError

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
    ) -> dict[str, Any]:
        """Call a specific tool on this agent.

        Args:
            tool_name: Tool name (must be one of this agent's discovered tools).
            arguments: Tool arguments.  A dict ``context`` is serialized to a
                JSON string at the MCP boundary.
            progress_handler: Optional async callback for progress notifications.

        Returns:
            Parsed result dict (never a raw MCP CallToolResult).

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
        arguments = dict(arguments or {})
        if isinstance(arguments.get("context"), dict):
            arguments["context"] = json.dumps(arguments["context"])

        mcp_result = await self._mcp.call_tool(
            tool_name, arguments, progress_handler=progress_handler
        )
        return _parse_mcp_result(mcp_result)

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
            "protocol": "mcp",
        }

    def __repr__(self) -> str:
        status = "healthy" if self._healthy else "unhealthy"
        return (
            f"AgentHandle(key={self.key!r}, tools={len(self._tools)}, "
            f"status={status})"
        )


class InProcessAgentHandle:
    """Agent handle that dispatches to agent code in the same process.

    No MCP connection. Calls ``shared.agent_dispatch.dispatch_agent()`` directly.
    """

    def __init__(self, key: str, config: AgentConfig):
        self.key = key
        self.config = config
        self.name = config.name
        self.description = config.description
        self.capabilities = config.capabilities or []
        self.endpoint = config.endpoint
        self._tools = ["agent_chat"]
        self._healthy = True
        self._last_latency_ms: float = 0.0
        _ensure_agents_repo_on_path()

    @property
    def is_connected(self) -> bool:
        return True

    @property
    def is_healthy(self) -> bool:
        return self._healthy

    @property
    def tools(self) -> list[str]:
        return self._tools

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools)

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def discover(self) -> list[str]:
        return self._tools

    async def health_check(self) -> Event:
        self._healthy = True
        self._last_latency_ms = 0.0
        return health_event(
            agent_name=self.key,
            healthy=True,
            latency_ms=0,
        )

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        progress_handler: Any | None = None,
    ) -> dict[str, Any]:
        """Dispatch to the in-process agent. Returns a result dict."""
        import asyncio
        from shared.agent_dispatch import dispatch_agent

        arguments = arguments or {}
        agent_type = arguments.get("agent_type")
        if not agent_type:
            agent_type = (self.config.chat_tool_params or {}).get("agent_type")
        query = arguments.get("query", "")
        context = arguments.get("context", {})
        token = arguments.get("token", "") or ""

        timeout = self.config.timeout_seconds or 300
        return await asyncio.wait_for(
            dispatch_agent(
                agent_type=agent_type,
                query=query,
                context=context,
                token=token,
                progress_callback=progress_handler,
            ),
            timeout=timeout,
        )

    def catalog_entry(self) -> str:
        """Format this agent as a catalog entry for the routing LLM."""
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
        return ""

    def summary(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "healthy": True,
            "protocol": "inprocess",
            "endpoint": self.endpoint,
            "connected": True,
            "latency_ms": 0.0,
            "tool_count": len(self._tools),
            "tools": self.tool_names,
            "capabilities": self.capabilities,
        }

    def __repr__(self) -> str:
        return (
            f"InProcessAgentHandle(key={self.key!r}, "
            f"tools={self._tools}, status=healthy)"
        )
