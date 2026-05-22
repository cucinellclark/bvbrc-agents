"""MCP client wrapper for connecting to agent MCP servers.

Uses fastmcp.Client for protocol handling. This module adds:
- Connection lifecycle management (connect / disconnect / reconnect)
- Tool discovery caching
- Health checks via ping
- Auth token injection via Bearer header (matching the Node.js client pattern)

Auth strategy:
    The BV-BRC MCP servers require a Bearer token on ALL requests (including
    tools/list). The token is a raw PATRIC user token passed as:
        Authorization: Bearer un=user@patricbrc.org|tokenid=...|expiry=...
    This bypasses the full OAuth dance. The orchestrator injects this header
    via StreamableHttpTransport's headers parameter.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from mcp.types import Tool as McpTool, CallToolResult

logger = logging.getLogger(__name__)


class MCPClientError(Exception):
    """Raised when an MCP operation fails."""


def _build_transport(
    endpoint: str,
    auth_token: str | None = None,
    timeout: int = 120,
) -> StreamableHttpTransport:
    """Build a StreamableHttpTransport with optional Bearer token auth.

    The BV-BRC MCP servers expect:
    - POST to <endpoint>/mcp (FastMCP appends this automatically if needed)
    - Authorization: Bearer <patric_token> on every request

    If the endpoint doesn't end with /mcp, we append it since the FastMCP
    server listens at the /mcp subpath.
    """
    # Ensure the endpoint includes /mcp path
    url = endpoint.rstrip("/")
    if not url.endswith("/mcp"):
        url = f"{url}/mcp"

    headers: dict[str, str] = {}
    if auth_token:
        if auth_token.startswith("Bearer "):
            headers["Authorization"] = auth_token
        else:
            headers["Authorization"] = f"Bearer {auth_token}"

    return StreamableHttpTransport(
        url=url,
        headers=headers if headers else None,
    )


class MCPAgentClient:
    """Client for a single agent's MCP server.

    Wraps fastmcp.Client with connection management, discovery caching,
    and health checking.

    Usage:
        # Without auth (for servers that don't require it):
        client = MCPAgentClient("http://localhost:8053", name="service")

        # With auth (for BV-BRC MCP servers):
        client = MCPAgentClient(
            "http://localhost:8053",
            name="service",
            auth_token="un=user@patricbrc.org|tokenid=...|expiry=...",
        )

        async with client:
            tools = await client.discover_tools()
            result = await client.call_tool("bvbrc_search_data", {"user_query": "..."})
    """

    def __init__(
        self,
        endpoint: str,
        name: str = "agent",
        timeout: int = 120,
        auth_token: str | None = None,
    ):
        self.endpoint = endpoint
        self.name = name
        self.timeout = timeout
        self.auth_token = auth_token

        self._client: Client | None = None
        self._tools: list[McpTool] | None = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self._client is not None

    def _build_client(self) -> Client:
        """Build a fastmcp Client with the appropriate transport and auth."""
        transport = _build_transport(
            endpoint=self.endpoint,
            auth_token=self.auth_token,
            timeout=self.timeout,
        )
        return Client(
            transport,
            name=f"orchestrator-{self.name}",
            timeout=self.timeout,
        )

    async def connect(self) -> None:
        """Establish connection to the MCP server."""
        if self._connected:
            return

        try:
            self._client = self._build_client()
            await self._client.__aenter__()
            self._connected = True
            logger.info(f"Connected to MCP server: {self.name} @ {self.endpoint}")
        except Exception as e:
            self._connected = False
            self._client = None
            raise MCPClientError(
                f"Failed to connect to {self.name} @ {self.endpoint}: {e}"
            ) from e

    async def disconnect(self) -> None:
        """Close the MCP connection."""
        if self._client and self._connected:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error disconnecting from {self.name}: {e}")
            finally:
                self._connected = False
                self._client = None
                self._tools = None

    async def reconnect(self) -> None:
        """Disconnect and reconnect (useful after auth token refresh)."""
        await self.disconnect()
        await self.connect()

    async def __aenter__(self) -> MCPAgentClient:
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()

    def update_auth_token(self, token: str) -> None:
        """Update the auth token. Requires reconnect() to take effect."""
        self.auth_token = token

    def _require_connection(self) -> Client:
        if not self._client or not self._connected:
            raise MCPClientError(
                f"Not connected to {self.name}. Call connect() first."
            )
        return self._client

    async def discover_tools(self, force: bool = False) -> list[McpTool]:
        """Discover available tools from the MCP server.

        Results are cached. Pass force=True to re-discover.
        """
        if self._tools is not None and not force:
            return self._tools

        client = self._require_connection()
        try:
            self._tools = await client.list_tools()
            logger.info(
                f"Discovered {len(self._tools)} tools from {self.name}: "
                f"{[t.name for t in self._tools]}"
            )
            return self._tools
        except Exception as e:
            raise MCPClientError(
                f"Tool discovery failed for {self.name}: {e}"
            ) from e

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        progress_handler: Any | None = None,
    ) -> CallToolResult:
        """Call a tool on the MCP server.

        Args:
            tool_name: The tool name (as returned by discover_tools).
            arguments: Tool arguments matching its input schema.
            progress_handler: Optional async callback for progress notifications.
                Signature: async (progress, total, message) -> None.

        Returns:
            CallToolResult with content blocks.
        """
        client = self._require_connection()
        try:
            result = await client.call_tool_mcp(
                name=tool_name,
                arguments=arguments or {},
                progress_handler=progress_handler,
            )
            return result
        except Exception as e:
            raise MCPClientError(
                f"Tool call {self.name}.{tool_name} failed: {e}"
            ) from e

    async def ping(self) -> tuple[bool, float]:
        """Health check the MCP server.

        Returns:
            Tuple of (is_healthy, latency_ms).
        """
        client = self._require_connection()
        start = time.monotonic()
        try:
            healthy = await client.ping()
            latency = (time.monotonic() - start) * 1000
            return healthy, latency
        except Exception:
            latency = (time.monotonic() - start) * 1000
            return False, latency

    def get_cached_tools(self) -> list[McpTool] | None:
        """Return cached tools without making a network call."""
        return self._tools

    def tool_schemas_for_prompt(self) -> str:
        """Format discovered tools as a human/LLM-readable string."""
        if not self._tools:
            return f"[{self.name}: no tools discovered]"

        lines = [f"Agent: {self.name} ({self.endpoint})"]
        for tool in self._tools:
            lines.append(f"  - {tool.name}: {tool.description or '(no description)'}")
            if tool.inputSchema and "properties" in tool.inputSchema:
                props = tool.inputSchema["properties"]
                required = tool.inputSchema.get("required", [])
                for pname, pschema in props.items():
                    req = " (required)" if pname in required else ""
                    ptype = pschema.get("type", "any")
                    pdesc = pschema.get("description", "")
                    lines.append(f"      {pname}: {ptype}{req} — {pdesc}")
        return "\n".join(lines)
