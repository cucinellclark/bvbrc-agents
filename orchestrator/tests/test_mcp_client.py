"""Tests for the MCP client wrapper."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from orchestrator.mcp.client import MCPAgentClient, MCPClientError, _build_transport


class TestBuildTransport:
    """Tests for transport construction logic."""

    def test_appends_mcp_path(self):
        transport = _build_transport("http://localhost:8053")
        assert str(transport.url).rstrip("/").endswith("/mcp")

    def test_preserves_existing_mcp_path(self):
        transport = _build_transport("http://localhost:8053/mcp")
        url = str(transport.url).rstrip("/")
        # Should NOT double the /mcp
        assert not url.endswith("/mcp/mcp")
        assert url.endswith("/mcp")

    def test_strips_trailing_slash(self):
        transport = _build_transport("http://localhost:8053/")
        assert str(transport.url).rstrip("/").endswith("/mcp")

    def test_no_auth_headers_without_token(self):
        transport = _build_transport("http://localhost:8053")
        assert transport.headers is None or "Authorization" not in (transport.headers or {})

    def test_auth_header_with_token(self):
        transport = _build_transport(
            "http://localhost:8053",
            auth_token="un=user@patricbrc.org|tokenid=abc|expiry=9999",
        )
        assert transport.headers is not None
        assert "Authorization" in transport.headers
        assert transport.headers["Authorization"].startswith("Bearer ")
        assert "un=user@patricbrc.org" in transport.headers["Authorization"]

    def test_auth_header_with_bearer_prefix(self):
        transport = _build_transport(
            "http://localhost:8053",
            auth_token="Bearer un=user@patricbrc.org|tokenid=abc",
        )
        assert transport.headers["Authorization"] == "Bearer un=user@patricbrc.org|tokenid=abc"
        # Should NOT double the Bearer prefix
        assert not transport.headers["Authorization"].startswith("Bearer Bearer")


class TestMCPAgentClient:
    """Tests for MCPAgentClient."""

    def test_initial_state(self):
        client = MCPAgentClient("http://localhost:8053", name="test")
        assert not client.is_connected
        assert client.get_cached_tools() is None

    def test_require_connection_raises(self):
        client = MCPAgentClient("http://localhost:8053", name="test")
        with pytest.raises(MCPClientError, match="Not connected"):
            client._require_connection()

    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        client = MCPAgentClient("http://localhost:8053", name="test")

        mock_fastmcp_client = AsyncMock()
        with patch.object(client, "_build_client", return_value=mock_fastmcp_client):
            await client.connect()
            assert client.is_connected

            await client.disconnect()
            assert not client.is_connected

    @pytest.mark.asyncio
    async def test_double_connect_is_noop(self):
        client = MCPAgentClient("http://localhost:8053", name="test")
        mock_fastmcp_client = AsyncMock()

        with patch.object(client, "_build_client", return_value=mock_fastmcp_client) as mock_build:
            await client.connect()
            await client.connect()  # Should not create a second client
            assert mock_build.call_count == 1

    @pytest.mark.asyncio
    async def test_context_manager(self):
        client = MCPAgentClient("http://localhost:8053", name="test")
        mock_fastmcp_client = AsyncMock()

        with patch.object(client, "_build_client", return_value=mock_fastmcp_client):
            async with client:
                assert client.is_connected
            assert not client.is_connected

    @pytest.mark.asyncio
    async def test_discover_tools_caching(self):
        client = MCPAgentClient("http://localhost:8053", name="test")
        mock_fastmcp_client = AsyncMock()

        from mcp.types import Tool as McpTool
        mock_tools = [McpTool(name="t1", description="d1", inputSchema={})]
        mock_fastmcp_client.list_tools = AsyncMock(return_value=mock_tools)

        with patch.object(client, "_build_client", return_value=mock_fastmcp_client):
            async with client:
                # First call discovers
                tools = await client.discover_tools()
                assert len(tools) == 1
                assert mock_fastmcp_client.list_tools.call_count == 1

                # Second call returns cache
                tools2 = await client.discover_tools()
                assert tools2 is tools
                assert mock_fastmcp_client.list_tools.call_count == 1

                # Force rediscovery
                await client.discover_tools(force=True)
                assert mock_fastmcp_client.list_tools.call_count == 2

    def test_update_auth_token(self):
        client = MCPAgentClient("http://localhost:8053", name="test")
        assert client.auth_token is None
        client.update_auth_token("new_token")
        assert client.auth_token == "new_token"

    def test_tool_schemas_for_prompt_empty(self):
        client = MCPAgentClient("http://localhost:8053", name="test")
        result = client.tool_schemas_for_prompt()
        assert "no tools discovered" in result
