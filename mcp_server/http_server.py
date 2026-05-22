#!/usr/bin/env python3
"""
BVBRC Consolidated MCP Server

This server consolidates the data, service, and workspace MCP servers into a single unified server.
"""

from fastmcp import FastMCP
from common.json_rpc import JsonRpcCaller
from tools.data_tools import register_data_tools, request_download_cancel
from tools.service_tools import register_service_tools
from tools.workspace_tools import register_workspace_tools
from tools.group_tools import register_group_tools
from tools.rag_database_tools import register_rag_database_tools
from tools.sra_tools import register_sra_tools
from tools.agent_chat_tool import register_agent_chat_tool
from common.token_provider import TokenProvider
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
import sys
from common.auth import BvbrcOAuthProvider
from common.config import get_config

# Load configuration
config = get_config()

# Get configuration values
base_url = config.base_url
workspace_api_url = config.workspace_url
service_api_url = config.service_api_url
similar_genome_finder_api_url = config.similar_genome_finder_api_url
authentication_url = config.authentication_url
openid_config_url = config.openid_config_url
port = config.port
mcp_url = config.mcp_url

# Initialize token provider for HTTP mode
token_provider = TokenProvider(mode="http")

# Initialize the JSON-RPC callers with configurable timeouts
# Workspace operations can be slow, so use longer timeout
workspace_api = JsonRpcCaller(workspace_api_url, timeout=config.workspace_timeout_seconds)
service_api = JsonRpcCaller(service_api_url)
similar_genome_finder_api = JsonRpcCaller(similar_genome_finder_api_url)

# Publicly reachable server URL for discovery and metadata
# The server URL is where this MCP server is accessible, not the data API URL
server_url = config.server_url

# Initialize OAuth provider (class-based) with server URL (not data API base_url)
oauth = BvbrcOAuthProvider(
    base_url=server_url,  # This is the MCP server URL, not the data API URL
    openid_config_url=openid_config_url,
    authentication_url=authentication_url,
)

# Create FastMCP server with auth provider so /mcp is protected by FastMCP
mcp = FastMCP("BVBRC Consolidated MCP Server", auth=oauth)

# Register all tools from the three modules
print("Registering data tools...", file=sys.stderr)
register_data_tools(mcp, base_url, token_provider)

print("Registering service tools...", file=sys.stderr)
register_service_tools(mcp, service_api, similar_genome_finder_api, token_provider)

print("Registering workspace tools...", file=sys.stderr)
register_workspace_tools(mcp, workspace_api, token_provider, config.file_utilities)

print("Registering group tools...", file=sys.stderr)
register_group_tools(mcp, workspace_api, token_provider)

print("Registering RAG database tools...", file=sys.stderr)
register_rag_database_tools(mcp, config.rag_database)

print("Registering SRA tools...", file=sys.stderr)
register_sra_tools(mcp, config.sra_tools)

print("Registering agent chat tool...", file=sys.stderr)
register_agent_chat_tool(mcp, token_provider)

# Add health check tool
# @mcp.tool()
def health_check() -> str:
    """Health check endpoint"""
    return '{"status": "healthy", "service": "bvbrc-consolidated-mcp"}'

# Add OAuth2 endpoints
@mcp.custom_route("/mcp/.well-known/openid-configuration", methods=["GET"])
async def openid_configuration_route(request) -> JSONResponse:
    """
    Serves the OIDC discovery document that ChatGPT expects.
    """
    return await oauth.openid_configuration(request)

# OAuth Authorization Server metadata (well-known)
# Per RFC 8414 section 3, if issuer is "https://example.com/issuer1",
# metadata is at "https://example.com/.well-known/oauth-authorization-server/issuer1"
# So for issuer {server_url}/mcp, metadata should be at /.well-known/oauth-authorization-server/mcp
@mcp.custom_route("/.well-known/oauth-authorization-server/mcp", methods=["GET"])
async def oauth_as_metadata(request) -> JSONResponse:
    issuer = f"{server_url}/mcp"
    return JSONResponse({
        "issuer": issuer,
        "authorization_endpoint": f"{issuer}/oauth2/authorize",
        "token_endpoint": f"{issuer}/oauth2/token",
        "registration_endpoint": f"{issuer}/oauth2/register",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code"],
        "token_endpoint_auth_methods_supported": ["none", "client_secret_post"],
        "code_challenge_methods_supported": ["S256"],
        "scopes_supported": ["profile", "token"],
    })

@mcp.custom_route("/mcp/oauth2/register", methods=["POST"])
async def oauth2_register_route(request) -> JSONResponse:
    """
    Registers a new client with the OAuth2 server.
    Implements RFC 7591 OAuth 2.0 Dynamic Client Registration.
    """
    return await oauth.oauth2_register(request)

@mcp.custom_route("/mcp/oauth2/authorize", methods=["GET"])
async def oauth2_authorize_route(request):
    """
    Authorization endpoint - displays login page for user authentication.
    This is where ChatGPT redirects the user to log in.
    """
    return await oauth.oauth2_authorize(request)

@mcp.custom_route("/mcp/oauth2/login", methods=["POST"])
async def oauth2_login_route(request):
    """
    Handles the login form submission.
    Authenticates the user and generates an authorization code.
    Redirects back to ChatGPT's callback URL with the code.
    """
    return await oauth.oauth2_login(request)

@mcp.custom_route("/mcp/oauth2/token", methods=["POST"])
async def oauth2_token_route(request):
    """
    Handles the token request.
    Exchanges an authorization code for an access token.
    Retrieves the stored user token using the authorization code.
    """
    return await oauth.oauth2_token(request)


@mcp.custom_route("/mcp/cancel-data-download", methods=["POST"])
async def cancel_data_download_route(request) -> JSONResponse:
    """
    Cancel an in-flight data download by cooperative cancel token.
    Intended for internal API use when a user aborts a running job.
    """
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    cancel_token = str((payload or {}).get("cancel_token", "")).strip()
    if not cancel_token:
        return JSONResponse(
            {"ok": False, "error": "cancel_token is required"},
            status_code=400
        )

    accepted = request_download_cancel(cancel_token)
    return JSONResponse(
        {
            "ok": bool(accepted),
            "cancel_token": cancel_token,
            "message": "Cancellation requested" if accepted else "Cancellation token rejected"
        },
        status_code=200 if accepted else 400
    )

def main() -> int:
    print(f"Starting BVBRC Consolidated MCP Server on port {port}...", file=sys.stderr)
    print(f"  - Data tools: {base_url}", file=sys.stderr)
    print(f"  - Service tools: {service_api_url}", file=sys.stderr)
    print(f"  - Workspace tools: {workspace_api_url}", file=sys.stderr)
    print(f"  - RAG database tools: enabled", file=sys.stderr)
    print(f"  - SRA tools: enabled", file=sys.stderr)
    
    try:
        mcp.run(transport="http", host=mcp_url, port=port)
    except KeyboardInterrupt:
        print("Server stopped.", file=sys.stderr)
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        return 1
    
    return 0

if __name__ == "__main__":
    main()

