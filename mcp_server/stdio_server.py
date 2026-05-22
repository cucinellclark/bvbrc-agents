#!/usr/bin/env python3
"""
BVBRC Consolidated MCP Server - STDIO Mode

This server consolidates the data, service, and workspace MCP servers into a single unified server
running in STDIO mode for use with MCP clients like Claude Desktop.
"""

from fastmcp import FastMCP
from common.json_rpc import JsonRpcCaller
from tools.data_tools import register_data_tools
from tools.service_tools import register_service_tools, extract_userid_from_token
from tools.workspace_tools import register_workspace_tools
from tools.group_tools import register_group_tools
from tools.rag_database_tools import register_rag_database_tools
from common.token_provider import TokenProvider
from functions.workflow_functions import initialize_service_catalog
import json
import sys
import os

# Load configuration
try:
    with open("config/config.json", "r") as f:
        config = json.load(f)
except FileNotFoundError:
    print("Warning: config/config.json not found, using defaults", file=sys.stderr)
    config = {}

# Get configuration values
base_url = config.get("base_url", "https://www.bv-brc.org/api-bulk")
workspace_api_url = config.get("workspace_url", "https://p3.theseed.org/services/Workspace")
workspace_timeout = config.get("workspace_timeout_seconds", 120)
service_api_url = config.get("service_api_url", "https://p3.theseed.org/services/app_service")
similar_genome_finder_api_url = config.get("similar_genome_finder_api_url", service_api_url)
file_utilities_config = config.get("file_utilities", {})
rag_database_config = config.get("rag_database", {})

# Initialize token provider for STDIO mode
# In STDIO mode, token comes from KB_AUTH_TOKEN environment variable
token_provider = TokenProvider(mode="stdio")

# Initialize the JSON-RPC callers with configurable timeouts
# Workspace operations can be slow, so use longer timeout
workspace_api = JsonRpcCaller(workspace_api_url, timeout=workspace_timeout)
service_api = JsonRpcCaller(service_api_url)
similar_genome_finder_api = JsonRpcCaller(similar_genome_finder_api_url)

# Create FastMCP server
mcp = FastMCP("BVBRC Consolidated MCP Server")

# Register all tools from the three modules
print("Registering data tools...", file=sys.stderr)
register_data_tools(mcp, base_url, token_provider)

print("Registering service tools...", file=sys.stderr)
register_service_tools(mcp, service_api, similar_genome_finder_api, token_provider)

print("Registering workspace tools...", file=sys.stderr)
register_workspace_tools(mcp, workspace_api, token_provider, file_utilities_config)

print("Registering group tools...", file=sys.stderr)
register_group_tools(mcp, workspace_api, token_provider)

print("Registering RAG database tools...", file=sys.stderr)
register_rag_database_tools(mcp, rag_database_config)

# Add health check tool
@mcp.tool()
def health_check() -> str:
    """Health check endpoint"""
    print("Health check endpoint called", file=sys.stderr)
    return '{"status": "healthy", "service": "bvbrc-consolidated-mcp", "mode": "stdio"}'

def main() -> int:
    """Main entry point for the BVBRC Consolidated MCP Server in STDIO mode."""
    print("Starting BVBRC Consolidated MCP Server in STDIO mode...", file=sys.stderr)
    print(f"  - Data tools: {base_url}", file=sys.stderr)
    print(f"  - Service tools: {service_api_url}", file=sys.stderr)
    print(f"  - Workspace tools: {workspace_api_url}", file=sys.stderr)
    print(f"  - RAG database tools: enabled", file=sys.stderr)
    print(f"  - Authentication: KB_AUTH_TOKEN environment variable", file=sys.stderr)
    
    # Check if KB_AUTH_TOKEN is set
    if not os.getenv("KB_AUTH_TOKEN"):
        print("Warning: KB_AUTH_TOKEN environment variable is not set", file=sys.stderr)
        print("  Service and workspace tools will require a token parameter", file=sys.stderr)
    else:
        print("  KB_AUTH_TOKEN is set ✓", file=sys.stderr)
        # Try to initialize service catalog at startup for better performance
        try:
            default_token = token_provider.get_token(None)
            if default_token:
                user_id = extract_userid_from_token(default_token)
                initialize_service_catalog(service_api, default_token, user_id)
        except Exception as e:
            print(f"  Note: Could not pre-initialize service catalog: {e}", file=sys.stderr)
            print("    Catalog will be built on first use instead", file=sys.stderr)
    
    try:
        # Run in stdio mode
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        print("Server stopped.", file=sys.stderr)
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

