"""
Agent-compatible MCP tool functions for BV-BRC data access.

This package contains tool functions that wrap the MCP server's
data_functions module with proper normalization (normalize_select,
normalize_sort) and structured error handling. These functions will
eventually be migrated into the MCP server's tools/data_tools.py.

For local development, imports from the MCP server use sys.path via
data_agent.tools._mcp_imports.
"""
