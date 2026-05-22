import os
import json
import sys
from typing import Optional

try:
    from fastmcp.server.dependencies import get_http_headers
except ImportError:
    # Fallback if fastmcp.server.dependencies is not available
    def get_http_headers():
        return {}

class TokenProvider:
    """Handles token retrieval for both stdio and HTTP modes"""
    
    def __init__(self, mode: str = "stdio", config_path: str = "config/config.json", mcp_config_path: str = "config/mcp_config.json"):
        self.mode = mode
        self.config_path = config_path
        self.mcp_config_path = mcp_config_path
        self._config_token = None
        self._mcp_config_token = None
    
    def get_token(self, provided_token: Optional[str] = None) -> Optional[str]:
        """
        Get token based on mode and provided token.
        In HTTP mode, automatically checks the Authorization header from the request.
        Priority: Authorization header (HTTP mode) > provided_token > environment/config
        
        Args:
            provided_token: Token provided by the tool call (for HTTP mode)
            
        Returns:
            The appropriate token to use
        """
        # In HTTP mode, first check the Authorization header from the request
        if self.mode == "http":
            auth_header_token = self._get_token_from_request_headers()
            if auth_header_token:
                return auth_header_token
        
        # Then check provided token
        if provided_token:
            # If token is provided (HTTP mode), use it
            return provided_token
        
        if self.mode == "stdio":
            # STDIO mode: get from environment
            env_token = os.getenv("KB_AUTH_TOKEN")
            if env_token:
                return env_token
            else:
                print("Warning: KB_AUTH_TOKEN environment variable is not set", file=sys.stderr)
                print("  Service and workspace tools will require a token parameter", file=sys.stderr)
                return None
        else:
            print("Warning: Invalid mode", file=sys.stderr)
            return None
    
    def _get_token_from_request_headers(self) -> Optional[str]:
        """
        Extract token from the Authorization header of the current HTTP request.
        Uses FastMCP's get_http_headers() to access request context.
        
        Returns:
            Extracted token or None if not found
        """
        try:
            headers = get_http_headers()
            # Headers are case-insensitive, but typically lowercase
            auth_header = headers.get("authorization") or headers.get("Authorization")
            if auth_header:
                return self._parse_authorization_header(auth_header)
        except Exception as e:
            print(f"Warning: Could not get HTTP headers: {e}", file=sys.stderr)
        return None
    
    def _parse_authorization_header(self, auth_header: str) -> Optional[str]:
        """
        Parse token from Authorization header.
        Supports both "Bearer <token>" and plain token formats.
        
        Args:
            auth_header: Authorization header value
            
        Returns:
            Extracted token or None if invalid format
        """
        if not auth_header:
            return None
        
        # Remove leading/trailing whitespace
        auth_header = auth_header.strip()
        
        # Check for "Bearer " prefix (case-insensitive)
        if auth_header.lower().startswith("bearer "):
            return auth_header[7:].strip()
        
        # If no "Bearer " prefix, assume it's a plain token
        return auth_header
    
    def _load_config_token(self):
        """Load token from config file"""
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
                self._config_token = config.get("token")
        except Exception as e:
            print(f"Warning: Could not load token from config: {e}", file=sys.stderr)
            self._config_token = None
    
