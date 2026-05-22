"""
Configuration management for BV-BRC MCP Server.

Provides centralized configuration loading from config.json with support for
environment variables and sensible defaults.
"""

import json
import os
from pathlib import Path
from typing import Optional, List


class OAuthConfig:
    """OAuth2/OIDC configuration settings."""
    
    def __init__(self, config_dict: dict):
        oauth_config = config_dict.get("oauth", {})
        
        # Allowed callback URLs for OAuth redirects (exact match)
        self.allowed_callback_urls = oauth_config.get(
            "allowed_callback_urls",
            [
                "https://chatgpt.com/connector_platform_oauth_redirect",
                "https://claude.ai/api/mcp/auth_callback"
            ]
        )
        
        # Allowed callback origins for OAuth redirects (origin/prefix match).
        # Any redirect_uri whose origin (scheme + host) matches an entry here
        # will be accepted, regardless of the path. This supports clients like
        # ChatGPT that use randomly-generated callback paths.
        self.allowed_callback_origins: List[str] = oauth_config.get(
            "allowed_callback_origins", []
        )
        
        # Trusted client IDs that can auto-register
        # None = disable auto-registration (most secure)
        # [] = allow all clients to auto-register
        # [list] = only these client_ids can auto-register
        self.trusted_client_ids: Optional[List[str]] = oauth_config.get(
            "trusted_client_ids", None
        )
        
        # Token expiration times
        self.access_token_expires_in_seconds = oauth_config.get(
            "access_token_expires_in_seconds", 7200  # 2 hours
        )
        self.authorization_code_expires_in_seconds = oauth_config.get(
            "authorization_code_expires_in_seconds", 3600  # 1 hour
        )


class AppConfig:
    """Main application configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Default to config/config.json relative to project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "config.json"
        
        # Load configuration file
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Config file not found at {config_path}, using defaults", file=os.sys.stderr)
            config = {}
        
        # API endpoints
        self.base_url = config.get(
            "base_url", 
            "https://www.bv-brc.org/api-bulk"
        )
        self.workspace_url = config.get(
            "workspace_url",
            "https://p3.theseed.org/services/Workspace"
        )
        self.workspace_timeout_seconds = config.get(
            "workspace_timeout_seconds",
            120  # Default 2 minutes for workspace operations
        )
        self.service_api_url = config.get(
            "service_api_url",
            "https://p3.theseed.org/services/app_service"
        )
        self.similar_genome_finder_api_url = config.get(
            "similar_genome_finder_api_url",
            self.service_api_url  # Default to service_api_url if not specified
        )
        self.file_utilities = config.get("file_utilities", {})
        
        # RAG database configuration
        self.rag_database = config.get("rag_database", {})
        
        # SRA tools configuration
        self.sra_tools = config.get("sra_tools", {})
        
        # Authentication
        self.authentication_url = config.get(
            "authentication_url",
            "https://user.patricbrc.org/authenticate"
        )
        self.openid_config_url = config.get(
            "openid_config_url",
            "https://dev-7.bv-brc.org"
        )
        
        # Server settings
        self.mcp_url = config.get("mcp_url", "127.0.0.1")
        self.port = int(os.environ.get("PORT", config.get("port", 12010)))
        
        # OAuth configuration
        self.oauth = OAuthConfig(config)
    
    @property
    def server_url(self) -> str:
        """Get the publicly reachable server URL."""
        # Use PUBLIC_BASE_URL env var if set, otherwise use openid_config_url
        return os.environ.get("PUBLIC_BASE_URL") or self.openid_config_url


# Global config instance
_config: Optional[AppConfig] = None


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load configuration from file.
    
    Args:
        config_path: Optional path to config file. If None, uses default location.
    
    Returns:
        AppConfig instance
    """
    global _config
    if _config is None:
        _config = AppConfig(config_path)
    return _config


def get_config() -> AppConfig:
    """
    Get the global configuration instance.
    
    Returns:
        AppConfig instance (loads if not already loaded)
    """
    if _config is None:
        return load_config()
    return _config


def reset_config():
    """Reset the global config (useful for testing)."""
    global _config
    _config = None

