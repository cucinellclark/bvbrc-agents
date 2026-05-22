"""Configuration loading for the orchestrator.

Reads agent definitions from config/agents.yaml and orchestrator settings
from environment variables or defaults. LLM settings are loaded from the
shared Agents/config/llm.yaml so the model endpoint is configured in one
place for the entire system.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# Make the shared config loader importable
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent.parent / "config")
if _CONFIG_DIR not in sys.path:
    sys.path.insert(0, _CONFIG_DIR)

from llm_config import load_llm_defaults  # noqa: E402

_LLM_DEFAULTS = load_llm_defaults()


# ---------------------------------------------------------------------------
# Agent configuration (from agents.yaml)
# ---------------------------------------------------------------------------


class AgentConfig(BaseModel):
    """Configuration for a single agent."""

    name: str
    description: str
    endpoint: str  # MCP server URL, e.g. "http://localhost:8053"
    protocol: str = "mcp"
    capabilities: list[str] = Field(default_factory=list)
    max_iterations: int = 5
    timeout_seconds: int = 120
    auth_token: str | None = None  # Override per-agent; usually from env
    chat_tool: str = "agent_chat"  # MCP tool name for the agent's chat entry point
    mcp_server_name: str | None = None  # MCP server prefix for tool name qualification
    chat_tool_params: dict[str, Any] = Field(default_factory=dict)  # Extra params merged into agent_chat calls


# ---------------------------------------------------------------------------
# Orchestrator configuration
# ---------------------------------------------------------------------------


class OrchestratorConfig(BaseModel):
    """Top-level orchestrator configuration."""

    agents: dict[str, AgentConfig] = Field(default_factory=dict)

    # MCP client settings
    mcp_connect_timeout: int = 10  # seconds
    mcp_request_timeout: int = 120  # seconds

    # Health check settings
    health_check_interval: int = 60  # seconds (0 = disabled)

    # Discovery
    auto_discover: bool = True  # Discover tools at startup

    # Auto-submit: when True, planned workflows are automatically submitted
    # without user confirmation. Intended for dedicated orchestrator instances.
    auto_submit: bool = os.environ.get(
        "ORCH_AUTO_SUBMIT", "false"
    ).lower() == "true"

    # LLM settings (for routing and synthesis)
    # Defaults loaded from shared Agents/config/llm.yaml
    llm_base_url: str = _LLM_DEFAULTS["base_url"]
    llm_api_key: str = _LLM_DEFAULTS["api_key"]
    llm_model: str = _LLM_DEFAULTS["model"]
    llm_temperature: float = _LLM_DEFAULTS["temperature"]
    llm_max_tokens: int = _LLM_DEFAULTS["max_tokens"]
    llm_timeout_seconds: int = _LLM_DEFAULTS["timeout_seconds"]

    # Optional faster/cheaper model for routing decisions.
    # Routing is a simple JSON classification task — a smaller model
    # like gpt41mini or gpt41nano is much faster and still accurate.
    # When set, the orchestrator creates a separate LLM client for routing.
    # When None, the default LLM model is used for routing.
    routing_model: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> OrchestratorConfig:
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f)

        agents = {}
        for key, agent_data in raw.get("agents", {}).items():
            # Allow env var override for auth tokens
            token_env = agent_data.pop("auth_token_env", None)
            if token_env:
                agent_data["auth_token"] = os.environ.get(token_env)
            agents[key] = AgentConfig(**agent_data)

        settings = raw.get("orchestrator", {})
        return cls(agents=agents, **settings)

    @classmethod
    def from_defaults(cls) -> OrchestratorConfig:
        """Load from the default config file location."""
        default_path = Path(__file__).parent.parent / "config" / "agents.yaml"
        if default_path.exists():
            return cls.from_yaml(default_path)
        return cls()
