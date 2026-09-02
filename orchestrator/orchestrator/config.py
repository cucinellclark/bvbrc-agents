"""Configuration loading for the orchestrator.

Reads agent definitions from config/agents.yaml and orchestrator settings
(including the internal routing model) from that same file. Structural LLM
defaults — temperature, max_tokens, timeout — are loaded from the shared
Agents/config/llm.yaml. Model / URL / API-key details are NOT loaded here;
they arrive per request via ``llm_override`` and are enforced by
``_resolve_llm`` in ``server.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

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
    endpoint: str | None = None  # MCP server URL; required when protocol is "mcp"
    protocol: str = "mcp"  # "inprocess" or "mcp"
    capabilities: list[str] = Field(default_factory=list)
    max_iterations: int = 1000
    timeout_seconds: int = 120
    auth_token: str | None = None  # Override per-agent; usually from env
    chat_tool: str = "agent_chat"  # MCP tool name; kept for MCP rollback
    mcp_server_name: str | None = None  # MCP server prefix; kept for MCP rollback
    chat_tool_params: dict[str, Any] = Field(default_factory=dict)  # Extra params merged into agent_chat calls

    @model_validator(mode="after")
    def _require_endpoint_for_mcp(self) -> "AgentConfig":
        if self.protocol == "mcp" and not self.endpoint:
            raise ValueError("endpoint is required when protocol is 'mcp'")
        return self


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

    # Structural LLM defaults (loaded from shared Agents/config/llm.yaml).
    # Model / URL / API-key details are NOT stored on the config — they
    # arrive per request via `llm_override`. See _resolve_llm in server.py.
    llm_temperature: float = _LLM_DEFAULTS["temperature"]
    llm_max_tokens: int = _LLM_DEFAULTS["max_tokens"]
    llm_timeout_seconds: int = _LLM_DEFAULTS["timeout_seconds"]

    # Required routing-model config. Routing is an internal JSON classification
    # task that must always succeed regardless of the user's chosen chat model,
    # so the orchestrator maintains its own dedicated LLM client for it.
    # These three fields must be set explicitly in agents.yaml.
    routing_model: str
    routing_base_url: str
    routing_api_key: str

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
