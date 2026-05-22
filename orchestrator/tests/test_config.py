"""Tests for configuration loading."""

import pytest
import tempfile
from pathlib import Path

from orchestrator.config import OrchestratorConfig, AgentConfig


def test_load_from_yaml(tmp_path):
    """Test loading config from a YAML file."""
    yaml_content = """
agents:
  test_agent:
    name: "Test Agent"
    description: "A test agent"
    endpoint: "http://localhost:9999"
    capabilities:
      - testing
    max_iterations: 3
    timeout_seconds: 30

orchestrator:
  mcp_connect_timeout: 5
  health_check_interval: 30
"""
    config_file = tmp_path / "agents.yaml"
    config_file.write_text(yaml_content)

    config = OrchestratorConfig.from_yaml(config_file)

    assert "test_agent" in config.agents
    agent = config.agents["test_agent"]
    assert agent.name == "Test Agent"
    assert agent.endpoint == "http://localhost:9999"
    assert agent.capabilities == ["testing"]
    assert agent.max_iterations == 3
    assert agent.timeout_seconds == 30
    assert config.mcp_connect_timeout == 5
    assert config.health_check_interval == 30


def test_missing_config_file():
    """Test that missing config file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        OrchestratorConfig.from_yaml("/nonexistent/path.yaml")


def test_empty_config(tmp_path):
    """Test loading an empty config file."""
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("agents: {}\n")

    config = OrchestratorConfig.from_yaml(config_file)
    assert config.agents == {}


def test_agent_config_defaults():
    """Test AgentConfig default values."""
    agent = AgentConfig(
        name="Test",
        description="A test agent",
        endpoint="http://localhost:1234",
    )
    assert agent.protocol == "mcp"
    assert agent.capabilities == []
    assert agent.max_iterations == 5
    assert agent.timeout_seconds == 120
    assert agent.auth_token is None
