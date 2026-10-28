"""The GoWe server is a deployment setting, never a hard-coded address.

agents.yaml ``orchestrator.gowe_url`` (or ``GOWE_URL``) → startup
``set_gowe_url`` → every agent config. Before 2026-10-28 the address of the
dev GoWe was written into ``BaseAgentConfig`` and ``bvbrc_mcp.tools.gowe``,
so an orchestrator on another host would have sent its jobs there.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from orchestrator import server
from orchestrator.config import OrchestratorConfig
from shared import agent_dispatch
from shared.agent_dispatch import _build_config_kwargs, set_gowe_url

AGENTS_YAML = Path(server.__file__).parent.parent / "config" / "agents.yaml"

_YAML = """
agents: {}
orchestrator:
  routing_model: "r"
  routing_base_url: "http://routing.test/v1"
  routing_api_key: "k"
"""


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.delenv("GOWE_URL", raising=False)
    monkeypatch.setattr(agent_dispatch, "_GOWE_URL", None)


def test_yaml_value_is_loaded(tmp_path):
    f = tmp_path / "a.yaml"
    f.write_text(_YAML + '  gowe_url: "http://gowe.yaml:7000"\n')
    assert OrchestratorConfig.from_yaml(f).gowe_url == "http://gowe.yaml:7000"


def test_env_overrides_yaml(tmp_path, monkeypatch):
    f = tmp_path / "a.yaml"
    f.write_text(_YAML + '  gowe_url: "http://gowe.yaml:7000"\n')
    monkeypatch.setenv("GOWE_URL", "http://127.0.0.1:7000")
    assert OrchestratorConfig.from_yaml(f).gowe_url == "http://127.0.0.1:7000"


def test_checked_in_config_names_a_gowe():
    assert OrchestratorConfig.from_yaml(AGENTS_YAML).gowe_url


def test_startup_refuses_without_gowe(tmp_path, monkeypatch):
    monkeypatch.setenv("COPILOT_INTERNAL_KEY", "k")
    f = tmp_path / "a.yaml"
    f.write_text(_YAML)
    app = server.create_app(str(f))
    with pytest.raises(RuntimeError, match="GoWe"):
        with TestClient(app):
            pass


def test_configured_url_reaches_agent_config():
    set_gowe_url("http://127.0.0.1:7000")
    assert _build_config_kwargs("t", {})["gowe_url"] == "http://127.0.0.1:7000"


def test_request_context_cannot_redirect_gowe():
    # The user's token goes to this address: a context value must not move it.
    set_gowe_url("http://127.0.0.1:7000")
    kwargs = _build_config_kwargs("t", {"gowe_url": "http://attacker.example"})
    assert kwargs["gowe_url"] == "http://127.0.0.1:7000"


def test_env_is_the_fallback_for_other_callers(monkeypatch):
    monkeypatch.setenv("GOWE_URL", "http://gowe.env:7000")
    assert _build_config_kwargs("t", {})["gowe_url"] == "http://gowe.env:7000"


def test_nothing_configured_means_no_gowe():
    assert "gowe_url" not in _build_config_kwargs("t", {})
    from shared.models import BaseAgentConfig
    assert BaseAgentConfig.model_fields["gowe_url"].default is None
