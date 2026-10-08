"""Gateway shared secret, no default token, and the LLM host allowlist.

Drives the real FastAPI app without its lifespan (no agent discovery, no
LLM): _state is filled in by hand and orchestrate_to_response is replaced,
so a request that clears every check ends in a recognisable 500.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from orchestrator import server
from orchestrator.config import OrchestratorConfig
from orchestrator.internal_auth import (
    INTERNAL_KEY_HEADER,
    internal_key_matches,
    llm_host_allowed,
    load_internal_key,
)

KEY = "test-internal-key"
TOKEN = "un=alice@patricbrc.org|tokenid=x|expiry=9999999999|sig=y"


def _body(base_url: str = "http://mango.cels.anl.gov:8004/v1", **extra) -> dict:
    body = {
        "query": "hello",
        "llm_override": {"base_url": base_url, "api_key": "EMPTY", "model": "m"},
    }
    body.update(extra)
    return body


@pytest.fixture
def client(monkeypatch):
    cfg = OrchestratorConfig(
        routing_model="r",
        routing_base_url="http://mango.cels.anl.gov:8004/v1",
        routing_api_key="k",
        llm_allowed_hosts=["mango.cels.anl.gov"],
    )
    monkeypatch.setattr(server._state, "config", cfg)
    monkeypatch.setattr(server._state, "internal_key", KEY)
    monkeypatch.setattr(server._state, "ready", True)
    monkeypatch.setattr(server._state, "registry", SimpleNamespace(agents={}))
    monkeypatch.setattr(
        server._state, "routing_llm", SimpleNamespace(config=SimpleNamespace(model="r", base_url="u"))
    )
    monkeypatch.setattr(server._state, "llm_cache", {})
    # A token file next to the package used to be picked up as a default.
    monkeypatch.setenv("BV_BRC_AUTH_TOKEN", "operator-token-must-not-be-used")

    async def reached_orchestration(request, *args, **kwargs):
        raise RuntimeError(f"reached orchestration as {request.auth_token}")

    monkeypatch.setattr(server, "orchestrate_to_response", reached_orchestration)
    return TestClient(server.create_app())


# --- internal key -----------------------------------------------------------


def test_health_needs_no_key(client):
    assert client.get("/health").status_code != 401


@pytest.mark.parametrize("path", ["/orchestrate", "/orchestrate/stream"])
def test_missing_key_is_rejected(client, path):
    r = client.post(path, json=_body(auth_token=TOKEN))
    assert r.status_code == 401
    assert "internal key" in r.json()["detail"]


def test_wrong_key_is_rejected(client):
    r = client.post(
        "/orchestrate",
        json=_body(auth_token=TOKEN),
        headers={INTERNAL_KEY_HEADER: "wrong"},
    )
    assert r.status_code == 401


def test_agents_catalog_needs_key(client):
    assert client.get("/agents").status_code == 401


# --- no default token -------------------------------------------------------


def test_no_token_is_rejected_even_with_env_default(client):
    r = client.post("/orchestrate", json=_body(), headers={INTERNAL_KEY_HEADER: KEY})
    assert r.status_code == 401
    assert "auth token" in r.json()["detail"]


def test_body_token_is_used(client):
    r = client.post(
        "/orchestrate", json=_body(auth_token=TOKEN), headers={INTERNAL_KEY_HEADER: KEY}
    )
    assert r.status_code == 500
    assert f"reached orchestration as {TOKEN}" in r.json()["detail"]


def test_bearer_header_token_is_used(client):
    r = client.post(
        "/orchestrate",
        json=_body(),
        headers={INTERNAL_KEY_HEADER: KEY, "Authorization": f"Bearer {TOKEN}"},
    )
    assert r.status_code == 500
    assert f"reached orchestration as {TOKEN}" in r.json()["detail"]


def test_empty_bearer_is_rejected(client):
    r = client.post(
        "/orchestrate",
        json=_body(),
        headers={INTERNAL_KEY_HEADER: KEY, "Authorization": "Bearer "},
    )
    assert r.status_code == 401


# --- LLM host allowlist -----------------------------------------------------


@pytest.mark.parametrize(
    "base_url",
    [
        "http://attacker.example.com/v1",
        "http://mango.cels.anl.gov.attacker.example.com/v1",
        "file:///etc/passwd",
        "http://169.254.169.254/latest",
    ],
)
def test_disallowed_llm_host_is_rejected(client, base_url):
    r = client.post(
        "/orchestrate",
        json=_body(base_url=base_url, auth_token=TOKEN),
        headers={INTERNAL_KEY_HEADER: KEY},
    )
    assert r.status_code == 400
    assert "llm_allowed_hosts" in r.json()["detail"]


def test_disallowed_llm_host_is_rejected_on_stream(client):
    r = client.post(
        "/orchestrate/stream",
        json=_body(base_url="http://attacker.example.com/v1", auth_token=TOKEN),
        headers={INTERNAL_KEY_HEADER: KEY},
    )
    assert r.status_code == 400


def test_allowed_host_any_port_passes(client):
    r = client.post(
        "/orchestrate",
        json=_body(base_url="http://MANGO.cels.anl.gov:8003/v1", auth_token=TOKEN),
        headers={INTERNAL_KEY_HEADER: KEY},
    )
    assert r.status_code == 500  # got past the allowlist into orchestration


def test_llm_host_allowed_helper():
    hosts = ["mango.cels.anl.gov", " apps.inside.anl.gov "]
    assert llm_host_allowed("https://apps.inside.anl.gov/argoapi/v1", hosts)
    assert not llm_host_allowed("ftp://mango.cels.anl.gov/v1", hosts)
    assert not llm_host_allowed("not a url", hosts)
    assert not llm_host_allowed("http://mango.cels.anl.gov/v1", [])


def test_shipped_config_allows_current_model_hosts():
    cfg = OrchestratorConfig.from_defaults()
    for url in (
        "http://mango.cels.anl.gov:8004/v1",
        "https://apps.inside.anl.gov/argoapi/v1",
        cfg.routing_base_url,
    ):
        assert llm_host_allowed(url, cfg.llm_allowed_hosts), url


# --- key loading / startup ----------------------------------------------------


def test_key_matches_is_strict():
    assert internal_key_matches("abc", "abc")
    assert not internal_key_matches("abc", "abd")
    assert not internal_key_matches(None, "abc")
    assert not internal_key_matches("abc", None)
    assert not internal_key_matches("", "")


def test_load_key_env_beats_file(monkeypatch, tmp_path):
    f = tmp_path / "key"
    f.write_text("from-file\n")
    monkeypatch.setenv("COPILOT_INTERNAL_KEY_FILE", str(f))
    monkeypatch.setenv("COPILOT_INTERNAL_KEY", "from-env")
    assert load_internal_key() == "from-env"
    monkeypatch.delenv("COPILOT_INTERNAL_KEY")
    assert load_internal_key() == "from-file"


def test_load_key_missing_file(monkeypatch, tmp_path):
    monkeypatch.delenv("COPILOT_INTERNAL_KEY", raising=False)
    monkeypatch.setenv("COPILOT_INTERNAL_KEY_FILE", str(tmp_path / "absent"))
    assert load_internal_key() is None


def test_startup_refuses_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("COPILOT_INTERNAL_KEY", raising=False)
    monkeypatch.setenv("COPILOT_INTERNAL_KEY_FILE", str(tmp_path / "absent"))
    config_path = Path(server.__file__).parent.parent / "config" / "agents.yaml"
    app = server.create_app(str(config_path))
    with pytest.raises(RuntimeError, match="shared secret"):
        with TestClient(app):
            pass
