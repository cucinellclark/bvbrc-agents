"""Shared test helpers for the orchestrator suite.

``make_test_config`` builds an ``OrchestratorConfig`` with the routing fields
filled in.  Those fields became required in ``dec38b6`` (the dedicated routing
LLM) and every fixture that built the config by hand failed from then on --
47 failures and 10 errors were "stale fixtures" for weeks (S0 in
current_plans/harness_structural_investments.md).
"""

from __future__ import annotations

from typing import Any

from orchestrator.config import OrchestratorConfig

TEST_ROUTING = {
    "routing_model": "test-routing-model",
    "routing_base_url": "http://routing.test/v1",
    "routing_api_key": "test-key",
}


def make_test_config(**kwargs: Any) -> OrchestratorConfig:
    """``OrchestratorConfig(**kwargs)`` with test routing defaults."""
    return OrchestratorConfig(**{**TEST_ROUTING, **kwargs})


def mock_llm_client():
    """``MagicMock(spec=LLMClient)`` with the ``config`` the router logs from.

    ``config`` is set in ``LLMClient.__init__``, so ``spec=`` alone leaves it
    out, and the router's ``llm.config.model`` raised inside orchestrate().
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from orchestrator.llm.client import LLMClient

    client = MagicMock(spec=LLMClient)
    client.config = SimpleNamespace(model="test-routing-model", base_url="http://routing.test/v1",
                                    api_key="test-key", max_tokens=1024, temperature=0.0)
    return client
