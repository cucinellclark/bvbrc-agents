"""One rule for the router's "direct" answers, plus no canned fallback.

The router model answers "direct" from memory. On Oct 9, after a prompt rule
said not to, it still answered BV-BRC facts wrongly (genome ids "look like
GCF_…", calling 5482.50 a typo; the microbiome pipeline is "16S/SILVA"). And
an empty direct answer reached users as "I can help with that. Could you
provide a bit more detail?" (5+ real requests, Sep).
"""

import json
from types import SimpleNamespace

import pytest

from orchestrator.router.router import _parse_routing_response

REGISTRY = SimpleNamespace(agents={"helpdesk": object(), "data": object(), "service": object()},
                           agent_keys=["helpdesk", "data", "service"])


def _direct(query, text="Sure, here you go."):
    raw = json.dumps({"decision": "direct", "reasoning": "r", "direct_response": text})
    return _parse_routing_response(raw, query, REGISTRY)


@pytest.mark.parametrize("query", ["hi there", "thanks!", "what is a plasmid?",
                                   "how does horizontal gene transfer work"])
def test_small_talk_and_general_biology_stay_direct(query):
    d = _direct(query)
    assert d.decision == "direct" and d.routing_override is None


@pytest.mark.parametrize("query", [
    "who runs BV-BRC?",
    "does bvbrc have a docking service",
    "is PATRIC the same thing?",
    "is 5482.50 a valid genome id?",
    "what is fig|83332.12.peg.3062",
    "what is SRR19542959",
])
def test_bvbrc_references_go_to_helpdesk(query):
    d = _direct(query)
    assert d.decision == "agent"
    assert d.plan.steps[0].agent_key == "helpdesk"
    assert d.plan.steps[0].task == query
    assert d.routing_override == "bvbrc_reference"


@pytest.mark.parametrize("text", [None, "", "   "])
def test_empty_direct_goes_to_helpdesk_not_a_canned_reply(text):
    d = _direct("Yes, please create that group and run the tree pipeline.", text)
    assert d.decision == "agent"
    assert d.plan.steps[0].agent_key == "helpdesk"
    assert d.routing_override == "empty_direct"


def test_unknown_decision_type_without_text_goes_to_helpdesk():
    raw = json.dumps({"decision": "maybe", "reasoning": "?"})
    d = _parse_routing_response(raw, "hmm", REGISTRY)
    assert d.decision == "agent" and d.routing_override == "empty_direct"


def test_no_helpdesk_agent_falls_back_to_direct():
    registry = SimpleNamespace(agents={"data": object()}, agent_keys=["data"])
    raw = json.dumps({"decision": "direct", "direct_response": ""})
    d = _parse_routing_response(raw, "q", registry)
    assert d.decision == "direct"
    assert "Could you provide a bit more detail" not in d.direct_response


# --- B3: the service agent's canned reply is gone ---------------------------

@pytest.mark.asyncio
async def test_service_unknown_intent_runs_the_populate_loop():
    import sys
    from pathlib import Path
    from unittest.mock import AsyncMock, patch
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import shared.agent_dispatch  # noqa: F401
    from service_agent import agent as service_agent
    from service_agent.models import AgentConfig, Intent

    flow = AsyncMock(return_value="ran-populate")
    cfg = AgentConfig(llm_base_url="http://llm", llm_api_key="k", llm_model="m")
    with patch.object(service_agent, "classify_intent", AsyncMock(return_value=Intent(action="unknown"))), \
         patch.object(service_agent, "_run_gowe_flow", flow):
        result = await service_agent.run_agent("what is the md5 of this protein", config=cfg, context={})
    assert result == "ran-populate"


# --- B6: "input_text" is not a context overflow ------------------------------

def test_input_text_error_is_not_reported_as_conversation_too_large():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from shared.agent_dispatch import _classify_agent_error

    msg = _classify_agent_error(ValueError("invalid input_text field in request"), "data")
    assert "grown too large" not in msg
    assert "more data than fits" in _classify_agent_error(ValueError("maximum context length exceeded"), "data")
