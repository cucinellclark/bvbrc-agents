"""CGA and Genome Annotation need a real organism and its NCBI taxonomy_id.

taxonomy_id is optional in the specs, but the website never submits without
it and the app does not cope: mshukla's CGA (scientific_name "metagenome", no
taxonomy_id) failed on "Cannot parse 'taxon_id:'"; marcnguy's CGA was sent
scientific_name "bau_sim" (a read-file prefix). See service_status.md.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import shared.agent_dispatch  # noqa: E402, F401  -- sets up agent package paths
from shared.tools import is_needs_user_input_result  # noqa: E402
from bvbrc_mcp.tools import gowe  # noqa: E402
from bvbrc_mcp.tools.gowe import _check_organism  # noqa: E402

CGA = "ComprehensiveGenomeAnalysis"










# --- submit_gowe_job ---------------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    gowe._recent_submissions.clear()
    c = SimpleNamespace(create_submission=AsyncMock(return_value={"id": "sub_1", "state": "PENDING"}))
    monkeypatch.setattr(gowe, "_get_client", lambda config=None: c)
    yield c
    gowe._recent_submissions.clear()


def _cfg():
    return SimpleNamespace(bvbrc_auth_token="un=u@patricbrc.org|t", session_id="s1", workspace_path="/u@patricbrc.org/home")








# --- populate loop: asks instead of failing ---------------------------------

def _reply(content="", tool_calls=None):
    calls = [SimpleNamespace(id=f"c{i}", type="function",
                             function=SimpleNamespace(name=n, arguments=json.dumps(a)))
             for i, (n, a) in enumerate(tool_calls or [])]
    msg = SimpleNamespace(content=content, tool_calls=calls or None, reasoning=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])


@pytest.mark.asyncio
async def test_populate_turns_the_check_into_a_question():
    from service_agent.models import AgentConfig, AgentState
    from service_agent.phases import populate

    need = {"error": "ComprehensiveGenomeAnalysis needs the NCBI taxonomy_id ...",
            "needs_user_input": True, "missing_input": "taxonomy_id", "workflow_id": "wf_cga"}
    question = "Which organism is this genome? I need it to run CGA."
    replies = [
        _reply(tool_calls=[("get_workflow_inputs", {"workflow_id": CGA})]),
        _reply(tool_calls=[("submit_gowe_job", {"workflow_id": CGA, "inputs": {"scientific_name": "bau_sim"}})]),
        _reply(question),
    ]
    cfg = AgentConfig(llm_base_url="http://llm", llm_api_key="k", llm_model="m", execution_mode="execute")
    state = AgentState(query="run CGA on my reads", context={"execution_mode": "execute"})
    with patch.object(populate, "create_client", lambda c: object()), \
         patch.object(populate, "chat_completion", AsyncMock(side_effect=replies)), \
         patch.object(populate, "execute_tool", AsyncMock(side_effect=[{"inputs": []}, need])):
        out = await populate.populate_and_submit("run CGA on my reads", cfg, state)
    assert out.status == "needs_input"
    assert out.question == question


# --- replay: plain message ----------------------------------------------------

@pytest.mark.asyncio
async def test_replay_reports_missing_organism_plainly():
    from service_agent.models import AgentConfig, AgentState
    from service_agent.phases import replay

    need = {"error": "... search_data(collection=\"taxonomy\" ...", "needs_user_input": True,
            "missing_input": "taxonomy_id"}
    actions = [{"tool": "submit_gowe_job", "summary": "ComprehensiveGenomeAnalysis → cga_run",
                "arguments": {"workflow_id": CGA, "inputs": {}}}]
    cfg = AgentConfig(llm_base_url="http://llm", llm_api_key="k", llm_model="m", execution_mode="execute")
    with patch.object(replay, "execute_tool", AsyncMock(return_value=need)):
        out = await replay.replay_blocked_actions(actions, cfg, AgentState(query="q"))
    assert out.status == "error"
    assert "needs the organism" in out.error_message
    assert "search_data" not in out.error_message
