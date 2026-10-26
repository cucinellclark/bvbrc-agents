"""Every job's results go under this chat's folder; the user is told where.

The model picks the folder name: the one the user named ("to the Testing
folder") or a descriptive one.  The code replaces it only when unusable, and
then says why.  Before: the chat folder passed as-is nested chats/<sid>/<sid>
(parrello, Oct 9), and a requested folder was replaced silently (Chris,
dovie, cmao).
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import shared.agent_dispatch  # noqa: E402, F401  -- sets up agent package paths
from bvbrc_mcp.tools import gowe  # noqa: E402
from bvbrc_mcp.tools.gowe import place_output_path  # noqa: E402

SID = "53dc1802-f87a-480b-86db-056da8c30e34"
HOME = "/chris@bvbrc/home"
BASE = f"{HOME}/chats/{SID}"
NOW = datetime(2026, 10, 9, 18, 46, tzinfo=timezone.utc)


def place(requested, ref="Taxonomic Classification"):
    return place_output_path(requested, BASE, SID, ref, now=NOW)




















# --- submit_gowe_job returns the final path and the note -------------------

class _Client:
    async def create_submission(self, workflow_id, inputs, auth_token, labels=None):
        self.inputs = inputs
        return {"id": "sub_1", "state": "PENDING"}


@pytest.fixture
def client(monkeypatch):
    gowe._recent_submissions.clear()
    c = _Client()
    monkeypatch.setattr(gowe, "_get_client", lambda config=None: c)
    monkeypatch.setattr(gowe, "_resolve_workflow_ref", AsyncMock(return_value=("wf_tc", None)))
    yield c
    gowe._recent_submissions.clear()


def _cfg():
    return SimpleNamespace(bvbrc_auth_token="un=chris@bvbrc|tok", session_id=SID, workspace_path=HOME)








# --- the reply carries the note --------------------------------------------

@pytest.mark.asyncio
async def test_populate_appends_output_location_note():
    import json
    from unittest.mock import patch

    from service_agent.models import AgentConfig, AgentState
    from service_agent.phases import populate

    def _reply(content="", tool_calls=None):
        calls = [SimpleNamespace(id=f"c{i}", type="function",
                                 function=SimpleNamespace(name=n, arguments=json.dumps(a)))
                 for i, (n, a) in enumerate(tool_calls or [])]
        msg = SimpleNamespace(content=content, tool_calls=calls or None, reasoning=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])

    ok = {"workflow_id": "wf_tc", "submission_id": "sub_1", "status": "PENDING",
          "output_path": f"{BASE}/Testing",
          "output_path_note": "Copilot keeps every job's results in this chat's workspace folder, ..."}
    cfg = AgentConfig(llm_base_url="http://llm", llm_api_key="k", llm_model="m", execution_mode="execute")
    state = AgentState(query="run it", context={"execution_mode": "execute"})
    replies = [_reply(tool_calls=[("submit_gowe_job", {"workflow_id": "wf_tc", "inputs": {"output_path": f"{HOME}/Testing"}})]),
               _reply("Submitted your Taxonomic Classification job.")]
    with patch.object(populate, "create_client", lambda c: object()), \
         patch.object(populate, "chat_completion", AsyncMock(side_effect=replies)), \
         patch.object(populate, "execute_tool", AsyncMock(side_effect=[ok])):
        out = await populate.populate_and_submit("run it", cfg, state)
    assert "**Output location:** Copilot keeps every job's results" in out.operation_message


def test_submitted_summary_uses_the_final_path():
    from service_agent.models import AgentState, ToolCall
    from service_agent.phases import populate

    state = AgentState(query="q")
    state.submission_ids.append("sub_1")
    state.record_execution(
        tc=ToolCall(id="c0", name="submit_gowe_job",
                    arguments={"workflow_id": "wf", "inputs": {"output_path": f"{HOME}/Testing"}}),
        result={"submission_id": "sub_1", "output_path": f"{BASE}/Testing"},
    )
    assert f"`{BASE}/Testing`" in populate._submitted_summary(state)
