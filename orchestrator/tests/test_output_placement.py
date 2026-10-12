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
from shared.tools import gowe  # noqa: E402
from shared.tools.gowe import place_output_path  # noqa: E402

SID = "53dc1802-f87a-480b-86db-056da8c30e34"
HOME = "/chris@bvbrc/home"
BASE = f"{HOME}/chats/{SID}"
NOW = datetime(2026, 10, 9, 18, 46, tzinfo=timezone.utc)


def place(requested, ref="Taxonomic Classification"):
    return place_output_path(requested, BASE, SID, ref, now=NOW)


def test_user_named_folder_is_used_inside_the_chat_folder():
    assert place("Testing") == (f"{BASE}/Testing", None)


def test_descriptive_name_is_used_as_is():
    assert place("ERR7916233_taxonomy") == (f"{BASE}/ERR7916233_taxonomy", None)


def test_path_already_in_the_chat_folder_is_kept():
    assert place(f"{BASE}/Testing/") == (f"{BASE}/Testing", None)


def test_chat_folder_itself_gets_a_generated_subfolder():
    final, note = place(BASE)
    assert final == f"{BASE}/TaxonomicClassification_20261009_1846"
    assert "not a usable folder name" in note and final in note


def test_session_id_as_last_segment_does_not_nest():
    final, _ = place(f"{HOME}/chats/{SID}")
    assert final.count(SID) == 1


def test_missing_name_is_generated_and_explained():
    final, note = place("")
    assert final == f"{BASE}/TaxonomicClassification_20261009_1846"
    assert note.startswith("No output folder name was given")


def test_uuid_workflow_ref_gives_job_prefix():
    final, _ = place("", ref="wf_4965af9b-8715-4cb0-a189-44546214c9ee")
    assert final == f"{BASE}/job_20261009_1846"


@pytest.mark.parametrize("bad", [HOME, f"{HOME}/", "/chris@bvbrc", "chats", "."])
def test_unusable_last_segments_are_replaced(bad):
    final, note = place(bad)
    assert final.startswith(f"{BASE}/TaxonomicClassification_")
    assert note


def test_absolute_path_elsewhere_keeps_its_name_and_explains():
    final, note = place(f"{HOME}/Testing")
    assert final == f"{BASE}/Testing"
    assert "keeps every job's results in this chat's workspace folder" in note
    assert f"`{HOME}/Testing`" in note and "`Testing` was kept" in note


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


@pytest.mark.asyncio
async def test_submit_reports_final_path_and_note(client):
    r = await gowe.submit_gowe_job("Taxonomic Classification",
                                   {"srr_ids": ["ERR7916233"], "output_path": f"{HOME}/Testing"}, config=_cfg())
    assert client.inputs["output_path"] == f"{BASE}/Testing"
    assert r["output_path"] == f"{BASE}/Testing"
    assert "keeps every job's results" in r["output_path_note"]


@pytest.mark.asyncio
async def test_submit_with_plain_name_has_no_note(client):
    r = await gowe.submit_gowe_job("Taxonomic Classification",
                                   {"srr_ids": ["ERR7916233"], "output_path": "Testing"}, config=_cfg())
    assert r["output_path"] == f"{BASE}/Testing"
    assert "output_path_note" not in r


@pytest.mark.asyncio
async def test_workflow_without_output_path_input_gets_none_added(client):
    await gowe.submit_gowe_job("Taxonomic Classification", {"srr_ids": ["ERR7916233"]}, config=_cfg())
    assert "output_path" not in client.inputs


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
