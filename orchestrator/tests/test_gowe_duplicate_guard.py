"""Duplicate-submission guard in submit_gowe_job.

2026-10-09: two "go ahead" messages each submitted the same Taxonomic
Classification (nbowers, jobs 23731569 / 23731575), and one card click became
five concurrent turns that submitted five identical Comparative Systems jobs
(parrello, session feebb83d).  The same chat submitting the same workflow with
the same inputs (output folder aside) inside the window is now refused.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import shared.agent_dispatch  # noqa: E402, F401  -- sets up agent package paths
from shared.tools import gowe, is_already_submitted_result  # noqa: E402

WF = "wf_4965af9b-8715-4cb0-a189-44546214c9ee"
INPUTS = {"srr_ids": ["ERR7916233"], "database": "SILVA", "host_genome": "no_host",
          "output_path": "tc_ERR7916233"}


class _Client:
    def __init__(self, fail_first=False, delay=0.0):
        self.calls = []
        self._fail_first = fail_first
        self._delay = delay

    async def create_submission(self, workflow_id, inputs, auth_token, labels=None):
        self.calls.append(inputs)
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._fail_first and len(self.calls) == 1:
            raise RuntimeError("GoWe 500")
        return {"id": f"sub_{len(self.calls)}", "state": "PENDING"}


@pytest.fixture
def client(monkeypatch):
    gowe._recent_submissions.clear()
    c = _Client()
    monkeypatch.setattr(gowe, "_get_client", lambda config=None: c)
    monkeypatch.setattr(gowe, "_resolve_workflow_ref", AsyncMock(return_value=(WF, None)))
    yield c
    gowe._recent_submissions.clear()


def _cfg(session="s1"):
    return SimpleNamespace(bvbrc_auth_token="un=alice@patricbrc.org|tok",
                           session_id=session, workspace_path="/alice@patricbrc.org/home")


async def _submit(inputs=INPUTS, session="s1"):
    return await gowe.submit_gowe_job(WF, dict(inputs), config=_cfg(session))


@pytest.mark.asyncio
async def test_second_identical_submit_is_refused(client):
    first = await _submit()
    second = await _submit()
    assert first["submission_id"] == "sub_1"
    assert is_already_submitted_result(second)
    assert second["duplicate_of"] == "sub_1"
    assert "NOT submitted again" in second["error"]
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_new_output_folder_name_is_still_a_duplicate(client):
    """The model invents a fresh folder name each turn; that must not let a repeat through."""
    await _submit()
    second = await _submit({**INPUTS, "output_path": "tc_ERR7916233_run2"})
    assert is_already_submitted_result(second)
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_different_inputs_are_not_duplicates(client):
    await _submit()
    other = await _submit({**INPUTS, "database": "Standard"})
    assert other["submission_id"] == "sub_2"


@pytest.mark.asyncio
async def test_other_session_is_not_a_duplicate(client):
    await _submit(session="s1")
    other = await _submit(session="s2")
    assert other["submission_id"] == "sub_2"


@pytest.mark.asyncio
async def test_window_expiry_allows_resubmission(client, monkeypatch):
    await _submit()
    real = gowe.time.monotonic
    monkeypatch.setattr(gowe.time, "monotonic", lambda: real() + gowe.DUPLICATE_SUBMIT_WINDOW_S + 1)
    again = await _submit()
    assert again["submission_id"] == "sub_2"


@pytest.mark.asyncio
async def test_failed_first_attempt_does_not_block_a_retry(monkeypatch):
    gowe._recent_submissions.clear()
    c = _Client(fail_first=True)
    monkeypatch.setattr(gowe, "_get_client", lambda config=None: c)
    monkeypatch.setattr(gowe, "_resolve_workflow_ref", AsyncMock(return_value=(WF, None)))
    first = await _submit()
    assert "error" in first and not is_already_submitted_result(first)
    second = await _submit()
    assert second["submission_id"] == "sub_2"
    gowe._recent_submissions.clear()


@pytest.mark.asyncio
async def test_concurrent_identical_submits_send_one(monkeypatch):
    """parrello's burst: several turns submitting at once."""
    gowe._recent_submissions.clear()
    c = _Client(delay=0.05)
    monkeypatch.setattr(gowe, "_get_client", lambda config=None: c)
    monkeypatch.setattr(gowe, "_resolve_workflow_ref", AsyncMock(return_value=(WF, None)))
    results = await asyncio.gather(*[_submit() for _ in range(5)])
    assert len(c.calls) == 1
    ok = [r for r in results if r.get("submission_id")]
    dups = [r for r in results if is_already_submitted_result(r)]
    assert len(ok) == 1 and len(dups) == 4
    assert all("still being submitted" in d["error"] for d in dups)
    gowe._recent_submissions.clear()


# --- populate loop: a duplicate is not a failure ---------------------------

from service_agent.models import AgentConfig, AgentState, ToolCall  # noqa: E402
from service_agent.phases import populate  # noqa: E402


def _llm_reply(content="", tool_calls=None):
    calls = [SimpleNamespace(id=f"c{i}", type="function",
                             function=SimpleNamespace(name=n, arguments=a))
             for i, (n, a) in enumerate(tool_calls or [])]
    msg = SimpleNamespace(content=content, tool_calls=calls or None, reasoning=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])


@pytest.mark.asyncio
async def test_populate_loop_reports_duplicate_without_counting_a_failure():
    import json
    dup = {"error": "This exact job ... NOT submitted again.", "already_submitted": True,
           "duplicate_of": "sub_1", "workflow_id": WF}
    replies = [
        _llm_reply(tool_calls=[("get_workflow_inputs", json.dumps({"workflow_id": WF}))]),
        _llm_reply(tool_calls=[("submit_gowe_job", json.dumps({"workflow_id": WF, "inputs": INPUTS}))]),
        _llm_reply(content="Your Taxonomic Classification job was submitted successfully."),
    ]
    tool_results = [{"inputs": []}, dup]
    cfg = AgentConfig(llm_base_url="http://llm", llm_api_key="k", llm_model="m",
                      execution_mode="execute")
    state = AgentState(query="run it", context={"execution_mode": "execute"})
    with patch.object(populate, "create_client", lambda c: object()), \
         patch.object(populate, "chat_completion", AsyncMock(side_effect=replies)), \
         patch.object(populate, "execute_tool", AsyncMock(side_effect=tool_results)):
        out = await populate.populate_and_submit("run it", cfg, state)
    assert out.status == "completed"
    assert out.submission_ids == []
    assert "Not submitted again" in out.operation_message
    assert "sub_1" in out.operation_message
