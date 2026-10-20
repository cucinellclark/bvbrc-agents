"""edit_group: remove / add members of an existing group, always after a question.

Before 2026-10-20 the only route was create_group(if_exists="replace") with
the whole remaining list, whose question did not say "replace"; when a tree
job's analysis found a fragmentary genome the agent passed 49 ids to the job
instead of fixing the group (marcnguy, 2026-10-16).
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import shared.agent_dispatch  # noqa: E402, F401  -- sets up agent package paths
from shared.models import BaseAgentConfig, ToolCall, ToolExecution  # noqa: E402
from shared.step_outputs import build_step_outputs  # noqa: E402
from shared.tools import REPLAY_IN_PROGRESS, describe_blocked_action, execute_tool, groups  # noqa: E402
from shared.tools.registry import TOOL_DISPATCH  # noqa: E402
from shared.tools.schemas import TOOL_SCHEMA_MAP  # noqa: E402

MEMBERS = [f"1313.{40000 + i}" for i in range(49)] + ["1313.42242"]
PATH = "/u@patricbrc.org/home/Genome Groups/Strep_Tet_Sus_tree50"
HEADERS = {"Authorization": "t"}


def _fakes(members=MEMBERS):
    gf = SimpleNamespace(
        resolve_group_path=AsyncMock(return_value={"name": "Strep_Tet_Sus_tree50", "path": PATH, "match_type": "exact"}),
        _workspace_get=AsyncMock(return_value={"metadata": {"name": "Strep_Tet_Sus_tree50"},
                                               "data": json.dumps({"id_list": {"genome_id": list(members)}})}),
    )
    api = SimpleNamespace(acall=AsyncMock(return_value=[[["Strep_Tet_Sus_tree50"]]]))
    return gf, api


def _patched(gf, api):
    return patch.multiple(groups, get_group_functions=lambda *_: gf, _get_api=lambda *_: api)


def _cfg(mode="plan", agent="data"):
    return BaseAgentConfig(llm_base_url="http://x", llm_api_key="k", llm_model="m",
                           execution_mode=mode, agent_key=agent, bvbrc_auth_token="t")


ARGS = {"group": "Strep_Tet_Sus_tree50", "group_type": "genome_group", "remove_ids": ["1313.42242"]}


# ---- planning the edit ------------------------------------------------------------

async def test_plan_reads_the_group_and_describes_the_change():
    gf, api = _fakes()
    with _patched(gf, api):
        plan = await groups.plan_group_edit(ARGS, None, HEADERS)
    assert plan["remove"] == ["1313.42242"] and len(plan["final_ids"]) == 49 and len(plan["previous_ids"]) == 50
    assert "1313.42242" not in plan["final_ids"]
    assert groups.edit_summary(plan) == "genome_group 'Strep_Tet_Sus_tree50': remove 1 (50 → 49): 1313.42242"
    gf.resolve_group_path.assert_awaited_once()
    api.acall.assert_not_awaited()  # planning writes nothing


async def test_a_path_is_used_as_given_and_adds_skip_members_already_present():
    gf, api = _fakes()
    with _patched(gf, api):
        plan = await groups.plan_group_edit(
            {"group": PATH, "group_type": "genome_group", "add_ids": ["1313.1", "1313.42242"]}, None, HEADERS)
    gf.resolve_group_path.assert_not_awaited()
    assert plan["add"] == ["1313.1"] and plan["already_present"] == ["1313.42242"] and len(plan["final_ids"]) == 51
    assert groups.edit_summary(plan).startswith("genome_group 'Strep_Tet_Sus_tree50': add 1 (50 → 51)")


@pytest.mark.parametrize("args, error_type", [
    ({"group": "g", "group_type": "genome_group", "remove_ids": ["1313.99999"]}, "NOT_IN_GROUP"),
    ({"group": "g", "group_type": "genome_group", "add_ids": ["1313.42242"]}, "NO_CHANGE"),
    ({"group": "g", "group_type": "genome_group", "remove_ids": MEMBERS}, "EMPTY_GROUP"),
    ({"group": "g", "group_type": "genome_group"}, "INVALID_PARAMETERS"),
    ({"group": "g", "group_type": "genome_group", "remove_ids": ["not-an-id"]}, "INVALID_PARAMETERS"),
    ({"group": "g", "group_type": "plasmid_group", "remove_ids": ["1313.1"]}, "INVALID_PARAMETERS"),
])
async def test_an_edit_that_cannot_be_what_the_user_meant_is_refused(args, error_type):
    gf, api = _fakes()
    with _patched(gf, api):
        plan = await groups.plan_group_edit(args, None, HEADERS)
    assert plan["errorType"] == error_type
    api.acall.assert_not_awaited()


# ---- the write ---------------------------------------------------------------------

async def test_edit_group_writes_the_group_back_with_overwrite():
    gf, api = _fakes()
    with _patched(gf, api):
        result = await groups.edit_group(**ARGS, headers=HEADERS)
    params = api.acall.await_args.args
    assert params[0] == "Workspace.create"
    payload = params[1][0]
    assert payload["overwrite"] == 1
    path, kind, _meta, content = payload["objects"][0]
    assert (path, kind) == (PATH, "genome_group")
    assert len(content["id_list"]["genome_id"]) == 49 and "1313.42242" not in content["id_list"]["genome_id"]
    assert result["action"] == "edited" and result["previous_count"] == 50 and result["count"] == 49
    assert result["removed"] == ["1313.42242"] and result["path"] == PATH
    assert "removed 1 (50 → 49 members)" in result["message"]


# ---- the gate: it asks in every mode -------------------------------------------------

async def test_plan_mode_asks_with_the_real_before_and_after_counts():
    gf, api = _fakes()
    with _patched(gf, api):
        result = await execute_tool("edit_group", dict(ARGS), TOOL_DISPATCH, config=_cfg("plan"), headers=HEADERS)
    assert result["blocked_by_mode"] is True
    assert result["summary"] == "genome_group 'Strep_Tet_Sus_tree50': remove 1 (50 → 49): 1313.42242"
    api.acall.assert_not_awaited()


async def test_execute_mode_asks_too_and_a_replay_runs_it():
    gf, api = _fakes()
    with _patched(gf, api):
        blocked = await execute_tool("edit_group", dict(ARGS), TOOL_DISPATCH, config=_cfg("execute"), headers=HEADERS)
        assert blocked["blocked_by_mode"] is True
        api.acall.assert_not_awaited()
        token = REPLAY_IN_PROGRESS.set(True)
        try:
            ran = await execute_tool("edit_group", dict(ARGS), TOOL_DISPATCH, config=_cfg("execute"), headers=HEADERS)
        finally:
            REPLAY_IN_PROGRESS.reset(token)
    assert ran["action"] == "edited" and api.acall.await_count == 1


async def test_an_impossible_edit_is_an_error_not_a_question():
    gf, api = _fakes()
    args = {**ARGS, "remove_ids": ["1313.99999"]}
    with _patched(gf, api):
        result = await execute_tool("edit_group", args, TOOL_DISPATCH, config=_cfg("plan"), headers=HEADERS)
    assert result["errorType"] == "NOT_IN_GROUP" and not result.get("blocked_by_mode")


async def test_other_agents_hand_the_call_to_the_data_agent():
    gf, api = _fakes()
    with _patched(gf, api):
        result = await execute_tool("edit_group", dict(ARGS), TOOL_DISPATCH, config=_cfg("plan", "analysis"), headers=HEADERS)
    assert result.get("handoff_to") == "data"


def test_wiring():
    assert "edit_group" in TOOL_SCHEMA_MAP and "edit_group" in TOOL_DISPATCH
    assert TOOL_SCHEMA_MAP["edit_group"]["function"]["parameters"]["required"] == ["group", "group_type"]
    assert describe_blocked_action("edit_group", ARGS) == "genome_group 'Strep_Tet_Sus_tree50': remove 1"
    assert describe_blocked_action("edit_group", {**ARGS, "add_ids": ["1.1", "2.2"]}).endswith("remove 1 and add 2")


def test_step_outputs_record_the_edited_group():
    ex = ToolExecution(tool_call=ToolCall(id="c1", name="edit_group", arguments=ARGS),
                       result={"name": "Strep_Tet_Sus_tree50", "path": PATH, "group_type": "genome_group",
                               "count": 49, "removed": ["1313.42242"], "added": []},
                       error=None, duration_ms=1.0)
    out = build_step_outputs([ex])
    assert out["groups"] == [{"name": "Strep_Tet_Sus_tree50", "path": PATH, "type": "genome_group", "count": 49,
                              "selection": {"removed": 1, "added": 0}}]


# ---- replay -----------------------------------------------------------------------------

async def test_yes_replays_the_edit_and_a_failed_edit_stops_the_set():
    from service_agent.models import AgentConfig, AgentState
    from service_agent.phases.replay import replay_blocked_actions

    cfg = AgentConfig(llm_base_url="http://x", llm_api_key="k", llm_model="m", execution_mode="execute",
                      agent_key="service", bvbrc_auth_token="t")
    edit = {"tool": "edit_group", "summary": "genome_group 'Strep_Tet_Sus_tree50': remove 1 (50 → 49): 1313.42242",
            "arguments": ARGS}
    gf, api = _fakes()
    with _patched(gf, api):
        state = await replay_blocked_actions([edit], cfg, AgentState(query="Yes", context={}))
    assert state.status == "completed" and api.acall.await_count == 1
    assert "Updated group **Strep_Tet_Sus_tree50** (49 items)" in state.operation_message

    bad = {**edit, "arguments": {**ARGS, "remove_ids": ["1313.99999"]}}
    job = {"tool": "submit_gowe_job", "summary": "Bacterial Genome Tree → tree", "arguments": {"workflow_id": "wf", "inputs": {}}}
    gf, api = _fakes()
    submit = AsyncMock(return_value={"submission_id": "sub_1"})
    with _patched(gf, api), patch.dict(TOOL_DISPATCH, {"submit_gowe_job": submit}):
        state = await replay_blocked_actions([bad, job], cfg, AgentState(query="Yes", context={}))
    submit.assert_not_awaited()
    assert state.status == "error" and "Not attempted after that" in state.error_message
