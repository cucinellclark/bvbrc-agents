"""D4 Phase 2: ``rerun_jobs`` prepares a rerun of earlier jobs, with no LLM.

Folder or task ids -> original app + parameters (AppService.query_tasks) ->
GoWe workflow (``steps[].hints.bvbrc_app_id``) -> inputs copied verbatim (declared
inputs only), output under the chat folder as ``<name>_rerun``.  The result's
``prepared_actions`` become ``blocked_actions`` through ``record_execution``,
so the user confirms the set with the same one question as any job set, and
Yes replays it verbatim.  Nothing is submitted by the tool.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import shared.agent_dispatch  # noqa: E402, F401  -- sets up agent package paths
from shared import agent_dispatch  # noqa: E402
from shared.models import BaseAgentState, ToolCall  # noqa: E402
from shared.tools import (  # noqa: E402
    HANDOFF_TARGETS,
    TOOL_AGENT_ALLOWLIST,
    TOOL_RESULT_CHAR_LIMITS,
    TOOL_TIMEOUT_OVERRIDES,
    EXECUTE_ONLY_TOOLS,
    execute_tool,
)
from shared.tools import gowe, rerun  # noqa: E402
from shared.tools.registry import TOOL_DISPATCH  # noqa: E402
from shared.tools.schemas import ALL_TOOL_SCHEMAS  # noqa: E402
from service_agent import agent as service_agent  # noqa: E402
from service_agent.models import AgentConfig, AgentState, Intent  # noqa: E402
from service_agent.phases import populate  # noqa: E402
from service_agent.phases import replay as replay_mod  # noqa: E402
from service_agent.prompts.populate import build_populate_prompt  # noqa: E402

USER = "alice@patricbrc.org"
HOME = f"/{USER}/home"
SID = "s-123"
CHAT = f"{HOME}/chats/{SID}"
TOKEN = f"un={USER}|tokenid=x|sig=y"


# ---------------------------------------------------------------------------
# Fakes: GoWe catalog, workspace listing, AppService.query_tasks
# ---------------------------------------------------------------------------

WORKFLOWS = {
    "wf_tc": ("Taxonomic Classification", "TaxonomicClassification",
              ["srr_libs", "paired_end_libs", "database", "host_genome", "analysis_type",
               "output_path", "output_file"]),
    "wf_asm": ("GenomeAssembly", "GenomeAssembly2",
               ["srr_ids", "recipe", "trim", "output_path", "output_file"]),
    "wf_ts": ("Tree Sort", "TreeSort", ["input_fasta_file", "output_path", "output_file"]),
    "wf_ann": ("GenomeAnnotation", None, ["contigs", "output_path", "output_file"]),
}


class FakeGoWe:
    def __init__(self):
        self.detail_calls = 0
        self.inputs_calls = 0
        self.refuse = {}
        self.dry_runs = []

    async def list_workflows(self, auth_token=None, limit=50, offset=0):
        return [{"id": i, "name": n} for i, (n, _, _) in WORKFLOWS.items()], {}

    async def get_workflow(self, workflow_id, auth_token=None):
        self.detail_calls += 1
        _, app, _ = WORKFLOWS[workflow_id]
        # The real shape (GoWe StepHints): the app id is under hints.
        step = {"id": "run_tool", "hints": {"executor": "bvbrc" if app else "local"}}
        if app:
            step["hints"]["bvbrc_app_id"] = app
        return {"id": workflow_id, "steps": [step]}

    async def get_workflow_inputs(self, workflow_id, auth_token=None):
        self.inputs_calls += 1
        return [{"id": k, "type": "string"} for k in WORKFLOWS[workflow_id][2]]

    refuse: dict = {}
    dry_runs: list = []

    async def dry_run(self, workflow_id, inputs, auth_token=None):
        self.dry_runs.append((workflow_id, inputs))
        why = self.refuse.get(json.dumps(inputs.get("srr_ids") or inputs.get("srr_libs")))
        if why == "raise":
            raise RuntimeError("GoWe down")
        if why:
            return {"dry_run": True, "valid": False, "inputs_valid": False,
                    "errors": [{"field": "inputs", "message": why}]}
        # An unavailable executor makes "valid" false without any input problem.
        return {"dry_run": True, "valid": False, "inputs_valid": True,
                "executor_availability": {"bvbrc": "unavailable"}}


def _item(rel, task_id, app, success):
    path = f"{HOME}/runs/{rel}"
    return {"name": path.rsplit("/", 1)[-1], "type": "job_result", "path": path,
            "job": {"task_id": task_id, "app": app, "success": success}}


def _tc_params(srr, folder, out):
    return {"srr_libs": [{"srr_accession": srr, "sample_id": srr}], "database": "standard",
            "host_genome": "homo_sapiens", "analysis_type": "microbiome",
            "output_path": f"{HOME}/runs/{folder}", "output_file": out,
            "_preflight": {"cpu": 6}, "save_classified_sequences": False}


def _asm_params(srr, folder, out):
    return {"srr_ids": [srr], "recipe": "auto", "trim": False,
            "output_path": f"{HOME}/runs/{folder}", "output_file": out,
            "_preflight": {"cpu": 12}, "pilon_iter": 2}


def _task(task_id, app, params, status="completed"):
    return {"id": int(task_id), "app": app, "application_name": app,
            "parameters": params, "status": status}


def _query_tasks(tasks):
    by_id = {str(t["id"]): t for t in tasks}

    async def fake(task_ids, token, **kw):
        return {str(i): by_id[str(i)] for i in task_ids if str(i) in by_id}

    return AsyncMock(side_effect=fake)


def _browse(items):
    return AsyncMock(return_value={"result": {"items": items, "count": len(items)}})


def _cfg(session=True):
    return SimpleNamespace(
        bvbrc_auth_token=TOKEN, gowe_url="http://gowe-test", mcp_server_path=None,
        session_id=SID if session else None, workspace_path=HOME if session else None,
    )


@pytest.fixture(autouse=True)
def fresh_caches():
    gowe._app_workflow_cache.clear()
    rerun._inputs_cache.clear()
    yield
    gowe._app_workflow_cache.clear()
    rerun._inputs_cache.clear()


async def _rerun(items=None, tasks=(), fake=None, browse=None, **kwargs):
    fake = fake or FakeGoWe()
    browse = browse if browse is not None else _browse(items or [])
    qt = _query_tasks(list(tasks))
    with patch.object(gowe, "_get_client", lambda config=None: fake), \
         patch.object(rerun, "workspace_browse", browse), \
         patch.object(rerun, "query_app_tasks", qt):
        kwargs.setdefault("config", _cfg())
        out = await rerun.rerun_jobs(**kwargs)
    return out, browse, qt, fake


THREE_ITEMS = [
    _item("a/SRR1_TC/SRR1", "101", "TaxonomicClassification", 1),
    _item("a/SRR2_TC/SRR2", "102", "TaxonomicClassification", 0),
    _item("b/SRR3_asm/SRR3", "103", "GenomeAssembly2", 1),
]
THREE_TASKS = [
    _task("101", "TaxonomicClassification", _tc_params("SRR1000001", "a/SRR1_TC", "SRR1")),
    _task("102", "TaxonomicClassification", _tc_params("SRR1000002", "a/SRR2_TC", "SRR2")),
    _task("103", "GenomeAssembly2", _asm_params("SRR1000003", "b/SRR3_asm", "SRR3")),
]


# ---------------------------------------------------------------------------
# rerun_jobs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_folder_with_three_jobs_prepares_three_and_marks_the_failed_one():
    out, browse, qt, _ = await _rerun(THREE_ITEMS, THREE_TASKS, folder=f"{HOME}/runs")
    assert browse.await_args.kwargs["workspace_types"] == ["job_result"]
    assert sorted(qt.await_args.args[0]) == ["101", "102", "103"]
    actions = out["prepared_actions"]
    assert len(actions) == 3
    assert all(a["tool"] == "submit_gowe_job" and a["origin"] == "rerun_jobs" for a in actions)
    by_task = {a["original"]["task_id"]: a for a in actions}
    assert by_task["102"]["original"]["success"] is False
    assert "(original failed)" in by_task["102"]["summary"]
    assert "(original failed)" not in by_task["101"]["summary"]
    assert by_task["101"]["summary"].startswith("Taxonomic Classification ← job 101")
    assert by_task["103"]["arguments"]["workflow_id"] == "wf_asm"
    assert "nothing" in out["note"].lower()


@pytest.mark.asyncio
async def test_inputs_are_the_declared_original_parameters_with_dropped_keys_reported():
    out, *_ = await _rerun(THREE_ITEMS, THREE_TASKS, folder=f"{HOME}/runs")
    tc = next(a for a in out["prepared_actions"] if a["original"]["task_id"] == "101")
    inputs = tc["arguments"]["inputs"]
    original = THREE_TASKS[0]["parameters"]
    for key in ("srr_libs", "database", "host_genome", "analysis_type"):
        assert inputs[key] == original[key]
    assert "save_classified_sequences" not in inputs and "_preflight" not in inputs
    # Undeclared keys are reported; internal _keys are not noise for the user.
    assert tc["dropped_inputs"] == ["save_classified_sequences"]


@pytest.mark.asyncio
async def test_output_goes_to_the_chat_folder_with_a_rerun_suffix():
    out, *_ = await _rerun(THREE_ITEMS, THREE_TASKS, folder=f"{HOME}/runs")
    tc = next(a for a in out["prepared_actions"] if a["original"]["task_id"] == "101")
    inputs = tc["arguments"]["inputs"]
    assert inputs["output_path"] == f"{CHAT}/SRR1_TC"
    assert inputs["output_file"] == "SRR1_rerun"
    # Placement is idempotent: the replayed submit keeps the same folder.
    final, note = gowe.place_output_path(inputs["output_path"], CHAT, SID, "wf_tc")
    assert (final, note) == (inputs["output_path"], None)


@pytest.mark.asyncio
async def test_two_jobs_with_the_same_output_name_get_distinct_rerun_names():
    items = [_item("x/TC/S", "201", "TaxonomicClassification", 1),
             _item("y/TC/S", "202", "TaxonomicClassification", 1)]
    tasks = [_task("201", "TaxonomicClassification", _tc_params("SRR2000001", "x/TC", "S")),
             _task("202", "TaxonomicClassification", _tc_params("SRR2000002", "y/TC", "S"))]
    out, *_ = await _rerun(items, tasks, folder=f"{HOME}/runs")
    outs = [(a["arguments"]["inputs"]["output_path"], a["arguments"]["inputs"]["output_file"])
            for a in out["prepared_actions"]]
    assert len(set(outs)) == 2


@pytest.mark.asyncio
async def test_identical_jobs_are_collapsed_into_one():
    items = [_item("x/TC/S", "301", "TaxonomicClassification", 1),
             _item("y/TC2/S", "302", "TaxonomicClassification", 1)]
    same = _tc_params("SRR3000001", "x/TC", "S")
    tasks = [_task("301", "TaxonomicClassification", same),
             _task("302", "TaxonomicClassification", {**same, "output_path": f"{HOME}/runs/y/TC2"})]
    out, *_ = await _rerun(items, tasks, folder=f"{HOME}/runs")
    assert len(out["prepared_actions"]) == 1
    assert out["collapsed"] == [{"task_id": "302", "same_as": "301"}]
    assert "identical" in out["note"]


@pytest.mark.asyncio
async def test_more_than_ten_runnable_jobs_is_too_many_and_prepares_nothing():
    items, tasks = [], []
    for i in range(11):
        sub = "p" if i < 6 else "q"
        tid = str(400 + i)
        items.append(_item(f"{sub}/A{i}/S{i}", tid, "GenomeAssembly2", 1))
        tasks.append(_task(tid, "GenomeAssembly2", _asm_params(f"SRR40000{i:02d}", f"{sub}/A{i}", f"S{i}")))
    out, *_ = await _rerun(items, tasks, folder=f"{HOME}/runs")
    assert out["errorType"] == "TOO_MANY"
    assert not out.get("prepared_actions")
    assert out["runnable_count"] == 11
    assert out["subfolder_counts"] == {f"{HOME}/runs/p": 6, f"{HOME}/runs/q": 5}


@pytest.mark.asyncio
async def test_too_many_job_results_in_the_folder_asks_to_narrow():
    items = [_item(f"f/J{i}/S{i}", str(1000 + i), "GenomeAssembly2", 1) for i in range(201)]
    out, _, qt, _ = await _rerun(items, [], folder=f"{HOME}/runs")
    assert out["errorType"] == "TOO_MANY"
    qt.assert_not_awaited()


@pytest.mark.asyncio
async def test_job_results_deeper_than_four_levels_are_left_out():
    items = [_item("a/b/c/d/e/S", "501", "GenomeAssembly2", 1),
             _item("a/S2_asm/S2", "502", "GenomeAssembly2", 1)]
    tasks = [_task("502", "GenomeAssembly2", _asm_params("SRR5000002", "a/S2_asm", "S2"))]
    out, _, qt, _ = await _rerun(items, tasks, folder=f"{HOME}/runs")
    assert qt.await_args.args[0] == ["502"]
    assert len(out["prepared_actions"]) == 1
    assert "deeper" in out["note"]


@pytest.mark.asyncio
async def test_unmapped_and_experimental_apps_are_listed_not_prepared():
    items = THREE_ITEMS + [_item("c/ann/G", "601", "GenomeAnnotation", 1),
                           _item("c/ts/T", "602", "TreeSort", 0)]
    tasks = THREE_TASKS + [
        _task("601", "GenomeAnnotation", {"contigs": "/x.fa", "output_path": f"{HOME}/runs/c/ann", "output_file": "G"}),
        _task("602", "TreeSort", {"input_fasta_file": "/y.fa", "output_path": f"{HOME}/runs/c/ts", "output_file": "T"}),
    ]
    out, *_ = await _rerun(items, tasks, folder=f"{HOME}/runs")
    assert len(out["prepared_actions"]) == 3
    skipped = {s["task_id"]: s["reason"] for s in out["skipped"]}
    assert "cannot rerun here" in skipped["601"]
    assert "experimental" in skipped["602"].lower() or "broken" in skipped["602"].lower()


@pytest.mark.asyncio
async def test_task_ids_path_needs_no_workspace_listing():
    out, browse, qt, _ = await _rerun(tasks=THREE_TASKS, task_ids=["103"])
    browse.assert_not_awaited()
    assert qt.await_args.args[0] == ["103"]
    [action] = out["prepared_actions"]
    assert action["arguments"]["inputs"]["srr_ids"] == ["SRR1000003"]
    assert action["original"] == {"task_id": "103", "app": "GenomeAssembly2", "success": None}


@pytest.mark.asyncio
async def test_unknown_task_id_is_skipped():
    out, *_ = await _rerun(tasks=THREE_TASKS, task_ids=["103", "999"])
    assert len(out["prepared_actions"]) == 1
    assert out["skipped"][0]["task_id"] == "999"


@pytest.mark.asyncio
async def test_identifier_check_failure_is_reported_per_job():
    tasks = list(THREE_TASKS)
    tasks[2] = _task("103", "GenomeAssembly2", _asm_params("SRR1000003.3", "b/SRR3_asm", "SRR3"))
    out, *_ = await _rerun(THREE_ITEMS, tasks, folder=f"{HOME}/runs")
    assert len(out["prepared_actions"]) == 2
    [skip] = out["skipped"]
    assert skip["task_id"] == "103" and "SRR1000003.3" in skip["reason"]


@pytest.mark.asyncio
async def test_include_failed_false_drops_failed_originals():
    out, *_ = await _rerun(THREE_ITEMS, THREE_TASKS, folder=f"{HOME}/runs", include_failed=False)
    assert sorted(a["original"]["task_id"] for a in out["prepared_actions"]) == ["101", "103"]
    assert any(s["task_id"] == "102" and "failed" in s["reason"] for s in out["skipped"])


@pytest.mark.asyncio
async def test_still_running_jobs_are_not_rerun():
    tasks = list(THREE_TASKS)
    tasks[0] = {**tasks[0], "status": "in-progress"}
    out, *_ = await _rerun(THREE_ITEMS, tasks, folder=f"{HOME}/runs")
    assert "101" not in [a["original"]["task_id"] for a in out["prepared_actions"]]
    assert any(s["task_id"] == "101" and "running" in s["reason"] for s in out["skipped"])


@pytest.mark.asyncio
async def test_needs_folder_or_task_ids_and_a_token():
    out, *_ = await _rerun()
    assert out["errorType"] == "INVALID_PARAMETERS"
    cfg = _cfg()
    cfg.bvbrc_auth_token = None
    out, *_ = await _rerun(task_ids=["1"], config=cfg)
    assert out["errorType"] == "AUTHENTICATION_ERROR"


@pytest.mark.asyncio
async def test_workflow_map_and_inputs_are_cached():
    fake = FakeGoWe()
    await _rerun(THREE_ITEMS, THREE_TASKS, fake=fake, folder=f"{HOME}/runs")
    details, inputs = fake.detail_calls, fake.inputs_calls
    assert details == len(WORKFLOWS) and inputs == 2
    await _rerun(THREE_ITEMS, THREE_TASKS, fake=fake, folder=f"{HOME}/runs")
    assert (fake.detail_calls, fake.inputs_calls) == (details, inputs)


@pytest.mark.asyncio
async def test_list_gowe_workflows_surfaces_bvbrc_app_id():
    fake = FakeGoWe()
    with patch.object(gowe, "_get_client", lambda config=None: fake):
        out = await gowe.list_gowe_workflows(config=_cfg())
    by_name = {w["name"]: w for w in out["workflows"]}
    assert by_name["Taxonomic Classification"]["bvbrc_app_id"] == "TaxonomicClassification"
    assert "bvbrc_app_id" not in by_name["GenomeAnnotation"]


@pytest.mark.asyncio
async def test_a_job_gowe_would_refuse_is_skipped_and_a_failed_dry_run_keeps_the_job():
    fake = FakeGoWe()
    fake.refuse = {json.dumps(["SRR1000003"]): 'recipe: "old" is not a permitted value'}
    out, *_ = await _rerun(THREE_ITEMS, THREE_TASKS, fake=fake, folder=f"{HOME}/runs")
    assert len(fake.dry_runs) == 3
    assert sorted(a["original"]["task_id"] for a in out["prepared_actions"]) == ["101", "102"]
    [skip] = out["skipped"]
    assert skip["task_id"] == "103" and "not a permitted value" in skip["reason"]

    fake = FakeGoWe()
    fake.refuse = {json.dumps(["SRR1000003"]): "raise"}
    out, *_ = await _rerun(THREE_ITEMS, THREE_TASKS, fake=fake, folder=f"{HOME}/runs")
    assert len(out["prepared_actions"]) == 3


@pytest.mark.asyncio
async def test_too_many_does_not_dry_run():
    items = [_item(f"p/A{i}/S{i}", str(800 + i), "GenomeAssembly2", 1) for i in range(11)]
    tasks = [_task(str(800 + i), "GenomeAssembly2", _asm_params(f"SRR80000{i:02d}", f"p/A{i}", f"S{i}"))
             for i in range(11)]
    fake = FakeGoWe()
    out, *_ = await _rerun(items, tasks, fake=fake, folder=f"{HOME}/runs")
    assert out["errorType"] == "TOO_MANY" and fake.dry_runs == []


# ---------------------------------------------------------------------------
# Wiring: schema, registry, gates
# ---------------------------------------------------------------------------

def test_32_tools_with_matching_schema_and_dispatch_names():
    names = [s["function"]["name"] for s in ALL_TOOL_SCHEMAS]
    assert len(names) == len(set(names)) == 32
    assert set(names) == set(TOOL_DISPATCH) and len(TOOL_DISPATCH) == 32
    assert "edit_group" in names
    assert "rerun_jobs" in names


def test_rerun_jobs_is_service_only_and_prepares_rather_than_submits():
    assert TOOL_AGENT_ALLOWLIST["rerun_jobs"] == frozenset({"service"})
    assert HANDOFF_TARGETS["rerun_jobs"] == "service"
    assert TOOL_TIMEOUT_OVERRIDES["rerun_jobs"] == 90
    assert "rerun_jobs" in TOOL_RESULT_CHAR_LIMITS
    # It never submits, so the plan gate does not apply to it.
    assert "rerun_jobs" not in EXECUTE_ONLY_TOOLS


@pytest.mark.asyncio
async def test_another_agent_calling_rerun_jobs_hands_off_to_service():
    cfg = SimpleNamespace(agent_key="workspace", handoff_from=None, execution_mode="plan")
    tool = AsyncMock()
    out = await execute_tool("rerun_jobs", {"task_ids": ["1"]}, {"rerun_jobs": tool}, config=cfg)
    tool.assert_not_awaited()
    assert out["handoff_to"] == "service"


def test_record_execution_turns_prepared_actions_into_blocked_actions():
    state = BaseAgentState()
    action = {"tool": "submit_gowe_job", "summary": "GenomeAssembly ← job 1 (x)",
              "arguments": {"workflow_id": "wf_asm", "inputs": {"srr_ids": ["SRR1000001"]}},
              "origin": "rerun_jobs", "original": {"task_id": "1", "app": "GenomeAssembly2", "success": True}}
    state.record_execution(ToolCall(id="c1", name="rerun_jobs", arguments={"task_ids": ["1"]}),
                           result={"prepared_actions": [action], "skipped": []})
    assert state.blocked_actions == [action]


def test_timeout_partial_result_keeps_prepared_actions():
    action = {"tool": "submit_gowe_job", "summary": "GenomeAssembly ← job 1 (x)",
              "arguments": {"workflow_id": "wf_asm", "inputs": {}}, "origin": "rerun_jobs"}
    r = agent_dispatch.timeout_partial_result(
        "service", [{"tool": "rerun_jobs", "arguments": {}, "result": {"prepared_actions": [action]}, "error": None}], 300)
    assert r["status"] == "completed"
    assert r["blocked_actions"][0]["summary"] == action["summary"]
    assert r["blocked_actions"][0]["origin"] == "rerun_jobs"


# ---------------------------------------------------------------------------
# Service agent: the populate loop and run_agent
# ---------------------------------------------------------------------------

RERUN_ACTIONS = [
    {"tool": "submit_gowe_job", "summary": f"GenomeAssembly ← job {i} (A{i})",
     "arguments": {"workflow_id": "wf_asm", "inputs": {"srr_ids": [f"SRR700000{i}"], "output_path": f"{CHAT}/A{i}",
                                                     "output_file": f"S{i}_rerun"}},
     "origin": "rerun_jobs", "original": {"task_id": str(i), "app": "GenomeAssembly2", "success": True}}
    for i in range(1, 4)
]


def _reply(content="", tool_calls=None):
    calls = [SimpleNamespace(id=f"c{i}", type="function",
                             function=SimpleNamespace(name=n, arguments=json.dumps(a)))
             for i, (n, a) in enumerate(tool_calls or [])]
    msg = SimpleNamespace(content=content, tool_calls=calls or None, reasoning=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])


def _agent_cfg(mode):
    return AgentConfig(llm_base_url="http://llm", llm_api_key="k", llm_model="m",
                       execution_mode=mode, max_iterations=20)


async def _populate(replies, rerun_result, mode="plan", multi_job=True):
    state = AgentState(query="rerun my jobs", context={"execution_mode": mode}, multi_job=multi_job)
    tool = AsyncMock(return_value=rerun_result)
    submit = AsyncMock(side_effect=AssertionError("must not submit"))
    chat = AsyncMock(side_effect=list(replies))
    with patch.object(populate, "create_client", lambda c: MagicMock()), \
         patch.object(populate, "chat_completion", chat), \
         patch.dict(populate.TOOL_DISPATCH, {"rerun_jobs": tool, "submit_gowe_job": submit}):
        out = await populate.populate_and_submit(state.query, _agent_cfg(mode), state)
    return out, chat


@pytest.mark.asyncio
async def test_populate_rerun_result_ends_the_turn_with_the_question():
    replies = [_reply(tool_calls=[("rerun_jobs", {"folder": f"{HOME}/runs"})]),
               _reply("Prepared 3 reruns. Answer **Yes** below to submit them.")]
    out, chat = await _populate(replies, {"prepared_actions": RERUN_ACTIONS, "skipped": []})
    assert out.status == "completed" and out.current_phase == "ready"
    assert [a["summary"] for a in out.blocked_actions] == [a["summary"] for a in RERUN_ACTIONS]
    # The set is complete: the next call is the summary, no more tools.
    assert chat.await_args_list[-1].kwargs["tool_choice"] == "none"


@pytest.mark.asyncio
async def test_populate_rerun_in_execute_single_job_mode_still_only_prepares():
    replies = [_reply(tool_calls=[("rerun_jobs", {"task_ids": ["1"]})]),
               _reply("The rerun has been submitted successfully!"),
               _reply("The rerun job was submitted and is now running.")]
    out, _ = await _populate(replies, {"prepared_actions": RERUN_ACTIONS[:1]}, mode="execute", multi_job=False)
    assert out.status == "completed" and out.current_phase == "ready"
    assert out.submission_ids == []
    # A success claim is replaced with the set wording, not "switch to Execute mode".
    assert "nothing was submitted" in out.operation_message
    assert "Yes" in out.operation_message


@pytest.mark.asyncio
async def test_rerun_error_is_not_a_failed_submission():
    replies = [_reply(tool_calls=[("rerun_jobs", {"folder": "x"})]),
               _reply("I could not find any jobs in that folder. Which folder did you mean?")]
    out, _ = await _populate(replies, {"error": "No job results under x", "errorType": "PATH_NOT_FOUND"})
    assert out.status == "needs_input"
    assert out.blocked_actions == []


@pytest.mark.asyncio
async def test_one_job_rerun_in_execute_mode_still_asks():
    """Phase 1 submits a lone prepared job in Execute mode; a rerun does not.

    The parameters are copied from an old job the user has not seen in this
    turn, so the user confirms even one.
    """
    replies = [_reply(tool_calls=[("rerun_jobs", {"task_ids": ["1"]})]),
               _reply("Prepared 1 rerun. Answer **Yes** below to submit it.")]
    tool = AsyncMock(return_value={"prepared_actions": RERUN_ACTIONS[:1]})
    submit = AsyncMock(side_effect=AssertionError("must not submit"))
    with patch.object(populate, "create_client", lambda c: MagicMock()), \
         patch.object(populate, "chat_completion", AsyncMock(side_effect=replies)), \
         patch.dict(populate.TOOL_DISPATCH, {"rerun_jobs": tool, "submit_gowe_job": submit}), \
         patch.dict(replay_mod.TOOL_DISPATCH, {"submit_gowe_job": submit}), \
         patch.object(service_agent, "classify_intent", AsyncMock(return_value=Intent(job_count="many"))):
        result = await service_agent.run_agent(
            "rerun job 1", config=_agent_cfg("execute"), context={"execution_mode": "execute"})
    submit.assert_not_awaited()
    assert result.submission_ids == []
    assert len(result.blocked_actions) == 1
    assert result.blocked_actions[0]["origin"] == "rerun_jobs"


@pytest.mark.asyncio
async def test_yes_on_a_rerun_set_replays_the_arguments_verbatim():
    submit = AsyncMock(side_effect=[{"workflow_id": "wf_asm", "submission_id": f"sub_{i}",
                                     "output_path": a["arguments"]["inputs"]["output_path"]}
                                    for i, a in enumerate(RERUN_ACTIONS)])
    with patch.dict(replay_mod.TOOL_DISPATCH, {"submit_gowe_job": submit}):
        result = await service_agent.run_agent(
            "Yes", config=_agent_cfg("execute"),
            context={"execution_mode": "execute", "replay_actions": RERUN_ACTIONS})
    assert [c.kwargs["inputs"] for c in submit.await_args_list] == [a["arguments"]["inputs"] for a in RERUN_ACTIONS]
    assert result.submission_ids == ["sub_0", "sub_1", "sub_2"]


# ---------------------------------------------------------------------------
# Prompts: classifier, router, populate
# ---------------------------------------------------------------------------

def test_prompts_route_and_count_reruns():
    from service_agent.classifier import CLASSIFIER_SYSTEM_PROMPT
    from orchestrator.router.prompts import ROUTING_SYSTEM_PROMPT

    assert "rerun" in CLASSIFIER_SYSTEM_PROMPT.lower()
    assert "rerun" in ROUTING_SYSTEM_PROMPT.lower()
    for mode, multi in (("plan", False), ("execute", False), ("plan", True)):
        prompt = build_populate_prompt(execution_mode=mode, multi_job=multi)
        assert "== RERUN ==" in prompt
        rerun_section = prompt.split("== RERUN ==", 1)[1].split("\n==", 1)[0]
        assert "rerun_jobs" in rerun_section and "get_job_details" in rerun_section


# ---------------------------------------------------------------------------
# Success-claim guard: a denial is not a claim
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,claims", [
    ("I prepared 9 reruns (nothing has been submitted yet) and skipped 4.", False),
    ("The jobs have not been submitted; answer Yes to submit them.", False),
    ("No jobs were submitted.", False),
    ("It hasn't been submitted yet.", False),
    ("Nothing failed, and the job has been submitted.", True),
    ("Both jobs have been submitted successfully!", True),
    ("Nothing was changed. The rerun has been submitted.", True),
])
def test_claims_success_ignores_negated_mentions(text, claims):
    assert populate._claims_success(text) is claims


@pytest.mark.asyncio
async def test_a_reply_saying_nothing_was_submitted_is_kept():
    reply = ("I prepared 3 reruns (nothing has been submitted yet). Skipped: none.\n\n"
             "Answer **Yes** below to submit them, **No** to submit nothing, or tell me what to change.")
    replies = [_reply(tool_calls=[("rerun_jobs", {"folder": f"{HOME}/runs"})]), _reply(reply)]
    out, chat = await _populate(replies, {"prepared_actions": RERUN_ACTIONS})
    assert chat.await_count == 2  # no nudge
    assert out.operation_message == reply
