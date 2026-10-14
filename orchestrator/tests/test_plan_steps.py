"""D4 Phase 3: chains run as plan steps, in Plan mode, with structured outputs.

- the router's "pipeline" decision is retired (a stray one becomes planning);
- a delegated step that prepared jobs or groups pauses the plan on its
  question (EXECUTION_BLOCKED + PLAN_STEP_BLOCKED, no PLAN_STEP_COMPLETED);
- Yes on that question (replay with plan_context) completes the step;
- step_outputs are built from tool results and reach the dependent steps;
- create_group takes explicit ids; a blocked group shows its member count;
- the planner may look things up at most four times before it must plan.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import shared.agent_dispatch  # noqa: E402, F401  -- sets up agent package paths
from shared import step_outputs as so  # noqa: E402
from shared.models import ToolCall, ToolExecution  # noqa: E402
from shared.tools import execute_tool, groups  # noqa: E402

from orchestrator.router.models import ROUTING_JSON_SCHEMA, RoutingDecision  # noqa: E402
from orchestrator.router.router import _parse_routing_response  # noqa: E402

pytestmark = pytest.mark.asyncio

PLAN_CTX = {"plan_id": "plan_1", "step_id": "make_group", "step_index": 1}


def _te(name, args, result, error=None):
    return ToolExecution(tool_call=ToolCall(id="c", name=name, arguments=args), result=result, error=error)


# ---------------------------------------------------------------------------
# Router: pipeline retired
# ---------------------------------------------------------------------------

REGISTRY = SimpleNamespace(agents={"planning": object(), "data": object(), "service": object()},
                           agent_keys=["planning", "data", "service"])


async def test_pipeline_is_not_a_decision_any_more():
    assert ROUTING_JSON_SCHEMA["properties"]["decision"]["enum"] == ["direct", "agent"]
    with pytest.raises(Exception):
        RoutingDecision(decision="pipeline")


async def test_a_stray_pipeline_answer_goes_to_planning_with_the_original_query():
    query = "find plfam for fig|83332.12.peg.3062, take 100 members, save as a group, run msa"
    raw = json.dumps({"decision": "pipeline", "reasoning": "r", "steps": [
        {"agent_key": "data", "task": "find plfam", "depends_on": []},
        {"agent_key": "service", "task": "run msa", "depends_on": [0]}]})
    d = _parse_routing_response(raw, query, REGISTRY)
    assert d.decision == "agent"
    assert d.plan.steps[0].agent_key == "planning"
    assert d.plan.steps[0].task == query
    assert d.routing_override == "pipeline_to_planning"


async def test_router_prompt_sends_chains_to_planning():
    from orchestrator.router.prompts import ROUTING_SYSTEM_PROMPT

    assert "pipeline" not in ROUTING_SYSTEM_PROMPT.lower()
    assert "goes to **planning**" in ROUTING_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# step_outputs
# ---------------------------------------------------------------------------

async def test_step_outputs_from_tool_results():
    execs = [
        _te("search_data", {"collection": "genome_feature", "query": "plfam_id:PLF_1"},
            {"numFound": 350, "results": [{"feature_id": "fig|1.1.peg.1"}, {"feature_id": "fig|1.1.peg.2"}]}),
        _te("create_group", {"group_name": "g", "group_type": "feature_group",
                             "collection": "genome_feature", "query": "plfam_id:PLF_1"},
            {"name": "g", "path": "/u/home/Feature Groups/g", "count": 100, "group_type": "feature_group"}),
        _te("create_group", {"group_name": "x", "group_type": "genome_group", "query": "q"},
            {"error": "Plan mode", "blocked_by_mode": True}),
        _te("submit_gowe_job", {"workflow_id": "MSA"},
            {"submission_id": "sub_1", "output_path": "/u/home/chats/s/msa"}),
        _te("workspace_browse", {"path": "/u/home/reads"}, {"result": {"items": []}}),
        _te("search_data", {"collection": "genome"}, {"error": "bad query"}),
    ]
    out = so.build_step_outputs(execs)
    assert out["ids"] == [{"collection": "genome_feature", "field": "feature_id",
                           "values": ["fig|1.1.peg.1", "fig|1.1.peg.2"], "total": 350,
                           "query": "plfam_id:PLF_1"}]
    assert out["groups"] == [{"name": "g", "path": "/u/home/Feature Groups/g", "type": "feature_group",
                              "count": 100, "selection": {"query": "plfam_id:PLF_1", "collection": "genome_feature"}}]
    assert out["submission_ids"] == ["sub_1"]
    assert out["output_paths"] == ["/u/home/chats/s/msa"]
    assert out["paths"] == ["/u/home/reads"]


async def test_values_extract_known_shapes_only():
    answer = ("The PLFam is PLF_1763_00002110 (PGFam PGF_00012345) for fig|83332.12.peg.3062 "
              "(feature PATRIC.83332.12.NC_000962.CDS.3335.4198.fwd).")
    assert so.extract_values(answer, ["plfam_id", "pgfam_id", "patric_id", "feature_id", "colour"]) == {
        "plfam_id": "PLF_1763_00002110", "pgfam_id": "PGF_00012345",
        "patric_id": "fig|83332.12.peg.3062", "feature_id": "PATRIC.83332.12.NC_000962.CDS.3335.4198.fwd"}
    assert so.extract_values("no ids here", ["plfam_id"]) == {}


async def test_format_step_outputs_lists_every_value():
    text = so.format_step_outputs({
        "find_plfam: Find the PLFam": {"values": {"plfam_id": "PLF_1"}},
        "make_group: Make the group": {"groups": [{"name": "g", "path": "/u/home/Feature Groups/g",
                                                   "type": "feature_group", "count": 100,
                                                   "selection": {"query": "plfam_id:PLF_1"}}]},
        "empty": {},
    })
    assert text.startswith("=== PRIOR STEP OUTPUTS ===")
    assert "plfam_id: PLF_1" in text and "path /u/home/Feature Groups/g, 100 members" in text
    assert "empty" not in text
    assert so.format_step_outputs({"a": {}}) == ""


async def test_dispatch_attaches_step_outputs():
    from shared.agent_dispatch import _with_blocked_actions

    result = SimpleNamespace(blocked_actions=[], tool_trace=[
        _te("create_group", {"group_name": "g", "group_type": "genome_group", "query": "q"},
            {"name": "g", "path": "/u/home/Genome Groups/g", "count": 7})])
    resp = _with_blocked_actions({"answer": "ok"}, result)
    assert resp["step_outputs"]["groups"][0]["path"] == "/u/home/Genome Groups/g"


# ---------------------------------------------------------------------------
# create_group: explicit ids, and the member count on a blocked group
# ---------------------------------------------------------------------------

def _gf(record):
    async def create(**kw):
        record.update(kw)
        return {"path": f"/u/home/Genome Groups/{kw['name']}", "count": len(kw["id_list"]), "action": "created"}
    return SimpleNamespace(create_group=create)


async def test_create_group_from_ids_keeps_order_and_drops_duplicates():
    record = {}
    with patch.object(groups, "get_group_functions", lambda *_: _gf(record)), \
         patch.object(groups, "get_json_rpc", lambda *_: SimpleNamespace(JsonRpcCaller=lambda **k: object())), \
         patch.object(groups, "_fetch_ids_by_query", AsyncMock(side_effect=AssertionError("no query"))):
        out = await groups.create_group("g", "genome_group", ids=["83332.12", "1313.5", "83332.12"],
                                        headers={"Authorization": "tok"})
    assert record["id_list"] == ["83332.12", "1313.5"]
    assert out["count"] == 2 and out["query_used"] is None


@pytest.mark.parametrize("kwargs", [
    {"ids": ["83332.12"], "query": "genus:X", "collection": "genome"},
    {},
])
async def test_create_group_needs_exactly_one_selection(kwargs):
    out = await groups.create_group("g", "genome_group", headers={"Authorization": "tok"}, **kwargs)
    assert out["errorType"] == "INVALID_PARAMETERS"


async def test_create_group_rejects_malformed_ids():
    out = await groups.create_group("g", "feature_group", ids=["fig|1.1.peg.1", "PLF_1"],
                                    headers={"Authorization": "tok"})
    assert out["errorType"] == "INVALID_PARAMETERS" and "PLF_1" in out["error"]


async def test_a_feature_group_from_patric_ids_holds_their_feature_ids():
    """fig|... is a patric_id; a feature group stores feature_id values."""
    record = {}
    solr = AsyncMock(return_value={"results": [
        {"patric_id": "fig|83332.12.peg.2", "feature_id": "PATRIC.83332.12.NC_000962.CDS.2.9.fwd"},
        {"patric_id": "fig|83332.12.peg.1", "feature_id": "PATRIC.83332.12.NC_000962.CDS.1.5.fwd"}]})
    with patch.object(groups, "get_group_functions", lambda *_: _gf(record)), \
         patch.object(groups, "get_json_rpc", lambda *_: SimpleNamespace(JsonRpcCaller=lambda **k: object())), \
         patch.object(groups, "_get_solr_query_fn", lambda: solr):
        out = await groups.create_group("g", "feature_group", headers={"Authorization": "tok"},
                                        ids=["fig|83332.12.peg.1", "PATRIC.83332.12.NC_000962.CDS.7.8.rev",
                                             "fig|83332.12.peg.2"])
    assert record["id_list"] == ["PATRIC.83332.12.NC_000962.CDS.1.5.fwd", "PATRIC.83332.12.NC_000962.CDS.7.8.rev",
                                 "PATRIC.83332.12.NC_000962.CDS.2.9.fwd"]
    assert out["count"] == 3
    with patch.object(groups, "_get_solr_query_fn", lambda: AsyncMock(return_value={"results": []})):
        out = await groups.create_group("g", "feature_group", headers={"Authorization": "tok"},
                                        ids=["fig|83332.12.peg.9"])
    assert out["errorType"] == "NOT_FOUND"


async def test_blocked_group_question_shows_the_selection_and_count():
    solr = AsyncMock(return_value={"numFound": 431})
    tool = AsyncMock()
    args = {"group_name": "strep_tet_R", "group_type": "genome_group", "collection": "genome_amr",
            "query": "antibiotic:tetracycline AND resistant_phenotype:Resistant", "limit": 200}
    with patch.object(groups, "_get_solr_query_fn", lambda: solr):
        out = await execute_tool("create_group", dict(args), {"create_group": tool},
                                 config=SimpleNamespace(execution_mode="plan", agent_key="data"))
    tool.assert_not_awaited()
    assert out["blocked_by_mode"] and out["member_count"] == 200
    assert out["summary"] == ("genome_group 'strep_tet_R' — up to 200 genomes "
                              "(genome_amr: antibiotic:tetracycline AND resistant_phenotype:Resistant)")
    # The replayed arguments are exactly what was shown.
    assert out["arguments"] == args
    # One genome per row on 'genome': the count is exact.
    with patch.object(groups, "_get_solr_query_fn", lambda: AsyncMock(return_value={"numFound": 12})):
        out = await execute_tool("create_group", {**args, "collection": "genome", "query": "genus:X"},
                                 {"create_group": tool}, config=SimpleNamespace(execution_mode="plan"))
    assert "— 12 genomes" in out["summary"]


# ---------------------------------------------------------------------------
# Planning agent
# ---------------------------------------------------------------------------

STEPS = [
    {"step_id": "find_plfam", "description": "Find the PLFam", "agent": "data", "depends_on": [],
     "expects": ["plfam_id"]},
    {"step_id": "make_group", "description": "Make a group of 100 members", "agent": "data",
     "depends_on": ["find_plfam"]},
    {"step_id": "side", "description": "Unrelated lookup", "agent": "helpdesk", "depends_on": []},
    {"step_id": "run_msa", "description": "Run MSA on the group", "agent": "service",
     "depends_on": ["make_group"]},
]
COMPLETED = {
    "find_plfam": {"answer": "PLF_1 found", "step_outputs": {"values": {"plfam_id": "PLF_1"}}},
    "make_group": {"answer": "group made " + "x" * 3000,
                   "step_outputs": {"groups": [{"name": "g", "path": "/u/home/Feature Groups/g",
                                                "type": "feature_group", "count": 100}]}},
    "side": {"answer": "unrelated"},
}


async def test_only_transitive_dependencies_feed_a_step():
    from planning_agent.agent import _relevant_prior_results

    assert list(_relevant_prior_results(STEPS, 3, COMPLETED)) == ["find_plfam", "make_group"]
    # A step without depends_on gets every earlier completed step.
    assert list(_relevant_prior_results(STEPS, 2, COMPLETED)) == ["find_plfam", "make_group"]


async def test_execute_step_puts_outputs_first_and_in_the_delegated_task():
    from planning_agent import agent as planning
    from planning_agent.models import AgentConfig

    chat = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="Run MSA SNP on the feature group.", tool_calls=None, reasoning=None),
        finish_reason="stop")]))
    ctx = {"workflow_context": {"plan_action": "execute_next", "current_step_index": 3,
                                "plan": {"plan_id": "p", "title": "t", "description": "d", "steps": STEPS},
                                "completed_step_results": COMPLETED}}
    cfg = AgentConfig(llm_base_url="http://llm", llm_api_key="k", llm_model="m")
    with patch.object(planning, "create_client", lambda c: MagicMock()), \
         patch.object(planning, "chat_completion", chat):
        result = await planning.run_agent("continue", config=cfg, context=ctx)
    system = chat.await_args.kwargs["messages"][0]["content"]
    assert system.index("=== PRIOR STEP OUTPUTS ===") < system.index("## Results from Prior Steps")
    assert "/u/home/Feature Groups/g" in system and "plfam_id: PLF_1" in system
    assert "unrelated" not in system                # not a dependency
    assert "x" * 1600 not in system                 # answers cut to 1500
    task = result.step_execution["task"]
    assert task.startswith("Run MSA SNP on the feature group.")
    assert "/u/home/Feature Groups/g" in task       # verbatim, whatever the paraphrase


async def test_recon_cap_leaves_only_plan_or_ask_after_four_lookups():
    from planning_agent import agent as planning
    from planning_agent.models import AgentConfig, AgentState

    def _call(i, name="workspace_browse"):
        tc = SimpleNamespace(id=f"c{i}", type="function",
                             function=SimpleNamespace(name=name, arguments=json.dumps({"path": f"/p{i}"})))
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="", tool_calls=[tc], reasoning=None), finish_reason="tool_calls")])

    final = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="I need more detail.", tool_calls=None, reasoning=None),
        finish_reason="stop")])
    chat = AsyncMock(side_effect=[_call(i) for i in range(4)] + [final])
    browse = AsyncMock(return_value={"result": {"items": []}})
    cfg = AgentConfig(llm_base_url="http://llm", llm_api_key="k", llm_model="m")
    state = AgentState(query="q", context={})
    state.add_system_message("sys")
    state.add_user_message("q")
    with patch.object(planning, "chat_completion", chat), \
         patch.dict("shared.tools.registry.TOOL_DISPATCH", {"workspace_browse": browse}):
        await planning._run_planning_loop(state, MagicMock(), cfg, None)
    tools_per_call = [[t["function"]["name"] for t in c.kwargs["tools"]] for c in chat.await_args_list]
    assert all(len(t) > 2 for t in tools_per_call[:4])
    assert sorted(tools_per_call[4]) == ["ask_clarification", "create_plan"]
    assert any("enough to plan" in str(m.get("content")) for m in chat.await_args_list[4].kwargs["messages"])


async def test_recon_cap_holds_within_one_response():
    from planning_agent import agent as planning
    from planning_agent.models import AgentConfig, AgentState

    calls = [SimpleNamespace(id=f"c{i}", type="function",
                             function=SimpleNamespace(name="search_data",
                                                      arguments=json.dumps({"collection": "genome", "query": f"q{i}"})))
             for i in range(6)]
    many = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="", tool_calls=calls, reasoning=None), finish_reason="tool_calls")])
    final = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="done", tool_calls=None, reasoning=None), finish_reason="stop")])
    search = AsyncMock(return_value={"numFound": 1, "results": []})
    state = AgentState(query="q", context={})
    state.add_system_message("sys")
    state.add_user_message("q")
    with patch.object(planning, "chat_completion", AsyncMock(side_effect=[many, final])), \
         patch.dict("shared.tools.registry.TOOL_DISPATCH", {"search_data": search}):
        await planning._run_planning_loop(state, MagicMock(),
                                          AgentConfig(llm_base_url="http://llm", llm_api_key="k", llm_model="m"), None)
    assert search.await_count == 4
    skipped = [m for m in state.messages if m.get("role") == "tool" and "_skipped" in m["content"]]
    assert len(skipped) == 2


async def test_create_plan_keeps_expects():
    from planning_agent.tools.plan_tools import handle_create_plan

    out = handle_create_plan({"title": "t", "description": "d", "steps": [
        {"step_id": "a", "description": "find", "agent": "data", "reasoning": "r", "expects": ["plfam_id"]},
        {"step_id": "b", "description": "use", "agent": "service", "reasoning": "r", "depends_on": ["a"]}]})
    assert out["plan"]["steps"][0]["expects"] == ["plfam_id"]
    assert out["plan"]["steps"][1]["expects"] == []


# ---------------------------------------------------------------------------
# Orchestrator: the plan pauses on a step's question; Yes completes the step
# ---------------------------------------------------------------------------

def _registry(*keys):
    from mcp.types import Tool as McpTool

    from orchestrator.config import AgentConfig, OrchestratorConfig
    from orchestrator.registry.agent_handle import AgentHandle
    from orchestrator.registry.agent_registry import AgentRegistry

    config = OrchestratorConfig(
        agents={k: AgentConfig(name=k, description=k, endpoint="http://localhost:8053", capabilities=[])
                for k in keys},
        health_check_interval=0, routing_model="m", routing_base_url="http://x", routing_api_key="k",
    )
    registry = AgentRegistry(config)
    for k in keys:
        h = AgentHandle(k, config.agents[k])
        h._healthy = True
        h._tools = [McpTool(name="agent_chat", description="Chat", inputSchema={})]
        registry._agents[k] = h
    return registry


def _agent_reply(**kw):
    return {"answer": "", "status": "completed", "sources": [], "elapsed_seconds": 0.1, "tool_trace": [], **kw}


async def _run(registry, **request_kw):
    from orchestrator.events.stream import collect_events
    from orchestrator.llm.client import LLMClient
    from orchestrator.models import OrchestratorRequest
    from orchestrator.orchestrate import orchestrate

    llm = MagicMock(spec=LLMClient)
    llm.complete = AsyncMock(return_value="unused")
    request = OrchestratorRequest(**request_kw)
    return await collect_events(orchestrate(request, registry, llm))


def _types(events):
    return [e.type.value for e in events]


GROUP_ACTION = {"tool": "create_group", "summary": "feature_group 'g' — 100 features (genome_feature: plfam_id:PLF_1)",
                "arguments": {"group_name": "g", "group_type": "feature_group",
                              "collection": "genome_feature", "query": "plfam_id:PLF_1", "limit": 100}}


async def test_a_step_that_prepared_actions_pauses_the_plan_on_its_question():
    registry = _registry("planning", "data")
    registry._agents["planning"].call_tool = AsyncMock(return_value=_agent_reply(
        status="step_ready", step_execution={**PLAN_CTX, "agent": "data", "task": "Make the group",
                                             "expects": ["plfam_id"]}))
    registry._agents["data"].call_tool = AsyncMock(return_value=_agent_reply(
        answer="Ready to create the group of PLF_1763_00002110 members.", blocked_actions=[GROUP_ACTION],
        step_outputs={"ids": [{"collection": "genome_feature", "field": "feature_id", "values": ["fig|1.1.peg.1"], "total": 1}]}))
    events = await _run(registry, query="continue", target_agent="planning",
                        workflow_context={"plan_action": "execute_next"})
    types = _types(events)
    assert "plan_step_completed" not in types
    blocked = next(e for e in events if e.type.value == "execution_blocked")
    assert blocked.data["plan_context"] == PLAN_CTX and blocked.data["blocked_actions"] == [GROUP_ACTION]
    paused = next(e for e in events if e.type.value == "plan_step_blocked")
    assert {k: paused.data[k] for k in PLAN_CTX} == PLAN_CTX
    assert paused.data["summaries"] == [GROUP_ACTION["summary"]]
    assert paused.data["step_outputs"]["values"] == {"plfam_id": "PLF_1763_00002110"}
    done = events[-1]
    assert done.type.value == "orchestrator_done" and done.data["response_text"]
    # A data step is not a batch: the flag is for service steps only.
    ctx = registry._agents["data"].call_tool.await_args.args[1]["context"]
    assert "batch_mode" not in ctx


async def test_an_empty_step_answer_still_gives_text_for_the_card_message():
    registry = _registry("planning", "data")
    registry._agents["planning"].call_tool = AsyncMock(return_value=_agent_reply(
        status="step_ready", step_execution={**PLAN_CTX, "agent": "data", "task": "Make the group"}))
    registry._agents["data"].call_tool = AsyncMock(return_value=_agent_reply(blocked_actions=[GROUP_ACTION]))
    events = await _run(registry, query="continue", target_agent="planning",
                        workflow_context={"plan_action": "execute_next"})
    assert "Answer the question below" in events[-1].data["response_text"]


async def test_a_service_step_is_a_batch_and_a_completed_step_carries_its_outputs():
    registry = _registry("planning", "service", "data")
    registry._agents["planning"].call_tool = AsyncMock(return_value=_agent_reply(
        status="step_ready", step_execution={**PLAN_CTX, "agent": "service", "task": "Run MSA"}))
    registry._agents["service"].call_tool = AsyncMock(return_value=_agent_reply(
        answer="Submitted.", submission_ids=["sub_1"], auto_submitted=True,
        step_outputs={"submission_ids": ["sub_1"], "output_paths": ["/u/home/chats/s/msa"]}))
    events = await _run(registry, query="continue", target_agent="planning",
                        workflow_context={"plan_action": "execute_next"})
    assert registry._agents["service"].call_tool.await_args.args[1]["context"]["batch_mode"] is True
    completed = next(e for e in events if e.type.value == "plan_step_completed")
    assert completed.data["step_outputs"]["output_paths"] == ["/u/home/chats/s/msa"]
    submitted = next(e for e in events if e.type.value == "plan_workflow_submitted")
    assert submitted.data["step_outputs"]["submission_ids"] == ["sub_1"]


async def test_yes_on_a_step_question_completes_the_step():
    registry = _registry("service")
    registry._agents["service"].call_tool = AsyncMock(return_value=_agent_reply(
        answer="Created feature group 'g' (100 items).",
        step_outputs={"groups": [{"name": "g", "path": "/u/home/Feature Groups/g", "count": 100}]}))
    events = await _run(registry, query="Yes", target_agent="service", execute_once=False,
                        replay_actions=[GROUP_ACTION], plan_context=PLAN_CTX, execution_mode="execute")
    types = _types(events)
    completed = next(e for e in events if e.type.value == "plan_step_completed")
    assert {k: completed.data[k] for k in PLAN_CTX} == PLAN_CTX
    assert completed.data["step_outputs"]["groups"][0]["path"] == "/u/home/Feature Groups/g"
    assert "plan_workflow_submitted" not in types       # groups only: the plan auto-advances
    assert types[-1] == "orchestrator_done"


async def test_yes_on_a_job_step_puts_the_plan_in_its_waiting_state():
    registry = _registry("service")
    registry._agents["service"].call_tool = AsyncMock(return_value=_agent_reply(
        answer="Submitted all 2 jobs.", submission_ids=["sub_1", "sub_2"],
        step_outputs={"submission_ids": ["sub_1", "sub_2"]}))
    events = await _run(registry, query="Yes", target_agent="service",
                        replay_actions=[GROUP_ACTION], plan_context=PLAN_CTX, execution_mode="execute")
    submitted = next(e for e in events if e.type.value == "plan_workflow_submitted")
    assert submitted.data["submission_ids"] == ["sub_1", "sub_2"]
    assert submitted.data["step_id"] == "make_group"


async def test_a_failed_replay_fails_the_step():
    registry = _registry("service")
    registry._agents["service"].call_tool = AsyncMock(return_value=_agent_reply(
        status="error", answer="Could not run **g**: Workspace error"))
    events = await _run(registry, query="Yes", target_agent="service",
                        replay_actions=[GROUP_ACTION], plan_context=PLAN_CTX, execution_mode="execute")
    assert "plan_step_failed" in _types(events)
    assert "plan_step_completed" not in _types(events)


async def test_a_revision_keeps_the_step_on_its_new_question():
    registry = _registry("service")
    revised = dict(GROUP_ACTION, summary="feature_group 'g' — 50 features")
    registry._agents["service"].call_tool = AsyncMock(return_value=_agent_reply(
        answer="Prepared the smaller group.", blocked_actions=[revised]))
    events = await _run(registry, query="only 50", target_agent="service",
                        prepared_actions=[GROUP_ACTION], plan_context=PLAN_CTX)
    blocked = next(e for e in events if e.type.value == "execution_blocked")
    assert blocked.data["plan_context"] == PLAN_CTX
    assert "plan_step_blocked" in _types(events)
    assert "plan_step_completed" not in _types(events)


async def test_a_step_whose_agent_failed_or_timed_out_is_a_failed_step():
    """Live, Oct 14: the timeout's error event was dropped, the step came back
    empty as "completed", and synthesis wrote "I don't have the current plan"."""
    registry = _registry("planning", "data")
    registry._agents["planning"].call_tool = AsyncMock(return_value=_agent_reply(
        status="step_ready", step_execution={**PLAN_CTX, "agent": "data", "task": "Find the PLFam"}))
    registry._agents["data"].call_tool = AsyncMock(return_value=_agent_reply(
        status="error", answer="I ran out of time (the 5-minute limit) before I could finish."))
    events = await _run(registry, query="continue", target_agent="planning",
                        workflow_context={"plan_action": "execute_next"})
    types = _types(events)
    assert "plan_step_completed" not in types
    failed = next(e for e in events if e.type.value == "plan_step_failed")
    assert "ran out of time" in failed.data["error"]
    assert "ran out of time" in events[-1].data["response_text"]


async def test_a_direct_turn_without_plan_context_is_unchanged():
    registry = _registry("service")
    registry._agents["service"].call_tool = AsyncMock(return_value=_agent_reply(
        answer="Ready.", blocked_actions=[GROUP_ACTION]))
    events = await _run(registry, query="make a group", target_agent="service")
    blocked = next(e for e in events if e.type.value == "execution_blocked")
    assert "plan_context" not in blocked.data
    assert not any(t.startswith("plan_") for t in _types(events))


# ---------------------------------------------------------------------------
# Service agent: a batch step is a job set (§2.2.1)
# ---------------------------------------------------------------------------

async def test_batch_step_prepares_jobs_even_in_execute_mode():
    from service_agent import agent as service_agent
    from service_agent.models import AgentConfig, Intent

    seen = {}

    async def fake_flow(query, config, state, progress_callback=None):
        seen["mode"], seen["multi_job"] = config.execution_mode, state.multi_job
        state.status = "completed"
        return state.to_result()

    cfg = AgentConfig(llm_base_url="http://llm", llm_api_key="k", llm_model="m", execution_mode="execute")
    with patch.object(service_agent, "classify_intent", AsyncMock(return_value=Intent(job_count="one"))), \
         patch.object(service_agent, "_run_gowe_flow", fake_flow):
        await service_agent.run_agent("assemble SRR1 and SRR2", config=cfg,
                                      context={"execution_mode": "execute", "batch_mode": True})
    assert seen == {"mode": "plan", "multi_job": True}
