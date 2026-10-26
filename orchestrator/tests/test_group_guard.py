"""create_group guard: one group per plan step; no two groups with the same members.

Live, 2026-10-15 (marcnguy's tet R vs S tree, re-run): the plan's data step
counted resistant and susceptible genomes correctly, then searched genomes
with no phenotype filter and built BOTH groups from those same 50 ids.  The
user's Yes created two identical groups.  The guard runs in execute_tool
before the plan-mode gate, so the second group is refused when prepared.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import shared.agent_dispatch  # noqa: E402, F401  -- sets up agent package paths
from shared.tools import execute_tool  # noqa: E402
from bvbrc_mcp.tools import groups  # noqa: E402

FIFTY = [f"1313.{22000 + i}" for i in range(50)]
OTHER = [f"1313.{23000 + i}" for i in range(50)]
R_QUERY = "antibiotic:tetracycline AND resistant_phenotype:Resistant AND genome_name:*pneumoniae*"
S_QUERY = "antibiotic:tetracycline AND resistant_phenotype:Susceptible AND genome_name:*pneumoniae*"


def _cfg(plan_step=None, mode="plan"):
    return SimpleNamespace(agent_key="data", handoff_from=None, execution_mode=mode,
                           bvbrc_auth_token="tok", plan_step=plan_step)


def _group(name, **sel):
    return {"group_name": name, "group_type": "genome_group", "limit": 50, **sel}


def _fetch(by_query):
    async def fake(collection, query, id_field, limit, token=None, base_url=None):
        return {"ids": by_query[query], "total_matching": len(by_query[query])}
    return fake


async def _run(calls, cfg, by_query=None, prior=None):
    """execute_tool each create_group in one agent run; return the results."""
    token = groups.GROUP_RUN.set({})
    tool = AsyncMock(return_value={"path": "/u/home/Genome Groups/x"})
    try:
        with patch.object(groups, "_fetch_ids_by_query", _fetch(by_query or {})), \
             patch.object(groups, "_workspace_group_members", AsyncMock(return_value=prior or {})), \
             patch.object(groups, "count_group_members", AsyncMock(return_value=50)):
            return [await execute_tool("create_group", dict(a), {"create_group": tool}, config=cfg)
                    for a in calls], tool
    finally:
        groups.GROUP_RUN.reset(token)


@pytest.mark.asyncio
async def test_the_live_case_same_ids_by_query_and_by_list_is_refused():
    calls = [_group("strep_pneumo_tetR", collection="genome", query="genome_id:(" + " OR ".join(FIFTY) + ")"),
             _group("strep_pneumo_tetS", collection="genome", ids=FIFTY)]
    (first, second), _ = await _run(calls, _cfg(), {calls[0]["query"]: FIFTY})
    assert first.get("blocked_by_mode")                       # prepared, as before
    assert second["errorType"] == "DUPLICATE_GROUP_MEMBERS" and second["same_as"] == "strep_pneumo_tetR"
    assert "Nothing was created" in second["error"] and not second.get("blocked_by_mode")


@pytest.mark.asyncio
async def test_two_groups_from_their_own_conditions_both_pass():
    calls = [_group("tetR", collection="genome_amr", query=R_QUERY),
             _group("tetS", collection="genome_amr", query=S_QUERY)]
    results, _ = await _run(calls, _cfg(), {R_QUERY: FIFTY, S_QUERY: OTHER})
    assert all(r.get("blocked_by_mode") for r in results)


@pytest.mark.asyncio
async def test_execute_mode_identical_group_is_never_created():
    calls = [_group("a", ids=FIFTY), _group("b", ids=list(reversed(FIFTY)))]
    (first, second), tool = await _run(calls, _cfg(mode="execute"))
    assert tool.await_count == 1 and second["errorType"] == "DUPLICATE_GROUP_MEMBERS"


@pytest.mark.asyncio
async def test_a_plan_step_makes_one_group():
    step = {"plan_id": "p1", "step_id": "make_r", "prior_groups": []}
    calls = [_group("tetR", collection="genome_amr", query=R_QUERY),
             _group("tetS", collection="genome_amr", query=S_QUERY)]
    (first, second), _ = await _run(calls, _cfg(plan_step=step), {R_QUERY: FIFTY, S_QUERY: OTHER})
    assert first.get("blocked_by_mode")
    assert second["errorType"] == "ONE_GROUP_PER_STEP" and "own step" in second["error"]


@pytest.mark.asyncio
async def test_a_plan_step_cannot_rebuild_an_earlier_steps_group():
    prior_path = "/u/home/Genome Groups/tetR"
    step = {"plan_id": "p1", "step_id": "make_s", "prior_groups": [{"name": "tetR", "path": prior_path}]}
    calls = [_group("tetS", collection="genome", ids=FIFTY)]
    (result,), _ = await _run(calls, _cfg(plan_step=step), prior={prior_path: frozenset(FIFTY)})
    assert result["errorType"] == "DUPLICATE_GROUP_MEMBERS" and result["same_as"] == "tetR"


@pytest.mark.asyncio
async def test_replacing_or_appending_to_the_same_group_is_not_a_duplicate():
    calls = [_group("tetR", ids=FIFTY), _group("tetR", ids=FIFTY, if_exists="replace")]
    results, _ = await _run(calls, _cfg())
    assert all(r.get("blocked_by_mode") for r in results)


@pytest.mark.asyncio
async def test_without_a_run_slot_only_prior_groups_are_checked():
    tool = AsyncMock()
    with patch.object(groups, "count_group_members", AsyncMock(return_value=50)):
        out = await execute_tool("create_group", _group("a", ids=FIFTY), {"create_group": tool}, config=_cfg())
    assert out.get("blocked_by_mode")


def test_planner_hands_each_step_the_earlier_groups():
    from planning_agent.agent import _prior_groups
    done = {
        "make_r": {"step_outputs": {"groups": [{"name": "tetR", "path": "/u/G/tetR", "count": 50}]}},
        "make_s": {"step_outputs": {"groups": [{"name": "tetS", "path": "/u/G/tetS"},
                                               {"name": "tetR", "path": "/u/G/tetR"}]}},
        "count": {"step_outputs": {}},
    }
    assert _prior_groups(done) == [{"name": "tetR", "path": "/u/G/tetR"}, {"name": "tetS", "path": "/u/G/tetS"}]


def test_plan_step_reaches_the_agent_config():
    from shared.agent_dispatch import _build_config_kwargs
    step = {"plan_id": "p1", "step_id": "make_r", "prior_groups": []}
    assert _build_config_kwargs("tok", {"plan_step": step})["plan_step"] == step
    assert "plan_step" not in _build_config_kwargs("tok", {})


def test_prompts_say_one_group_per_step_and_build_from_the_condition():
    from planning_agent.prompts.system import build_system_prompt
    from shared.prompts.groups_sra_skill import GROUPS_SRA_SKILL_PROMPT
    planner = " ".join(build_system_prompt("catalog").split())
    assert "ONE GROUP PER STEP" in planner and "One step for both groups" not in planner
    assert "DUPLICATE_GROUP_MEMBERS" in GROUPS_SRA_SKILL_PROMPT and "ONE_GROUP_PER_STEP" in GROUPS_SRA_SKILL_PROMPT


def test_prompts_never_wildcard_a_species_epithet():
    """Live, 2026-10-15: the planner's own example query, genome_name:*pneumoniae*,
    was copied into a "S. pneumoniae" group, which then held Klebsiella."""
    import re
    from data_agent.prompts import system as data_system
    from planning_agent.prompts.system import build_system_prompt
    from shared.prompts.data_skill import DATA_SKILL_PROMPT
    data_prompts = " ".join(v for v in vars(data_system).values() if isinstance(v, str))
    for text in (build_system_prompt("catalog"), data_prompts, DATA_SKILL_PROMPT):
        uses = [m for m in re.findall(r"genome_name:\*[a-z]+\*", text)]
        # The only allowed mention is the warning itself.
        assert all("also matches" in text for _ in uses)
    # Recipes use placeholders, so a matching request has nothing to copy verbatim.
    assert 'genome_name:"<Genus species>"' in build_system_prompt("catalog")
    assert "Streptococcus pneumoniae" not in build_system_prompt("catalog")
