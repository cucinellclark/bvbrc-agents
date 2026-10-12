"""Only the service agent submits jobs; only data and service create groups.

Every agent has every tool schema. The data agent submitted jobs in answer to
plain questions (parrello a05b7324 Sep 18, e88f5e12 Oct 9), the helpdesk agent
resubmitted a job when a user complained it had not run (Chris dae698e5), and
a job from a non-service agent carried no submission_ids, so it got no
workflow watch and no completion message.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared import agent_dispatch  # noqa: E402
from shared.tools import (  # noqa: E402
    execute_tool,
    is_blocked_result,
    is_wrong_agent_result,
)


def _cfg(agent_key, mode="execute"):
    return SimpleNamespace(agent_key=agent_key, execution_mode=mode)


async def _call(tool, agent_key, mode="execute"):
    fn = AsyncMock(return_value={"ok": True})
    result = await execute_tool(tool, {"workflow_id": "wf", "inputs": {}}, {tool: fn},
                                config=_cfg(agent_key, mode), inject_config=False)
    return result, fn


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["data", "helpdesk", "workspace", "analysis", "planning"])
async def test_submit_refused_for_non_service_agents(agent):
    result, fn = await _call("submit_gowe_job", agent)
    fn.assert_not_awaited()
    assert is_wrong_agent_result(result)
    assert f"not available to the {agent} agent" in result["error"]
    assert "Nothing was run" in result["error"]


@pytest.mark.asyncio
async def test_service_agent_submits():
    result, fn = await _call("submit_gowe_job", "service")
    fn.assert_awaited_once()
    assert result == {"ok": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["data", "service"])
async def test_data_and_service_create_groups(agent):
    result, fn = await _call("create_group", agent)
    fn.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["planning", "helpdesk", "workspace", "analysis"])
async def test_other_agents_cannot_create_groups(agent):
    result, fn = await _call("create_group", agent)
    fn.assert_not_awaited()
    assert is_wrong_agent_result(result)


@pytest.mark.asyncio
async def test_agent_gate_runs_before_plan_mode_gate():
    """A data agent in plan mode must not produce a replayable blocked card."""
    result, fn = await _call("submit_gowe_job", "data", mode="plan")
    fn.assert_not_awaited()
    assert is_wrong_agent_result(result)
    assert not is_blocked_result(result)


@pytest.mark.asyncio
async def test_service_in_plan_mode_is_still_blocked_by_mode():
    result, _ = await _call("submit_gowe_job", "service", mode="plan")
    assert is_blocked_result(result)


@pytest.mark.asyncio
async def test_no_agent_key_is_unrestricted():
    result, fn = await _call("submit_gowe_job", None)
    fn.assert_awaited_once()


@pytest.mark.asyncio
async def test_read_only_tools_are_unaffected():
    fn = AsyncMock(return_value={"results": []})
    await execute_tool("search_data", {"collection": "genome", "query": "*:*"}, {"search_data": fn},
                       config=_cfg("helpdesk"), inject_config=False)
    fn.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["data", "service", "planning"])
async def test_dispatch_sets_agent_key(agent):
    seen = {}

    async def runner(query, config_kwargs, ctx, progress_callback):
        seen.update(config_kwargs)
        return {"answer": "ok", "status": "completed"}

    with patch.dict(agent_dispatch._AGENT_RUNNERS, {agent: runner}):
        await agent_dispatch.dispatch_agent(agent, "q", token="tok", context={})
    assert seen["agent_key"] == agent


def test_every_agent_config_accepts_agent_key():
    from data_agent.models import AgentConfig as DataConfig
    from service_agent.models import AgentConfig as ServiceConfig
    for cls in (DataConfig, ServiceConfig):
        cfg = cls(llm_base_url="http://x", llm_api_key="k", llm_model="m", agent_key="data")
        assert cfg.agent_key == "data"
