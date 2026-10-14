"""A model that repeats a tool call is stopped, not left to run out the clock.

Benchmark run smoke-20261014b: a data step repeated one search ~90 times.
The duplicate check refused each repeat, but its ``break`` only left the
current response's tool calls, so the next LLM call repeated it again until
the 300 s agent limit (2.4 M prompt tokens).  Now the loop gives one nudge,
one final call with ``tool_choice="none"``, and stops.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import shared.agent_dispatch  # noqa: E402, F401  -- sets up agent package paths
from shared.agent_loop import run_agent_loop  # noqa: E402
from shared.models import BaseAgentConfig, BaseAgentState  # noqa: E402

pytestmark = pytest.mark.asyncio

SEARCH = ("search_data", {"collection": "genome_feature", "query": "plfam_id:PLF_1763_00002110"})


def _reply(content="", calls=()):
    tcs = [SimpleNamespace(id=f"c{i}", type="function",
                           function=SimpleNamespace(name=n, arguments=json.dumps(a)))
           for i, (n, a) in enumerate(calls)]
    msg = SimpleNamespace(content=content, tool_calls=tcs or None, reasoning=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls" if tcs else "stop")])


def _cfg():
    return BaseAgentConfig(llm_base_url="http://llm", llm_api_key="k", llm_model="m", max_iterations=200)


async def _run(replies, tool_result=None):
    state = BaseAgentState(query="make a group of 100 members", context={})
    tool = AsyncMock(return_value=tool_result or {"numFound": 48143, "results": []})
    chat = AsyncMock(side_effect=list(replies))
    with patch.dict("shared.tools.registry.TOOL_DISPATCH", {"search_data": tool}):
        await run_agent_loop(state=state, config=_cfg(), client=MagicMock(), system_prompt="sys",
                             chat_completion_fn=chat)
    return state, tool, chat


async def test_a_model_repeating_one_call_is_stopped_and_answers():
    replies = [_reply(calls=[SEARCH])] * 3 + [_reply("The family has 48,143 members; I could not finish the group.")]
    state, tool, chat = await _run(replies)
    tool.assert_awaited_once()                      # the repeats never ran
    assert chat.await_count == 4                    # call, 2 repeats, forced answer
    assert chat.await_args_list[-1].kwargs["tool_choice"] == "none"
    assert state.status == "completed"
    assert state.final_answer.startswith("The family has 48,143 members")


async def test_every_call_in_the_stuck_response_gets_a_result():
    other = ("search_data", {"collection": "genome", "query": "genus:X"})
    replies = [_reply(calls=[SEARCH]), _reply(calls=[SEARCH]), _reply(calls=[SEARCH, other]), _reply("done")]
    state, tool, chat = await _run(replies)
    forced = chat.await_args_list[-1].kwargs["messages"]
    call_ids = [tc["id"] for m in forced if m.get("role") == "assistant" for tc in m.get("tool_calls") or []]
    result_ids = [m["tool_call_id"] for m in forced if m.get("role") == "tool"]
    assert sorted(call_ids) == sorted(result_ids)   # a valid conversation for the last call
    assert tool.await_count == 1                    # the trailing call was skipped, not run


async def test_an_empty_forced_answer_falls_back_to_a_fixed_text():
    replies = [_reply(calls=[SEARCH])] * 3 + [_reply("")]
    state, _, _ = await _run(replies)
    assert state.status == "completed"
    assert "kept repeating the same lookup" in state.final_answer
    assert "`search_data`" in state.final_answer


async def test_one_repeat_is_still_tolerated():
    replies = [_reply(calls=[SEARCH]), _reply(calls=[SEARCH]), _reply("48,143 members.")]
    state, tool, chat = await _run(replies)
    assert state.final_answer == "48,143 members." and chat.await_count == 3
    assert chat.await_args_list[-1].kwargs.get("tool_choice") != "none"


async def test_planning_loop_is_stopped_the_same_way():
    from planning_agent import agent as planning
    from planning_agent.models import AgentConfig, AgentState

    replies = [_reply(calls=[SEARCH])] * 3 + [_reply("I could not plan this; please name the family.")]
    chat = AsyncMock(side_effect=replies)
    tool = AsyncMock(return_value={"numFound": 1, "results": []})
    state = AgentState(query="q", context={})
    state.add_system_message("sys")
    state.add_user_message("q")
    with patch.object(planning, "chat_completion", chat), \
         patch.dict("shared.tools.registry.TOOL_DISPATCH", {"search_data": tool}):
        result = await planning._run_planning_loop(
            state, MagicMock(), AgentConfig(llm_base_url="http://llm", llm_api_key="k", llm_model="m"), None)
    tool.assert_awaited_once()
    assert chat.await_count == 4 and chat.await_args_list[-1].kwargs["tool_choice"] == "none"
    assert result.status == "completed" and "please name the family" in result.answer
