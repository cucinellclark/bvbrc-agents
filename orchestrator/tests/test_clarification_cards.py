"""Any agent's ask_clarification questions reach the user as the questions card.

2026-10-20: "I want to download influenza protein data" -> the data agent
called ask_clarification with two good questions, but only the planning agent
passed questions on, so the user saw "I have some questions before I can
proceed." and nothing else.  Now every agent's questions become
``clarification_questions`` (dispatch), the orchestrator emits ASK_QUESTIONS
naming the agent that asked, and a paused plan step carries them too.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import shared.agent_dispatch as agent_dispatch  # noqa: E402  -- sets up agent package paths
from shared.agent_loop import run_agent_loop  # noqa: E402
from shared.agent_utils import (  # noqa: E402
    CLARIFICATION_INTRO,
    format_clarification_questions,
    parse_clarification_questions,
)
from shared.models import BaseAgentConfig, BaseAgentState  # noqa: E402

from tests.test_plan_steps import PLAN_CTX, _agent_reply, _registry, _run, _types  # noqa: E402

ASK = ("ask_clarification", {"questions": [
    {"question": "Which <data kind>?", "options": ["<option A>", "<option B>"]},
    {"question": "Which <scope>?", "options": ["<scope A>", "<scope B>"]},
]})
QUESTIONS = [
    {"id": "q_1", "question": "Which <data kind>?", "options": ["<option A>", "<option B>"], "required": True},
    {"id": "q_2", "question": "Which <scope>?", "options": ["<scope A>", "<scope B>"], "required": True},
]


def _reply(content="", calls=()):
    tcs = [SimpleNamespace(id=f"c{i}", type="function",
                           function=SimpleNamespace(name=n, arguments=json.dumps(a)))
           for i, (n, a) in enumerate(calls)]
    msg = SimpleNamespace(content=content, tool_calls=tcs or None, reasoning=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls" if tcs else "stop")])


# ---- helpers ------------------------------------------------------------------

def test_parse_accepts_the_stored_json_and_rejects_plain_text():
    assert parse_clarification_questions(json.dumps(QUESTIONS)) == QUESTIONS
    assert parse_clarification_questions(QUESTIONS) == QUESTIONS
    assert parse_clarification_questions("Which genome do you mean?") is None
    assert parse_clarification_questions("[not json") is None
    assert parse_clarification_questions("[]") is None
    assert parse_clarification_questions(json.dumps([{"options": ["a"]}])) is None
    assert parse_clarification_questions(None) is None


def test_format_lists_questions_and_options():
    assert format_clarification_questions(QUESTIONS[:1]) == "1. Which <data kind>?\n   - <option A>\n   - <option B>"


# ---- the shared loop (data, workspace, helpdesk, analysis) ----------------------

async def test_shared_loop_stops_on_a_valid_question_with_the_intro():
    state = BaseAgentState(query="I want to download <data>", context={})
    chat = AsyncMock(side_effect=[_reply(calls=[ASK])])
    cfg = BaseAgentConfig(llm_base_url="http://llm", llm_api_key="k", llm_model="m")
    await run_agent_loop(state=state, config=cfg, client=MagicMock(), system_prompt="sys",
                         chat_completion_fn=chat)
    assert state.status == "needs_input"
    assert state.final_answer == CLARIFICATION_INTRO
    assert json.loads(state.question) == QUESTIONS


# ---- dispatch -------------------------------------------------------------------

def _response(status="needs_input", answer=CLARIFICATION_INTRO):
    return {"answer": answer, "status": status, "tool_trace": []}


def test_dispatch_passes_any_agents_questions_on():
    response = _response()
    agent_dispatch._with_blocked_actions(response, SimpleNamespace(question=json.dumps(QUESTIONS)))
    assert response["clarification_questions"] == QUESTIONS
    assert response["question"].startswith("1. Which <data kind>?")
    assert response["answer"] == CLARIFICATION_INTRO


def test_the_raw_json_never_becomes_the_answer():
    raw = json.dumps(QUESTIONS)
    response = _response(answer=raw)          # the service dispatch uses question as answer
    agent_dispatch._with_blocked_actions(response, SimpleNamespace(question=raw))
    assert response["answer"] == CLARIFICATION_INTRO


def test_the_models_own_text_is_kept():
    response = _response(answer="Influenza protein data comes in several forms.")
    agent_dispatch._with_blocked_actions(response, SimpleNamespace(question=json.dumps(QUESTIONS)))
    assert response["answer"] == "Influenza protein data comes in several forms."


def test_plain_text_questions_and_other_statuses_are_left_alone():
    response = _response(answer="Which genome?")
    agent_dispatch._with_blocked_actions(response, SimpleNamespace(question="Which genome?"))
    assert "clarification_questions" not in response
    response = _response(status="completed")
    agent_dispatch._with_blocked_actions(response, SimpleNamespace(question=json.dumps(QUESTIONS)))
    assert "clarification_questions" not in response


def test_planning_questions_are_not_replaced():
    mine = [{"id": "q_1", "question": "Plan?", "options": ["a", "b"]}]
    response = {**_response(), "clarification_questions": mine}
    agent_dispatch._with_blocked_actions(response, SimpleNamespace(question=json.dumps(QUESTIONS)))
    assert response["clarification_questions"] == mine


# ---- the service agent's own loop -------------------------------------------------

async def test_service_populate_stops_on_a_valid_question():
    from service_agent.models import AgentConfig, AgentState
    from service_agent.phases import populate

    cfg = AgentConfig(llm_base_url="http://llm", llm_api_key="k", llm_model="m", max_iterations=5)
    state = AgentState(query="assemble my reads", context={})
    chat = AsyncMock(side_effect=[_reply(calls=[ASK])])
    with patch.object(populate, "create_client", lambda c: MagicMock()), \
         patch.object(populate, "chat_completion", chat):
        out = await populate.populate_and_submit(state.query, cfg, state)
    assert chat.await_count == 1
    assert out.status == "needs_input"
    assert json.loads(out.question) == QUESTIONS


# ---- orchestrator events ------------------------------------------------------------

async def test_a_data_agent_question_becomes_the_card_for_the_data_agent():
    registry = _registry("data")
    registry._agents["data"].call_tool = AsyncMock(return_value=_agent_reply(
        status="needs_input", answer=CLARIFICATION_INTRO, clarification_questions=QUESTIONS,
        question=format_clarification_questions(QUESTIONS)))
    events = await _run(registry, query="I want to download <data>", target_agent="data")
    types = _types(events)
    ask = next(e for e in events if e.type.value == "ask_questions")
    assert ask.data == {"agent": "data", "questions": QUESTIONS}
    assert "needs_input" not in types                 # no second, text-only copy
    assert events[-1].data["response_text"] == CLARIFICATION_INTRO


async def test_a_plan_steps_questions_ride_on_its_needs_input_event():
    registry = _registry("planning", "data")
    registry._agents["planning"].call_tool = AsyncMock(return_value=_agent_reply(
        status="step_ready", step_execution={**PLAN_CTX, "agent": "data", "task": "Find the genomes"}))
    registry._agents["data"].call_tool = AsyncMock(return_value=_agent_reply(
        status="needs_input", answer=CLARIFICATION_INTRO, clarification_questions=QUESTIONS,
        question=format_clarification_questions(QUESTIONS)))
    events = await _run(registry, query="continue", target_agent="planning",
                        workflow_context={"plan_action": "execute_next"})
    paused = next(e for e in events if e.type.value == "plan_step_needs_input")
    assert paused.data["questions"] == QUESTIONS
    assert paused.data["question"].startswith("1. Which <data kind>?")
    assert paused.data["agent"] == "data"
