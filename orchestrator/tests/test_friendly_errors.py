"""User-facing error text for failed agent runs.

Hackathon users saw "**data**: Agent 'data' execution failed: " (an asyncio
timeout has an empty str) and "Agent 'service' returned an error: LLM
produced no tool calls and no text." These tests pin what they see now.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from orchestrator.events.events import EventType
from orchestrator.events.stream import collect_events
from orchestrator.executor.agent_executor import (
    execute_agent_step,
    user_facing_agent_error,
)
from orchestrator.models import OrchestratorRequest
from orchestrator.router.models import Step
from orchestrator.synthesizer.synthesizer import synthesize


class _FakeAgent:
    """Just enough of an agent handle for execute_agent_step."""

    def __init__(self, key: str, *, raises: BaseException | None = None, result=None):
        self.key = key
        self.config = SimpleNamespace(
            chat_tool="agent_chat", chat_tool_params={}, mcp_server_name=None,
            timeout_seconds=300,
        )
        self.tool_names = ["agent_chat"]
        self._raises = raises
        self._result = result

    async def call_tool(self, name, arguments, progress_handler=None):
        if self._raises is not None:
            raise self._raises
        return self._result


def _request() -> OrchestratorRequest:
    return OrchestratorRequest(query="how many E. coli genomes resist ciprofloxacin?")


async def _error_message(agent: _FakeAgent) -> dict:
    events = await collect_events(
        execute_agent_step(Step(agent_key=agent.key, task="q"), agent, _request())
    )
    errors = [e for e in events if e.type == EventType.ORCHESTRATOR_ERROR]
    assert len(errors) == 1
    return errors[0].data


def test_timeout_message_names_the_limit_and_a_next_step():
    msg = user_facing_agent_error("data", asyncio.TimeoutError(), 300)
    assert "5-minute limit" in msg
    assert "data search" in msg
    assert "narrowing" in msg
    assert "execution failed" not in msg


def test_unknown_agent_key_is_used_as_is():
    assert "the mystery step" in user_facing_agent_error("mystery", ValueError("x"), None)


@pytest.mark.asyncio
async def test_agent_timeout_is_explained_not_blank():
    data = await _error_message(_FakeAgent("data", raises=asyncio.TimeoutError()))
    assert "took longer than the 5-minute limit" in data["error"]
    assert "Agent 'data'" not in data["error"]
    assert data["internal_error"].startswith("TimeoutError")


@pytest.mark.asyncio
async def test_unexpected_exception_hides_internals_from_the_user():
    data = await _error_message(
        _FakeAgent("service", raises=KeyError("srr_accession"))
    )
    assert "srr_accession" not in data["error"]
    assert "unexpected problem" in data["error"]
    assert "srr_accession" in data["internal_error"]


@pytest.mark.asyncio
async def test_agent_error_answer_is_shown_without_prefix():
    agent = _FakeAgent("service", result={
        "status": "error",
        "answer": "The job was NOT submitted. Submission failed with: bad genome id",
    })
    data = await _error_message(agent)
    assert data["error"] == "The job was NOT submitted. Submission failed with: bad genome id"


@pytest.mark.asyncio
async def test_single_error_result_passes_through_without_llm():
    llm = MagicMock()
    llm.complete_stream = MagicMock(side_effect=AssertionError("no synthesis call"))
    results = [{"agent": "data", "status": "error",
                "answer": "Sorry — the data search step took longer than the 5-minute limit."}]
    events = await collect_events(synthesize(_request(), results, llm))
    done = next(e for e in events if e.type == EventType.SYNTHESIS_DONE)
    assert done.data["method"] == "pass_through"
    assert done.data["response_text"].startswith("Sorry — the data search step")
    assert "**data**" not in done.data["response_text"]
