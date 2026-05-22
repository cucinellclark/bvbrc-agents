"""Tests for the response synthesizer."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from orchestrator.events.events import EventType
from orchestrator.events.stream import collect_events
from orchestrator.llm.client import LLMClient
from orchestrator.models import OrchestratorRequest
from orchestrator.synthesizer.prompts import build_synthesis_prompt
from orchestrator.synthesizer.synthesizer import synthesize


# --- Fixtures ---


def _make_llm_client(response: str = "Synthesized response") -> LLMClient:
    """Create a mock LLM client."""
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock(return_value=response)
    return client


def _make_request(query: str = "test query") -> OrchestratorRequest:
    return OrchestratorRequest(query=query)


# --- Tests: Prompts ---


class TestSynthesisPrompts:
    def test_build_basic_prompt(self):
        system, user = build_synthesis_prompt(
            query="Find E. coli genomes",
            agent_results=[{
                "agent": "data",
                "answer": "Found 100 genomes.",
                "status": "completed",
                "sources": ["genome"],
            }],
        )
        assert "BV-BRC" in system
        assert "E. coli" in user
        assert "Found 100 genomes" in user
        assert "genome" in user  # sources

    def test_build_prompt_with_context(self):
        system, user = build_synthesis_prompt(
            query="What about AMR?",
            agent_results=[{
                "agent": "data",
                "answer": "Found AMR data.",
                "status": "completed",
            }],
            conversation_context="Previously discussed E. coli genomes",
        )
        assert "Previously discussed" in user

    def test_build_prompt_multiple_results(self):
        system, user = build_synthesis_prompt(
            query="Analyze and search",
            agent_results=[
                {
                    "agent": "data",
                    "answer": "Found data.",
                    "status": "completed",
                },
                {
                    "agent": "service2",
                    "answer": "Planned workflow.",
                    "status": "completed",
                },
            ],
        )
        assert "data" in user
        assert "service2" in user
        assert "Found data" in user
        assert "Planned workflow" in user


# --- Tests: synthesize() ---


class TestSynthesize:
    @pytest.mark.asyncio
    async def test_passthrough_single_result(self):
        """Test pass-through for a single successful agent result."""
        llm = _make_llm_client()
        request = _make_request()
        agent_results = [{
            "agent": "data",
            "answer": "Found 100 E. coli genomes in BV-BRC.",
            "status": "completed",
            "sources": ["genome"],
        }]

        events = await collect_events(
            synthesize(request, agent_results, llm)
        )

        event_types = [e.type for e in events]
        assert EventType.SYNTHESIS_START in event_types
        assert EventType.SYNTHESIS_DONE in event_types

        done_event = next(e for e in events if e.type == EventType.SYNTHESIS_DONE)
        assert done_event.data["response_text"] == "Found 100 E. coli genomes in BV-BRC."
        assert done_event.data["method"] == "pass_through"

        # LLM should NOT have been called
        llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_passthrough_max_iterations(self):
        """Test pass-through also works for max_iterations status."""
        llm = _make_llm_client()
        request = _make_request()
        agent_results = [{
            "agent": "data",
            "answer": "Partial results found.",
            "status": "max_iterations",
        }]

        events = await collect_events(
            synthesize(request, agent_results, llm)
        )

        done_event = next(e for e in events if e.type == EventType.SYNTHESIS_DONE)
        assert done_event.data["method"] == "pass_through"

    @pytest.mark.asyncio
    async def test_llm_synthesis_for_error_result(self):
        """Test LLM synthesis when agent returned an error."""
        llm = _make_llm_client("I'm sorry, the search failed.")
        request = _make_request()
        agent_results = [{
            "agent": "data",
            "answer": "Connection timeout",
            "status": "error",
        }]

        events = await collect_events(
            synthesize(request, agent_results, llm)
        )

        done_event = next(e for e in events if e.type == EventType.SYNTHESIS_DONE)
        assert done_event.data["method"] == "llm_synthesis"
        assert "sorry" in done_event.data["response_text"].lower()
        llm.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_synthesis_for_empty_answer(self):
        """Test LLM synthesis when agent returned empty answer."""
        llm = _make_llm_client("No results were found.")
        request = _make_request()
        agent_results = [{
            "agent": "data",
            "answer": "",
            "status": "completed",
        }]

        events = await collect_events(
            synthesize(request, agent_results, llm)
        )

        done_event = next(e for e in events if e.type == EventType.SYNTHESIS_DONE)
        assert done_event.data["method"] == "llm_synthesis"

    @pytest.mark.asyncio
    async def test_force_llm_synthesis(self):
        """Test forcing LLM synthesis even for good single results."""
        llm = _make_llm_client("Enhanced response.")
        request = _make_request()
        agent_results = [{
            "agent": "data",
            "answer": "Found data.",
            "status": "completed",
        }]

        events = await collect_events(
            synthesize(request, agent_results, llm, force_llm=True)
        )

        done_event = next(e for e in events if e.type == EventType.SYNTHESIS_DONE)
        assert done_event.data["method"] == "llm_synthesis"
        llm.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_synthesis_failure_fallback(self):
        """Test fallback when synthesis LLM call fails."""
        llm = _make_llm_client()
        llm.complete = AsyncMock(side_effect=Exception("LLM down"))
        request = _make_request()
        agent_results = [{
            "agent": "data",
            "answer": "",
            "status": "error",
        }]

        events = await collect_events(
            synthesize(request, agent_results, llm)
        )

        done_event = next(e for e in events if e.type == EventType.SYNTHESIS_DONE)
        assert done_event.data["method"] == "fallback"

    @pytest.mark.asyncio
    async def test_no_results(self):
        """Test synthesis with no agent results."""
        llm = _make_llm_client("No results to summarize.")
        request = _make_request()

        events = await collect_events(
            synthesize(request, [], llm)
        )

        done_event = next(e for e in events if e.type == EventType.SYNTHESIS_DONE)
        assert done_event.data["method"] == "llm_synthesis"
