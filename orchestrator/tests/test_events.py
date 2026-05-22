"""Tests for the event system."""

import pytest

from orchestrator.events.events import (
    Event,
    EventType,
    discovery_event,
    agent_start_event,
    agent_result_event,
    error_event,
    health_event,
)
from orchestrator.events.stream import collect_events, merge_streams


def test_event_creation():
    """Test basic event creation."""
    event = Event(type=EventType.AGENT_START, data={"agent": "data"})
    assert event.type == EventType.AGENT_START
    assert event.data["agent"] == "data"
    assert event.id  # Auto-generated
    assert event.timestamp > 0


def test_event_str():
    event = Event(type=EventType.AGENT_START, agent_name="data")
    assert "AGENT_START" in str(event).upper() or "agent_start" in str(event)
    assert "data" in str(event)


def test_discovery_event():
    event = discovery_event("data", 5)
    assert event.type == EventType.DISCOVERY_AGENT
    assert event.data["agent"] == "data"
    assert event.data["tool_count"] == 5


def test_agent_start_event():
    event = agent_start_event("service", "build a workflow")
    assert event.type == EventType.AGENT_START
    assert event.agent_name == "service"
    assert event.data["task"] == "build a workflow"


def test_agent_result_event():
    event = agent_result_event("data", "Found 10 genomes", {"count": 10})
    assert event.type == EventType.AGENT_RESULT
    assert event.data["result_for_llm"] == "Found 10 genomes"
    assert event.data["result_for_ui"]["count"] == 10


def test_error_event():
    event = error_event("connection failed", agent_name="data")
    assert event.type == EventType.ORCHESTRATOR_ERROR
    assert "connection failed" in event.data["error"]
    assert event.agent_name == "data"


def test_health_event():
    event = health_event("data", True, 42.5)
    assert event.type == EventType.HEALTH_CHECK
    assert event.data["healthy"] is True
    assert event.data["latency_ms"] == 42.5


@pytest.mark.asyncio
async def test_collect_events():
    """Test collecting events from an async generator."""
    async def gen():
        yield Event(type=EventType.AGENT_START, data={"n": 1})
        yield Event(type=EventType.AGENT_RESULT, data={"n": 2})

    events = await collect_events(gen())
    assert len(events) == 2
    assert events[0].type == EventType.AGENT_START
    assert events[1].type == EventType.AGENT_RESULT


@pytest.mark.asyncio
async def test_merge_streams():
    """Test merging multiple event streams."""
    async def stream_a():
        yield Event(type=EventType.AGENT_START, data={"source": "a"})

    async def stream_b():
        yield Event(type=EventType.AGENT_START, data={"source": "b"})

    merged = await collect_events(merge_streams(stream_a(), stream_b()))
    assert len(merged) == 2
    sources = {e.data["source"] for e in merged}
    assert sources == {"a", "b"}
