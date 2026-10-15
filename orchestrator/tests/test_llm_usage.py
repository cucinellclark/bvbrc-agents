"""Per-turn LLM usage (shared/usage.py): every model call a turn makes is counted.

Before this nothing counted LLM calls or tokens; a data step that repeated
one search ~90 times was found only because a benchmark arm ran behind a
logging proxy.  The orchestrator opens a tally per request, the retry
wrappers record every attempt, and ORCHESTRATOR_DONE carries the summary.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import shared.agent_dispatch  # noqa: E402, F401  -- sets up agent package paths
from shared import usage  # noqa: E402
from shared.agent_utils import llm_call_with_retry, llm_stream_with_retry  # noqa: E402
from orchestrator import orchestrate as orch  # noqa: E402
from orchestrator.events.events import Event, EventType  # noqa: E402

pytestmark = pytest.mark.asyncio


def _response(prompt=100, completion=20, reasoning=None, model="Qwen3.8-27B"):
    details = SimpleNamespace(reasoning_tokens=reasoning) if reasoning is not None else None
    return SimpleNamespace(
        model=model, choices=[],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                              total_tokens=prompt + completion, completion_tokens_details=details),
    )


async def _in_turn(coro_fn):
    """Run coro_fn in a fresh context with a turn open; return (result, calls)."""
    async def run():
        calls = usage.start_turn()
        return await coro_fn(), calls
    return await asyncio.create_task(run())


async def test_a_call_is_recorded_with_its_tokens_model_and_agent():
    async def go():
        token = usage.CURRENT_AGENT.set("data")
        try:
            return await llm_call_with_retry(AsyncMock(return_value=_response(reasoning=7)))
        finally:
            usage.CURRENT_AGENT.reset(token)
    _, calls = await _in_turn(go)
    (c,) = calls
    assert c["site"] == "agent" and c["agent"] == "data" and c["model"] == "Qwen3.8-27B"
    assert (c["prompt_tokens"], c["completion_tokens"], c["reasoning_tokens"]) == (100, 20, 7)
    assert c["latency_ms"] >= 0 and "error" not in c and "retry" not in c


async def test_a_failed_attempt_and_its_retry_are_both_counted():
    fn = AsyncMock(side_effect=[ConnectionError("connection reset"), _response()])
    with patch("shared.agent_utils.asyncio.sleep", AsyncMock()):
        _, calls = await _in_turn(lambda: llm_call_with_retry(fn, _usage_site="classifier"))
    assert [c["site"] for c in calls] == ["classifier", "classifier"]
    assert "connection reset" in calls[0]["error"]
    assert calls[1].get("retry") is True and calls[1]["prompt_tokens"] == 100


async def test_a_stream_is_recorded_when_it_ends_with_the_last_chunks_usage():
    chunks = [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hi"))], usage=None, model="m"),
              SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=50, completion_tokens=2), model="m")]

    class Stream:
        def __init__(self):
            self._it = iter(chunks)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._it)
            except StopIteration:
                raise StopAsyncIteration

    async def go():
        stream = await llm_stream_with_retry(AsyncMock(return_value=Stream()))
        assert usage.current_calls() == []          # not yet: the stream is open
        return [c async for c in stream]
    out, calls = await _in_turn(go)
    assert len(out) == 2
    (c,) = calls
    assert c["site"] == "agent_stream" and c["prompt_tokens"] == 50 and "ttft_ms" in c


async def test_outside_a_turn_nothing_is_recorded():
    async def go():
        return await llm_call_with_retry(AsyncMock(return_value=_response()))
    assert (await asyncio.create_task(go())).usage.prompt_tokens == 100   # no turn: no error


async def test_summary_totals_by_site_and_agent():
    calls = [
        {"site": "router", "agent": None, "model": "r", "latency_ms": 900, "prompt_tokens": 2000, "completion_tokens": 30},
        {"site": "classifier", "agent": "service", "model": "q", "latency_ms": 1500, "prompt_tokens": 800, "completion_tokens": 40},
        {"site": "agent", "agent": "service", "model": "q", "latency_ms": 9000, "prompt_tokens": 12000,
         "completion_tokens": 300, "reasoning_tokens": 200},
        {"site": "agent", "agent": "service", "model": "q", "latency_ms": 30, "error": "Timeout", "retry": True},
    ]
    s = usage.summarize(calls)
    assert s["llm_calls"] == 4 and s["prompt_tokens"] == 14800 and s["completion_tokens"] == 370
    assert s["reasoning_tokens"] == 200 and s["errors"] == 1 and s["retries"] == 1 and s["llm_ms"] == 11430
    assert s["by_site"]["agent"]["calls"] == 2 and s["by_agent"] == {
        "service": {"calls": 3, "prompt_tokens": 12800, "completion_tokens": 340}}
    assert s["models"] == ["r", "q"] and len(s["calls"]) == 4
    assert usage.log_line(s).startswith("LLM_USAGE calls=4 prompt=14800 completion=370")
    many = usage.summarize([{"site": "agent", "latency_ms": 1}] * 250)
    assert many["llm_calls"] == 250 and len(many["calls"]) == 200 and many["calls_truncated"]


async def _fake_turn(*args, **kwargs):
    """Router call in the turn's task, an agent in its own task (as
    InProcessAgentHandle runs it), then DONE."""
    await llm_call_with_retry(AsyncMock(return_value=_response(2000, 30)), _usage_site="router")

    async def agent():
        usage.CURRENT_AGENT.set("data")
        await llm_call_with_retry(AsyncMock(return_value=_response(5000, 100)))
        await llm_call_with_retry(AsyncMock(return_value=_response(6000, 200)))
        await asyncio.sleep(10)                     # cut off by the 300 s limit

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(agent(), timeout=0.05)
    yield Event(type=EventType.ORCHESTRATOR_DONE, data={"response_text": "ok", "decision": "agent"})


async def test_done_carries_the_turn_usage_including_a_timed_out_agents_calls():
    req = SimpleNamespace(session_id="s1")
    with patch.object(orch, "_orchestrate", _fake_turn):
        events = [e async for e in orch.orchestrate(req, None, None)]
    u = events[-1].data["usage"]
    assert u["llm_calls"] == 3 and u["prompt_tokens"] == 13000
    assert u["by_site"]["router"]["calls"] == 1 and u["by_agent"]["data"]["calls"] == 2
    assert events[-1].data["response_text"] == "ok"


async def test_orchestrate_to_response_returns_the_usage():
    with patch.object(orch, "_orchestrate", _fake_turn):
        resp = await orch.orchestrate_to_response(SimpleNamespace(session_id="s1"), None, None)
    assert resp.usage["llm_calls"] == 3


async def test_agent_streams_ask_vllm_for_token_counts():
    from data_agent import llm_client
    from data_agent.models import AgentConfig

    class Empty:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=Empty())
    cfg = AgentConfig(llm_base_url="http://llm", llm_api_key="k", llm_model="m")
    [c async for c in llm_client.chat_completion_stream(client, [{"role": "user", "content": "x"}], cfg)]
    kwargs = client.chat.completions.create.await_args.kwargs
    assert kwargs["stream"] is True and kwargs["stream_options"] == {"include_usage": True}


async def test_router_and_synthesizer_calls_go_through_the_shared_wrapper():
    """Live, 2026-10-15: "Hello!" counted 0 calls.  The orchestrator client's
    import of the shared wrapper failed (repo root not yet on sys.path) and a
    silent no-op fallback ran instead: no count, and no retry, ever."""
    from orchestrator.llm import client as llm_client_mod
    from orchestrator.llm.client import LLMClient
    from orchestrator.llm.config import LLMConfig

    assert llm_client_mod.llm_call_with_retry is llm_call_with_retry
    llm = LLMClient(LLMConfig(base_url="http://llm/v1", api_key="k", model="router-model"))
    resp = _response(1500, 25, model="router-model")
    resp.choices = [SimpleNamespace(message=SimpleNamespace(content='{"decision": "direct"}'))]
    llm._client = MagicMock()
    llm._client.chat.completions.create = AsyncMock(return_value=resp)
    _, calls = await _in_turn(lambda: llm.complete("hi", usage_site="router"))
    assert [(c["site"], c["prompt_tokens"], c["model"]) for c in calls] == [("router", 1500, "router-model")]
