"""Per-turn LLM usage: every model call a chat turn makes, counted.

``current_plans/llm_usage_tracking.md``.  Before this nothing counted LLM
calls or tokens; the first benchmark run found a step that repeated one
search ~90 times (2.4 M prompt tokens) only because its arm ran behind a
logging proxy.

How it works:

- The orchestrator opens a tally per request (``start_turn``); it lives in a
  context variable, so the router, the agents (in-process, in their own
  tasks) and the synthesizer all record into the same list, and a turn cut
  off by the 300 s agent limit keeps what it counted.
- ``llm_call_with_retry`` / ``llm_stream_with_retry`` (every agent call, the
  classifier) and the orchestrator ``LLMClient`` (router, synthesizer) record
  each attempt: site, agent, model, tokens, latency, retry, error.  A
  streamed call is recorded when its stream ends, with ``ttft_ms``; agents ask
  vLLM for the token counts (``stream_options.include_usage``).
- ``summarize`` turns the list into the ``usage`` stored on the assistant
  message.  Outside a turn (tests, scripts) recording is a no-op.
"""

from __future__ import annotations

import time
from contextvars import ContextVar
from typing import Any, Optional

# The current turn's call list (None outside a turn).
_TURN: ContextVar[Optional[list]] = ContextVar("llm_usage_turn", default=None)
# The agent running (set by dispatch_agent), for attribution.
CURRENT_AGENT: ContextVar[Optional[str]] = ContextVar("llm_usage_agent", default=None)

# Keep the stored per-call list bounded; totals always count every call.
MAX_STORED_CALLS = 200


def start_turn() -> list:
    """Open a tally for this context and return it."""
    calls: list = []
    _TURN.set(calls)
    return calls


def current_calls() -> Optional[list]:
    return _TURN.get()


def _usage_numbers(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}

    def get(obj: Any, name: str) -> Any:
        return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)

    out = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = get(usage, key)
        if isinstance(value, int):
            out[key] = value
    details = get(usage, "completion_tokens_details")
    reasoning = get(details, "reasoning_tokens") if details is not None else None
    if isinstance(reasoning, int):
        out["reasoning_tokens"] = reasoning
    return out


def record_call(
    site: str,
    *,
    started: float,
    response: Any = None,
    usage: Any = None,
    model: Optional[str] = None,
    error: Optional[str] = None,
    retry: bool = False,
    ttft_ms: Optional[int] = None,
) -> None:
    """Append one call to the current turn's tally (no-op outside a turn)."""
    calls = _TURN.get()
    if calls is None:
        return
    if usage is None and response is not None:
        usage = getattr(response, "usage", None)
    if model is None and response is not None:
        model = getattr(response, "model", None)
    entry: dict[str, Any] = {
        "site": site,
        "agent": CURRENT_AGENT.get(),
        "model": model,
        "latency_ms": int((time.monotonic() - started) * 1000),
        **_usage_numbers(usage),
    }
    if ttft_ms is not None:
        entry["ttft_ms"] = ttft_ms
    if retry:
        entry["retry"] = True
    if error:
        entry["error"] = error[:200]
    calls.append(entry)


class UsageStream:
    """Wraps an async stream of chunks; records the call when it ends.

    The token counts come from the final chunk when the request asked for
    ``stream_options={"include_usage": true}``.
    """

    def __init__(self, stream: Any, site: str, started: float, retry: bool = False):
        self._stream = stream
        self._site = site
        self._started = started
        self._retry = retry
        self._first: Optional[float] = None
        self._usage: Any = None
        self._model: Optional[str] = None
        self._done = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self._stream.__anext__()
        except StopAsyncIteration:
            self._finish(None)
            raise
        except Exception as e:
            self._finish(f"{type(e).__name__}: {e}")
            raise
        if self._first is None:
            self._first = time.monotonic()
        if getattr(chunk, "usage", None) is not None:
            self._usage = chunk.usage
        if self._model is None:
            self._model = getattr(chunk, "model", None)
        return chunk

    def _finish(self, error: Optional[str]) -> None:
        if self._done:
            return
        self._done = True
        ttft = int((self._first - self._started) * 1000) if self._first else None
        record_call(self._site, started=self._started, usage=self._usage, model=self._model,
                    error=error, retry=self._retry, ttft_ms=ttft)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def wrap_result(result: Any, site: str, started: float, retry: bool) -> Any:
    """Record a finished call now, or wrap a stream to record when it ends."""
    if hasattr(result, "__aiter__") and not hasattr(result, "choices"):
        return UsageStream(result, site, started, retry)
    record_call(site, started=started, response=result, retry=retry)
    return result


def summarize(calls: Optional[list]) -> dict[str, Any]:
    """``usage`` for the assistant message: totals, by site and agent, the calls."""
    calls = calls or []
    total = {"llm_calls": len(calls), "prompt_tokens": 0, "completion_tokens": 0,
             "reasoning_tokens": 0, "llm_ms": 0, "errors": 0, "retries": 0}
    by_site: dict[str, dict[str, int]] = {}
    by_agent: dict[str, dict[str, int]] = {}
    models: list[str] = []
    for c in calls:
        for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
            total[key] += c.get(key, 0)
        total["llm_ms"] += c.get("latency_ms", 0)
        total["errors"] += 1 if c.get("error") else 0
        total["retries"] += 1 if c.get("retry") else 0
        for bucket, key in ((by_site, c.get("site") or "?"), (by_agent, c.get("agent"))):
            if key is None:
                continue
            b = bucket.setdefault(key, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
            b["calls"] += 1
            b["prompt_tokens"] += c.get("prompt_tokens", 0)
            b["completion_tokens"] += c.get("completion_tokens", 0)
        if c.get("model") and c["model"] not in models:
            models.append(c["model"])
    return {
        **total,
        "by_site": by_site,
        "by_agent": by_agent,
        "models": models,
        "calls": calls[:MAX_STORED_CALLS],
        **({"calls_truncated": True} if len(calls) > MAX_STORED_CALLS else {}),
    }


def log_line(summary: dict[str, Any]) -> str:
    """One grep-able line: ``LLM_USAGE calls=12 prompt=48000 ...``."""
    sites = ",".join(f"{k}:{v['calls']}" for k, v in summary.get("by_site", {}).items())
    agents = ",".join(f"{k}:{v['calls']}" for k, v in summary.get("by_agent", {}).items())
    return (
        f"LLM_USAGE calls={summary['llm_calls']} prompt={summary['prompt_tokens']} "
        f"completion={summary['completion_tokens']} reasoning={summary['reasoning_tokens']} "
        f"llm_ms={summary['llm_ms']} errors={summary['errors']} retries={summary['retries']} "
        f"sites={sites or '-'} agents={agents or '-'}"
    )
