"""Context budget for agent loops: compaction, a hard stop, overflow recovery.

Why (2026-10-20, testing session #12): every data-agent call starts ~32k
tokens deep (system prompt + 31 tool schemas) and the model's 40k reply
reservation leaves ~91k of a 131k window for the request, so about 60k
tokens of tool results fit in a turn.  Results stayed in the context in
full forever and no loop had a budget: acapria's turn added ~3k tokens per
search until call 23 overflowed and the whole turn was thrown away;
marcnguy's service turns grew to 76k-token prompts whose latency (60-80 s
each under load) ate the 300 s limit.

Three things, used by the shared loop, the service populate loop and the
planning loop:

* ``compact_tool_result`` / ``compact_older_results`` -- a result the model
  has already acted on is kept as its counts, ids, links and a small sample
  (a 3k-token search result becomes ~300), not as 25 full rows.  Compaction
  happens once the predicted prompt passes the soft line, and then for every
  older result at once, so the prompt prefix changes rarely (prefix caching).
* ``ContextBudget`` -- knows the model's window and reply reservation, reads
  the measured ``prompt_tokens`` off each response to calibrate its estimate,
  and says when to compact (soft, 50% of the usable window) and when to stop
  offering tools and answer (hard, 80%).
* ``resolve_context_window`` -- the window from the model server
  (``GET /models`` -> ``max_model_len`` on vLLM), cached; a default when
  unknown.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

from shared.agent_utils import estimate_messages_tokens

logger = logging.getLogger(__name__)

#: Window assumed when the model server does not say (Qwen3.6 / 3.8 on vLLM
#: serve 131072; a larger real window only makes compaction start early).
DEFAULT_CONTEXT_WINDOW = 131072
#: Below this the usable window is treated as this (a wrong reservation must
#: not stop the loop before it starts).
MIN_USABLE_TOKENS = 16000
SOFT_RATIO = 0.5
HARD_RATIO = 0.8

#: A result shorter than this is never compacted (the compact form would not
#: be smaller).
COMPACT_MIN_CHARS = 600
COMPACT_MARK = "_compacted"
COMPACT_NOTE = (
    "Older result compacted: counts, ids and links kept, rows dropped. "
    "Call the tool again if you need the rows."
)
_SAMPLE_ROWS = 3
_SAMPLE_ITEMS = 5
_MAX_IDS = 25
_FACET_BUCKETS = 10
_ID_FIELDS = ("genome_id", "feature_id", "patric_id", "id", "taxon_id", "pdb_id")
_LIST_KEYS = ("items", "files", "workflows", "jobs", "matches", "sample_rows", "inputs", "groups", "samples")
_KEEP_DICT_KEYS = ("_summary", "summary", "distinct_counts", "bucket_counts", "metadata", "column_stats", "stats")


def _ids_of(rows: list) -> Optional[list]:
    for field in _ID_FIELDS:
        values = [r.get(field) for r in rows if isinstance(r, dict) and r.get(field) is not None]
        if values:
            return values
    return None


def compact_tool_result(content: Any) -> Any:
    """The compact form of a tool result the model has already acted on.

    Keeps every scalar (counts, errors, notes, URLs), the first rows of a
    result list plus the ids of all its rows, facet buckets cut to ten per
    field, and the summary dicts file tools attach; drops long lists and
    long strings with a note of their size.  Idempotent; a result below
    COMPACT_MIN_CHARS, or one that would not shrink, is returned unchanged.
    """
    if not isinstance(content, str) or len(content) <= COMPACT_MIN_CHARS:
        return content
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        head = content[:400].rstrip()
        return f"{head} …[compacted: {len(content)} chars; call the tool again for the full text]"
    if isinstance(data, list):
        out: Dict[str, Any] = {"count": len(data), "sample": data[:_SAMPLE_ROWS]}
        ids = _ids_of(data)
        if ids:
            out["ids"] = ids[:_MAX_IDS]
            if len(ids) > _MAX_IDS:
                out["ids_truncated"] = len(ids)
    elif isinstance(data, dict):
        if data.get(COMPACT_MARK):
            return content
        out = _compact_fields(data)
    else:
        return content
    out[COMPACT_MARK] = True
    out["_note"] = COMPACT_NOTE
    compact = json.dumps(out, default=str)
    return compact if len(compact) < len(content) else content


def _compact_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """The dict branch of ``compact_tool_result``; steps into a ``result`` envelope."""
    out: Dict[str, Any] = {}
    for key, value in data.items():
        if key == "result" and isinstance(value, dict):
            # workspace_browse / group tools nest everything here; without
            # this the whole envelope (total, summary, paging) was dropped.
            out[key] = _compact_fields(value)
        elif key == "results" and isinstance(value, list):
            out["results_sample"] = value[:_SAMPLE_ROWS]
            out["results_returned"] = len(value)
            ids = _ids_of(value)
            if ids:
                out["ids"] = ids[:_MAX_IDS]
                if len(ids) > _MAX_IDS:
                    out["ids_truncated"] = len(ids)
        elif key == "facets" and isinstance(value, dict):
            out[key] = {
                f: (b[:_FACET_BUCKETS] if isinstance(b, list) else b) for f, b in value.items()
            }
        elif key in _LIST_KEYS and isinstance(value, list):
            out[f"{key}_sample"] = value[:_SAMPLE_ITEMS]
            out[f"{key}_count"] = len(value)
        elif isinstance(value, str):
            out[key] = value if len(value) <= 300 else value[:300] + "…"
        elif isinstance(value, list):
            out[key] = value if len(json.dumps(value, default=str)) <= 600 else f"[{len(value)} items omitted]"
        elif isinstance(value, dict):
            if key in _KEEP_DICT_KEYS or len(json.dumps(value, default=str)) <= 800:
                out[key] = value
            else:
                out[key] = f"[object with {len(value)} fields omitted]"
        else:
            out[key] = value
    return out


def _last_assistant_index(messages: list) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            return i
    return -1


def compact_older_results(messages: list, include_latest: bool = False) -> int:
    """Compact the tool results in *messages* in place; returns how many changed.

    "Older" means before the latest assistant message: the results of the
    exchange the model is about to react to stay in full.  ``include_latest``
    compacts those too (overflow recovery).
    """
    limit = len(messages) if include_latest else _last_assistant_index(messages)
    changed = 0
    for i in range(max(limit, 0)):
        msg = messages[i]
        if msg.get("role") != "tool":
            continue
        compact = compact_tool_result(msg.get("content"))
        if compact != msg.get("content"):
            msg["content"] = compact
            changed += 1
    return changed


class ContextBudget:
    """The prompt budget of one agent run.

    ``prepare()`` before each LLM call compacts older results past the soft
    line and says whether the hard line is passed (stop offering tools and
    answer).  ``record()`` after each call reads the measured prompt tokens
    and calibrates the estimate (the estimator errs high on prose, low on
    tables; after one call the factor is right for this conversation).
    """

    def __init__(self, window_tokens: int, reply_reserve: int) -> None:
        self.window = int(window_tokens or DEFAULT_CONTEXT_WINDOW)
        self.reserve = int(reply_reserve or 0)
        self.usable = max(self.window - self.reserve, MIN_USABLE_TOKENS)
        self.soft = int(self.usable * SOFT_RATIO)
        self.hard = int(self.usable * HARD_RATIO)
        self.factor = 1.0
        self.measured: Optional[int] = None
        self.compactions = 0
        self.last_predicted = 0

    @classmethod
    def for_config(cls, config: Any) -> "ContextBudget":
        return cls(
            getattr(config, "context_window_tokens", None) or DEFAULT_CONTEXT_WINDOW,
            getattr(config, "max_tokens", 0) or 0,
        )

    def predict(self, messages: list, tools: Optional[list]) -> int:
        return int(estimate_messages_tokens(messages, tools) * self.factor)

    def record(self, response: Any, messages: list, tools: Optional[list]) -> None:
        """Calibrate on the prompt size the server measured for this call."""
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        if prompt_tokens is None and isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")
        if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
            return
        estimated = estimate_messages_tokens(messages, tools)
        if estimated > 0:
            self.factor = min(2.0, max(0.5, prompt_tokens / estimated))
        self.measured = prompt_tokens

    def prepare(self, messages: list, tools: Optional[list]) -> Dict[str, Any]:
        """Compact if past the soft line; report whether the hard line is passed."""
        predicted = self.predict(messages, tools)
        compacted = 0
        if predicted > self.soft:
            compacted = compact_older_results(messages)
            if compacted:
                self.compactions += 1
                predicted = self.predict(messages, tools)
        self.last_predicted = predicted
        return {"predicted": predicted, "compacted": compacted, "exhausted": predicted > self.hard}

    def compact_all(self, messages: list) -> int:
        """Every tool result to its compact form (after an overflow)."""
        changed = compact_older_results(messages, include_latest=True)
        if changed:
            self.compactions += 1
        return changed

    def describe(self) -> str:
        return (
            f"~{self.last_predicted} of {self.usable} usable tokens "
            f"(window {self.window}, reply reserve {self.reserve}, "
            f"measured {self.measured}, {self.compactions} compaction(s))"
        )


# ---------------------------------------------------------------------------
# Context window from the model server
# ---------------------------------------------------------------------------

_WINDOW_CACHE: Dict[str, tuple[float, Optional[int]]] = {}
_WINDOW_TTL_S = 3600.0
_WINDOW_FAIL_TTL_S = 600.0


async def resolve_context_window(base_url: Optional[str], api_key: Optional[str], model: Optional[str]) -> Optional[int]:
    """The model's context window from ``GET <base_url>/models``, or None.

    vLLM reports ``max_model_len`` per model; other OpenAI-compatible servers
    may not, and then the caller falls back to DEFAULT_CONTEXT_WINDOW.
    Cached per base_url (an hour; ten minutes after a failure).  The call
    never raises.
    """
    if not base_url:
        return None
    key = base_url.rstrip("/")
    hit = _WINDOW_CACHE.get(key)
    now = time.monotonic()
    if hit and now < hit[0]:
        return hit[1]
    window: Optional[int] = None
    try:
        import httpx

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{key}/models", headers=headers)
            resp.raise_for_status()
            entries = (resp.json() or {}).get("data") or []
        chosen = next((e for e in entries if e.get("id") == model), None) or (entries[0] if entries else None)
        if chosen:
            value = chosen.get("max_model_len") or chosen.get("context_window") or chosen.get("context_length")
            if isinstance(value, int) and value > 0:
                window = value
    except Exception as exc:  # unknown window is not an error for the turn
        logger.info("Context window lookup failed for %s: %s", key, exc)
    _WINDOW_CACHE[key] = (now + (_WINDOW_TTL_S if window else _WINDOW_FAIL_TTL_S), window)
    return window
