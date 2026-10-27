"""Structured progress steps for the chat UI (testing session 2026-10-16, #5).

Before this, every progress update was a free-text string ("Planning next
step...", "Tool workspace_browse completed.") and the UI replaced one line
with the next.  Users saw little of a 1–3 minute turn.  Now every update is a
**step**: a small dict the frontend can render as a single line, a running
list, a collapsed list or a timeline — the render choice is the frontend's.

A step::

    {
      "step_id": "t3",          # stable id: a later report with the same id
                                # updates the step (running → done)
      "phase": "tool",          # routing | thinking | tool | answer | note
      "status": "running",      # running | done | error
      "label": "Looking up tetracycline-resistant S. pneumoniae genomes",
      "detail": "genome collection",      # optional small text
      "result": "1,234 genomes found",    # set when done
      "agent_label": "Searching BV-BRC data",
      "ts_ms": 4120,            # ms since the agent run started
      "took_ms": 2310,          # set when done
      "transient": true,        # optional: replace the live line, never a row
    }

Phases: ``routing`` (the router), ``thinking`` (a model call), ``tool`` (a
tool call), ``answer`` (writing the reply), ``note`` (something the loop did
on its own: a retry, compaction, a stopped repeat).

How steps reach the UI: the orchestrator's agent executor installs a **sink**
in a context variable before it starts the agent task (``start_run``), so
code deep in the loop — the retry wrapper, the classifier, the budget — can
``report()`` without a callback threaded through every call site.  The
legacy ``emit_progress(cb, progress, total, message)`` keeps working: it
wraps the string as a transient note and goes through the same sink.

Labels come from code (``describe_tool_call`` / ``describe_result``) for all
32 tools; when the model wrote one short sentence next to its tool calls
(``shared/prompts/progress_skill.py``), ``intent_line`` makes it the label
and the code description becomes the detail.  Never show session ids,
submission ids, tokens or URLs here.
"""

from __future__ import annotations

import json
import os
import re
import time
from contextvars import ContextVar
from typing import Any, Awaitable, Callable, Optional

Sink = Callable[[dict], Awaitable[None]]

_SINK: ContextVar[Optional[Sink]] = ContextVar("progress_sink", default=None)
_RUN: ContextVar[Optional[dict]] = ContextVar("progress_run", default=None)

AGENT_LABELS = {
    "data": "Searching BV-BRC data",
    "workspace": "Checking your workspace",
    "service": "Preparing a job",
    "helpdesk": "Looking up documentation",
    "analysis": "Reading job results",
    "planning": "Planning",
}

MAX_LABEL = 140
MAX_DETAIL = 120


# ---------------------------------------------------------------------------
# Run context
# ---------------------------------------------------------------------------


def start_run(sink: Optional[Sink], agent: Optional[str] = None) -> tuple:
    """Install ``sink`` for this context and open a step counter.

    Returns the tokens to pass to ``end_run``.  Tasks created after this call
    copy the context, so the agent task reports into the same sink.
    """
    run = {"started": time.monotonic(), "n": 0, "agent": agent}
    return (_SINK.set(sink), _RUN.set(run))


def end_run(tokens: tuple) -> None:
    sink_token, run_token = tokens
    try:
        _SINK.reset(sink_token)
        _RUN.reset(run_token)
    except ValueError:
        # Reset from another context (the executor's generator was resumed
        # in a different task): leave the values, they are harmless.
        pass


def set_agent(agent: Optional[str]) -> None:
    run = _RUN.get()
    if run is not None:
        run["agent"] = agent


def current_sink() -> Optional[Sink]:
    return _SINK.get()


def agent_label(agent: Optional[str]) -> Optional[str]:
    if not agent:
        return None
    return AGENT_LABELS.get(agent, agent.replace("_", " ").capitalize())


def _run() -> dict:
    run = _RUN.get()
    if run is None:
        run = {"started": time.monotonic(), "n": 0, "agent": None}
        _RUN.set(run)
    return run


def _elapsed_ms() -> int:
    return int((time.monotonic() - _run()["started"]) * 1000)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def new_step(
    phase: str,
    label: str,
    detail: Optional[str] = None,
    *,
    step_id: Optional[str] = None,
    transient: bool = False,
    **extra: Any,
) -> dict:
    """Build a running step with a fresh id (not yet reported)."""
    run = _run()
    if step_id is None:
        run["n"] += 1
        step_id = f"{phase[:1]}{run['n']}"
    step: dict[str, Any] = {
        "step_id": step_id,
        "phase": phase,
        "status": "running",
        "label": _clip(label, MAX_LABEL),
        "_started": time.monotonic(),
    }
    if detail:
        step["detail"] = _clip(detail, MAX_DETAIL)
    if transient:
        step["transient"] = True
    step.update(extra)
    return step


def finish(step: dict, status: str = "done", result: Optional[str] = None, label: Optional[str] = None) -> dict:
    """Mark a step done (or error) in place; the caller reports it."""
    step["status"] = status
    started = step.get("_started")
    if started is not None:
        step["took_ms"] = int((time.monotonic() - started) * 1000)
    if result:
        step["result"] = _clip(result, MAX_DETAIL)
    if label:
        step["label"] = _clip(label, MAX_LABEL)
    return step


def public(step: dict) -> dict:
    """The step as sent to the UI: no private keys, with the run clock."""
    run = _run()
    out = {k: v for k, v in step.items() if not k.startswith("_")}
    out["ts_ms"] = _elapsed_ms()
    label = agent_label(run.get("agent"))
    if label and "agent_label" not in out:
        out["agent_label"] = label
    return out


async def report(step: dict, cb: Any = None) -> None:
    """Send a step to the sink (or to ``cb`` when no sink is installed).

    Errors in the sink never reach the agent.
    """
    sink = _SINK.get()
    payload = public(step)
    try:
        if sink is not None:
            await sink(payload)
        elif cb is not None:
            await cb(0, None, payload)
    except Exception:
        pass


async def note(label: str, detail: Optional[str] = None, *, transient: bool = False, cb: Any = None) -> dict:
    """Report a one-shot note (already done)."""
    step = finish(new_step("note", label, detail, transient=transient))
    await report(step, cb)
    return step


async def start(phase: str, label: str, detail: Optional[str] = None, *, cb: Any = None, **extra: Any) -> dict:
    """Create and report a running step."""
    step = new_step(phase, label, detail, **extra)
    await report(step, cb)
    return step


async def done(step: dict, result: Optional[str] = None, *, status: str = "done", label: Optional[str] = None, cb: Any = None) -> dict:
    finish(step, status=status, result=result, label=label)
    await report(step, cb)
    return step


# Thinking labels: the model call is most of a turn, so say what it is doing.
THINKING_FIRST = "Reading your question and deciding what to look up"
THINKING_NEXT = "Reading the results and deciding the next step"
THINKING_ANSWER = "Writing the answer"
DETAIL_FROM_SO_FAR = "from what was found so far"

# Notes: what a loop did on its own.
NOTE_COMPACTED = "Shortened earlier results to make room"
NOTE_BUDGET_SPENT = "Out of room for more lookups"
NOTE_OVERFLOW = "Shortened earlier results; answering from them"
NOTE_REPEAT_SKIPPED = "Skipped a repeated lookup"
NOTE_STUCK = "Stopped repeating the same lookup"
NOTE_STEP_LIMIT = "Reached the step limit after {n} lookups"
RESULT_TOO_LARGE = "the request grew too large"
RESULT_EMPTY_REPLY = "no text came back; asking again"


def thinking_label(iteration: int, lookups: int = 0) -> str:
    if iteration <= 0 and lookups == 0:
        return THINKING_FIRST
    return THINKING_NEXT


# ---------------------------------------------------------------------------
# The model's own one-line intent
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_MD_RE = re.compile(r"[*_`#>]+")


def intent_line(content: Optional[str]) -> Optional[str]:
    """One short sentence the model wrote next to its tool calls, or None.

    The progress skill asks for under 15 words; a long text, JSON, or a
    multi-paragraph reply is an answer draft, not an intent, and is ignored.
    """
    if not content or not isinstance(content, str):
        return None
    text = _THINK_RE.sub("", content).strip()
    if not text or text[0] in "{[":
        return None
    first = text.splitlines()[0].strip()
    first = _MD_RE.sub("", first).strip().strip("\"'").strip()
    if not first or len(first) > 200 or len(text) > 600:
        return None
    if first.lower().startswith(("i have", "here is", "here are", "the following", "based on")):
        return None
    return first[:MAX_LABEL]


# ---------------------------------------------------------------------------
# Tool call descriptions (code only, all 32 tools)
# ---------------------------------------------------------------------------


def _clip(text: Any, n: int) -> str:
    s = str(text or "").strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _collection(name: Any) -> str:
    s = str(name or "").strip().replace("_", " ")
    return s or "data"


def _base(path: Any) -> str:
    p = str(path or "").rstrip("/")
    if not p or p in (".", "./"):
        return "your home folder"
    return os.path.basename(p) or p


def _n(x: Any) -> str:
    try:
        return f"{int(x):,}"
    except (TypeError, ValueError):
        return str(x)


def _plural(count: int, singular: str, plural: Optional[str] = None) -> str:
    word = singular if count == 1 else (plural or singular + "s")
    return f"{_n(count)} {word}"


def _quote(text: Any, n: int = 60) -> str:
    return "'" + _clip(text, n) + "'"


def humanize_query(query: Any) -> str:
    """A Solr query as readable words: field names unslugged, operators as commas."""
    q = str(query or "").strip()
    if not q or q == "*:*":
        return "everything"
    q = re.sub(r'"([^"]*)"', r"\1", q)
    q = re.sub(r"\(|\)", "", q)
    q = re.sub(r"\s+AND\s+", ", ", q)
    q = re.sub(r"\s+OR\s+", " or ", q)
    q = re.sub(r"(?<![\w])NOT\s+", "not ", q)
    q = re.sub(r"\b([a-z0-9]+(?:_[a-z0-9]+)+):", lambda m: m.group(1).replace("_", " ") + ": ", q)
    q = re.sub(r"\b([a-z]+):(?=\S)", r"\1: ", q)
    q = re.sub(r"\s+", " ", q).strip(" ,")
    return _clip(q, 100)


def workflow_name(workflow_id: Any) -> str:
    """A GoWe workflow id as its name when known (``wf_…`` ids are opaque)."""
    ref = str(workflow_id or "").strip()
    if not ref:
        return "the service"
    try:
        from bvbrc_mcp.tools.gowe import _workflow_names  # populated by list_gowe_workflows
        name = _workflow_names.get(ref)
        if name:
            return name
    except Exception:
        pass
    if ref.startswith("wf_"):
        return "the selected service"
    return ref


def _ids(value: Any, n: int = 3) -> str:
    if isinstance(value, str):
        value = [v.strip() for v in value.split(",") if v.strip()]
    if not isinstance(value, list):
        return ""
    shown = ", ".join(str(v) for v in value[:n])
    if len(value) > n:
        shown += f" and {len(value) - n} more"
    return shown


def describe_tool_call(name: str, args: Optional[dict]) -> tuple[str, Optional[str]]:
    """(label, detail) for a tool call, from its arguments only."""
    a = args or {}
    try:
        return _describe_tool_call(name, a)
    except Exception:
        return (f"Running {name.replace('_', ' ')}", None)


def _describe_tool_call(name: str, a: dict) -> tuple[str, Optional[str]]:
    if name == "search_data":
        coll = _collection(a.get("collection"))
        what = humanize_query(a.get("query"))
        if a.get("count_only"):
            return (f"Counting {coll} records", what)
        return (f"Searching {coll}", what)
    if name == "facet_query":
        coll = _collection(a.get("collection"))
        fields = a.get("facet_fields") or []
        if isinstance(fields, str):
            fields = [fields]
        by = ", ".join(str(f).replace("_", " ") for f in fields[:3])
        label = f"Counting {coll} by {by}" if by else f"Counting {coll}"
        return (label, humanize_query(a.get("query")))
    if name == "probe_data":
        kw = a.get("keywords")
        return (f"Probing {_collection(a.get('collection'))} for {_quote(_ids(kw, 4) or kw)}", None)
    if name == "list_collections":
        return ("Listing the data collections", None)
    if name == "get_collection_fields":
        return (f"Looking up the fields of {_collection(a.get('collection'))}", None)

    if name == "workspace_browse" and a.get("scope") == "shared" and (
            a.get("recursive") is True or a.get("file_extensions") or a.get("workspace_types")):
        return ("Searching the workspaces shared with you",
                f"names containing {_quote(a['name_contains'], 30)}" if a.get("name_contains") else None)
    if name == "workspace_browse" and a.get("scope") in ("shared", "public"):
        label = "Listing workspaces shared with you" if a["scope"] == "shared" else "Listing public workspaces"
        return (label, f"names containing {_quote(a['name_contains'], 30)}" if a.get("name_contains") else None)
    if name == "workspace_browse":
        filtered = any(a.get(k) for k in ("name_contains", "file_extensions", "workspace_types"))
        recursive = a.get("recursive")
        if recursive is None:
            recursive = filtered  # the tool's rule when recursive is not given
        many = a.get("paths") if isinstance(a.get("paths"), list) and len(a["paths"]) > 1 else None
        where = f"{len(many)} folders" if many else _base((a.get("paths") or [None])[0] if a.get("paths") else a.get("path"))
        label = f"{'Searching' if recursive else 'Listing'} {where}"
        filters = []
        if a.get("name_contains"):
            how = {"any": "names containing any of", "word": "names with the word"}.get(a.get("name_match"), "names containing")
            filters.append(f"{how} {_quote(a['name_contains'], 30)}")
        exts = a.get("file_extensions")
        if exts:
            filters.append("." + ", .".join(str(e).lstrip(".") for e in (exts if isinstance(exts, list) else [exts])[:4]) + " files")
        if a.get("created_after"):
            filters.append(f"since {_clip(a['created_after'], 10)}")
        if a.get("created_before"):
            filters.append(f"before {_clip(a['created_before'], 10)}")
        return (label, "; ".join(filters) or None)
    if name == "get_file_metadata":
        return (f"Checking {_base(a.get('path'))}", None)
    if name == "read_file_preview":
        sb = a.get("start_byte") or 0
        detail = None
        if isinstance(sb, (int, float)) and sb > 0:
            detail = f"page {int(sb) // 32768 + 1}"
        return (f"Reading {_base(a.get('path'))}", detail)
    if name == "search_file":
        return (f"Searching {_base(a.get('path'))} for {_quote(a.get('pattern'), 40)}", None)
    if name == "summarize_file":
        detail = []
        if a.get("where"):
            detail.append(f"where {a['where']}")
        if a.get("sort_by"):
            detail.append(f"top by {a['sort_by']}")
        return (f"Summarizing {_base(a.get('path'))}", "; ".join(detail) or None)
    if name == "list_samples":
        if a.get("path"):
            return (f"Finding read pairs in {_base(a.get('path'))}", None)
        ids = a.get("sra_ids") or []
        return (f"Checking {_plural(len(ids) if isinstance(ids, list) else 1, 'SRA accession')}", _ids(ids))

    if name == "list_gowe_workflows":
        return ("Listing the available services", None)
    if name == "get_workflow_inputs":
        return (f"Checking the inputs of {workflow_name(a.get('workflow_id'))}", None)
    if name == "submit_gowe_job":
        return (f"Submitting a {workflow_name(a.get('workflow_id'))} job", _job_detail(a.get("inputs")))
    if name == "rerun_jobs":
        if a.get("folder"):
            return (f"Preparing reruns of the jobs under {_base(a['folder'])}", None)
        ids = a.get("task_ids") or []
        return (f"Preparing a rerun of {_plural(len(ids) if isinstance(ids, list) else 1, 'job')}", _ids(ids))
    if name == "prepare_batch":
        samples = a.get("samples") or []
        n = len(samples) if isinstance(samples, list) else 0
        return (f"Preparing {workflow_name(a.get('workflow_id'))} for {_plural(n, 'sample')}", None)

    if name == "create_group":
        kind = str(a.get("group_type") or "").replace("_group", "").replace("_", " ")
        label = f"Creating {kind} group {_quote(a.get('group_name'), 40)}".replace("  ", " ")
        ids = a.get("ids")
        if isinstance(ids, list) and ids:
            return (label, f"{_plural(len(ids), 'member')}")
        return (label, humanize_query(a.get("query")) if a.get("query") else None)
    if name == "edit_group":
        rem = a.get("remove_ids") or []
        add = a.get("add_ids") or []
        parts = []
        if rem:
            parts.append(f"remove {_plural(len(rem), 'member')}")
        if add:
            parts.append(f"add {_plural(len(add), 'member')}")
        return (f"Changing group {_quote(_base(a.get('group')), 40)}", ", ".join(parts) or None)
    if name == "list_groups":
        kind = str(a.get("group_type") or "").replace("_group", "").replace("_", " ")
        return (f"Listing your {kind} groups".replace("  ", " "), None)
    if name == "get_group_ids":
        group = a.get("group_name")
        if isinstance(group, str) and group.startswith("/"):
            group = _base(group)
        return (f"Reading group {_quote(group, 40)}", None)

    if name == "get_sra_metadata":
        ids = a.get("sra_ids") or []
        n = len(ids) if isinstance(ids, list) else 1
        return (f"Fetching SRA metadata for {_plural(n, 'run')}", _ids(ids))
    if name == "find_similar_genomes":
        return (f"Finding genomes similar to {a.get('genome_id') or _base(a.get('fasta_file'))}", None)
    if name == "search_literature":
        return (f"Searching the literature for {_quote(a.get('query'))}", None)
    if name == "query_helpdesk":
        return (f"Searching the BV-BRC documentation for {_quote(a.get('query'))}", None)
    if name == "list_services":
        return ("Listing the BV-BRC services", None)
    if name == "get_service_schema":
        return (f"Looking up the {a.get('service_name') or 'service'} parameters", None)
    if name == "get_expected_outputs":
        return (f"Looking up what {a.get('service_name') or 'the service'} produces", None)
    if name == "get_job_details":
        ids = a.get("task_ids") or []
        n = len(ids) if isinstance(ids, list) else 1
        return (f"Reading {_plural(n, 'job')}", _ids(ids))
    if name == "list_jobs":
        detail = []
        if a.get("status"):
            detail.append(str(a["status"]))
        if a.get("service"):
            detail.append(str(a["service"]))
        if a.get("search"):
            detail.append(_quote(a["search"], 30))
        return ("Listing your jobs", ", ".join(detail) or None)

    if name == "ask_clarification":
        return ("Asking you a question", None)
    if name == "create_plan":
        return (f"Writing the plan: {a.get('title') or ''}".rstrip(": "), None)
    if name == "list_agents":
        return ("Listing the agents", None)

    return (f"Running {name.replace('_', ' ')}", None)


def _job_detail(inputs: Any) -> Optional[str]:
    """What a job runs on: a genome id, an SRA run, a file name."""
    if not isinstance(inputs, dict):
        return None
    for key in ("genome_id", "genome_ids", "srr_ids", "srr_accession"):
        v = inputs.get(key)
        if v:
            return _ids(v)
    libs = inputs.get("srr_libs") or inputs.get("paired_end_libs") or inputs.get("single_end_libs")
    if isinstance(libs, list) and libs:
        first = libs[0] if isinstance(libs[0], dict) else {}
        v = first.get("srr_accession") or first.get("read1") or first.get("read")
        shown = _base(v) if v else None
        if shown and len(libs) > 1:
            shown += f" and {len(libs) - 1} more"
        return shown
    for key in ("contigs", "fasta_file", "input_file", "reads"):
        v = inputs.get(key)
        if isinstance(v, str) and v:
            return _base(v)
    return None


# ---------------------------------------------------------------------------
# Result descriptions
# ---------------------------------------------------------------------------

_ERROR_WORDS = {
    "PATH_NOT_FOUND": "not found",
    "ALREADY_EXISTS": "already exists",
    "NOT_FOUND": "not found",
    "BINARY_FILE": "binary file",
    "UNSUPPORTED_FILE": "unsupported file",
    "INVALID_PARAMETERS": "invalid parameters",
    "TOO_MANY": "too many to prepare at once",
    "DUPLICATE_GROUP_MEMBERS": "same members as an existing group",
    "ONE_GROUP_PER_STEP": "one group per step",
    "NOT_IN_GROUP": "not in the group",
    "NO_CHANGE": "no change",
    "EMPTY_GROUP": "would leave the group empty",
}


def _count_of(result: dict, keys: tuple) -> Optional[int]:
    for k in keys:
        v = result.get(k)
        if isinstance(v, list):
            return len(v)
        if isinstance(v, int) and not isinstance(v, bool):
            return v
    return None


def describe_result(name: str, args: Optional[dict], result: Any) -> tuple[str, str]:
    """(status, text) for a finished tool call: status is done or error."""
    try:
        return _describe_result(name, args or {}, result)
    except Exception:
        return ("done", "done")


def _describe_result(name: str, a: dict, r: Any) -> tuple[str, str]:
    if not isinstance(r, dict):
        return ("done", "done")

    if r.get("handoff_to"):
        return ("done", f"handing over to: {agent_label(r['handoff_to'])}")
    if r.get("blocked_by_mode"):
        return ("done", "prepared, waiting for your answer")
    if r.get("already_submitted"):
        return ("done", "already submitted a moment ago")
    if r.get("_duplicate") or r.get("_skipped"):
        return ("done", "skipped (repeat)")
    err = r.get("error")
    if err:
        et = r.get("errorType")
        if isinstance(et, str) and et in _ERROR_WORDS:
            return ("error", _ERROR_WORDS[et])
        if isinstance(err, str):
            return ("error", _clip(err.split("\n")[0], 80))
        return ("error", "failed")

    if name == "search_data":
        nf = r.get("numFound")
        if nf is None:
            nf = r.get("count")
        if a.get("count_only") or nf is not None:
            text = f"{_n(nf)} found" if nf is not None else "done"
            rows = _count_of(r, ("results", "items", "docs", "data"))
            if rows is not None and nf is not None and rows < int(nf):
                text += f", showing {_n(rows)}"
            return ("done", text)
        return ("done", "done")
    if name == "facet_query":
        dc = r.get("distinct_counts")
        if isinstance(dc, dict) and dc:
            field, value = next(iter(dc.items()))
            return ("done", f"{_n(value)} distinct {str(field).replace('_', ' ')}")
        nf = r.get("numFound")
        return ("done", f"{_n(nf)} records counted" if nf is not None else "counted")
    if name == "probe_data":
        nf = r.get("numFound")
        return ("done", f"{_n(nf)} found" if nf is not None else "done")
    if name == "list_collections":
        n = _count_of(r, ("collections", "items"))
        return ("done", _plural(n, "collection") if n is not None else "done")
    if name == "get_collection_fields":
        n = _count_of(r, ("fields", "items"))
        return ("done", _plural(n, "field") if n is not None else "done")

    if name == "workspace_browse":
        inner = r.get("result") if isinstance(r.get("result"), dict) else r
        n = _count_of(inner, ("items", "results"))
        # A scope listing lists workspaces; a scope search finds items.
        noun = "workspace" if inner.get("scope") and "result_type" not in inner else "item"
        total = inner.get("total", inner.get("matched"))
        if n is not None and isinstance(total, int) and total > n:
            return ("done", f"{_n(n)} of {_plural(total, noun)}")
        return ("done", _plural(n, noun) if n is not None else "done")
    if name == "get_file_metadata":
        size = r.get("size") or r.get("total_size")
        return ("done", f"{_size(size)}" if size else "done")
    if name == "read_file_preview":
        if r.get("is_complete"):
            return ("done", "read to the end")
        br = r.get("bytes_read")
        return ("done", f"{_size(br)} read" if br else "page read")
    if name == "search_file":
        n = _count_of(r, ("matches",))
        text = _plural(n, "match", "matches") if n is not None else "done"
        if r.get("next_start"):
            text += ", more to scan"
        return ("done", text)
    if name == "summarize_file":
        for key, word in (("row_count", "row"), ("sequence_count", "sequence"), ("read_count", "read"),
                          ("row_count_scanned", "row"), ("sequence_count_scanned", "sequence")):
            if r.get(key) is not None:
                return ("done", _plural(r[key], word))
        return ("done", "summarized")
    if name == "list_samples":
        paired = len(r.get("paired") or [])
        single = len(r.get("single") or [])
        sra = len(r.get("sra") or [])
        parts = []
        if paired:
            parts.append(_plural(paired, "pair"))
        if single:
            parts.append(_plural(single, "single file"))
        if sra:
            parts.append(_plural(sra, "SRA run"))
        return ("done", ", ".join(parts) or "no samples found")

    if name == "list_gowe_workflows":
        n = _count_of(r, ("workflows", "items"))
        return ("done", _plural(n, "service") if n is not None else "done")
    if name == "get_workflow_inputs":
        n = _count_of(r, ("inputs",))
        return ("done", _plural(n, "input") if n is not None else "done")
    if name == "submit_gowe_job":
        return ("done", "submitted")
    if name in ("rerun_jobs", "prepare_batch"):
        n = len(r.get("prepared_actions") or [])
        skipped = len(r.get("skipped") or [])
        text = _plural(n, "job") + " prepared"
        if skipped:
            text += f", {skipped} skipped"
        return ("done", text)

    if name == "create_group":
        n = r.get("count") or r.get("member_count") or _count_of(r, ("ids", "id_list"))
        return ("done", f"created with {_plural(n, 'member')}" if n else "created")
    if name == "edit_group":
        return ("done", "updated")
    if name == "list_groups":
        n = _count_of(r, ("groups", "items"))
        total = r.get("total")
        if n is not None and isinstance(total, int) and total > n:
            return ("done", f"{_n(n)} of {_plural(total, 'group')}")
        return ("done", _plural(n, "group") if n is not None else "done")
    if name == "get_group_ids":
        n = _count_of(r, ("ids", "id_list", "items"))
        return ("done", _plural(n, "member") if n is not None else "done")

    if name == "get_sra_metadata":
        n = _count_of(r, ("results", "items", "runs", "metadata"))
        return ("done", _plural(n, "run") if n is not None else "done")
    if name == "find_similar_genomes":
        n = r.get("count")
        if n is None:
            n = _count_of(r, ("results", "hits"))
        return ("done", _plural(n, "similar genome") if n is not None else "done")
    if name == "search_literature":
        n = _count_of(r, ("papers", "results"))
        return ("done", _plural(n, "paper") if n is not None else "done")
    if name == "query_helpdesk":
        n = _count_of(r, ("count", "results", "documents", "chunks"))
        return ("done", _plural(n, "document") if n is not None else "done")
    if name == "list_services":
        n = _count_of(r, ("services", "items"))
        return ("done", _plural(n, "service") if n is not None else "done")
    if name in ("get_service_schema", "get_expected_outputs"):
        return ("done", "found")
    if name == "get_job_details":
        n = _count_of(r, ("jobs", "tasks", "items"))
        return ("done", _plural(n, "job") if n is not None else "done")
    if name == "list_jobs":
        n = _count_of(r, ("jobs", "items", "tasks"))
        total = r.get("total")
        if n is not None and isinstance(total, int) and total > n:
            return ("done", f"{_n(n)} of {_plural(total, 'job')}")
        return ("done", _plural(n, "job") if n is not None else "done")

    if name == "ask_clarification":
        n = _count_of(r, ("questions",))
        return ("done", _plural(n, "question") if n else "asked")
    if name == "create_plan":
        n = _count_of(r, ("steps",))
        if n is None and isinstance(r.get("plan"), dict):
            n = _count_of(r["plan"], ("steps",))
        return ("done", f"{_plural(n, 'step')}" if n is not None else "plan ready")
    if name == "list_agents":
        return ("done", "done")
    return ("done", "done")


def _size(n: Any) -> str:
    try:
        b = int(n)
    except (TypeError, ValueError):
        return str(n)
    if b < 1024:
        return f"{b} B"
    if b < 1024 * 1024:
        return f"{b / 1024:.0f} KB"
    return f"{b / (1024 * 1024):.1f} MB"


# ---------------------------------------------------------------------------
# Convenience for the loops
# ---------------------------------------------------------------------------


async def tool_start(name: str, args: Optional[dict], intent: Optional[str] = None, *, cb: Any = None) -> dict:
    """Report a running tool step: the model's intent as label when present."""
    label, detail = describe_tool_call(name, args)
    if intent:
        detail = label if not detail else f"{label} — {detail}"
        label = intent
    return await start("tool", label, detail, tool=name, cb=cb)


async def tool_done(step: dict, name: str, args: Optional[dict], result: Any, *, cb: Any = None) -> dict:
    status, text = describe_result(name, args, result)
    return await done(step, text, status=status, cb=cb)


def steps_json(step: dict) -> str:
    return json.dumps(public(step), default=str)
