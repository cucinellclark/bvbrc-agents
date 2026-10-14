"""Structured outputs of a plan step (D4 Phase 3, multi_step_requests.md §4.2 D).

A plan step used to hand the next step only its answer text (cut to 4000
chars) plus whatever ``structured_data`` the data agent happened to build;
``submission_ids``, output folders and group paths never crossed a step, and
the planner paraphrasing the answer could drop an id.  ``step_outputs`` is
built deterministically from the step's tool results -- no LLM -- and is
injected verbatim into the prompt of every step that depends on it::

    {
      "ids":            [{"collection", "field", "values": [...<=200], "total"}],
      "groups":         [{"path", "type", "count", "selection": {"query"} | {"ids_count"}}],
      "paths":          ["/u/home/..."],          # files / folders the step used
      "submission_ids": ["sub_..."], "output_paths": ["..."],
      "values":         {"plfam_id": "PLF_..."},  # named scalars (``expects``)
    }
"""

from __future__ import annotations

import re
from typing import Any, Iterable

MAX_IDS = 200
MAX_PATHS = 20

_ID_FIELDS = ("feature_id", "genome_id", "patric_id", "id")

# Known identifier shapes for ``expects`` (create_plan) -> values.  Anything
# else stays in the answer text.
VALUE_PATTERNS: dict[str, re.Pattern] = {
    "plfam_id": re.compile(r"\bPLF_\d+_\d+\b"),
    "pgfam_id": re.compile(r"\bPGF_\d+\b"),
    # fig|... is a PATRIC id (patric_id); feature_id is PATRIC.<genome>.<...>.
    "patric_id": re.compile(r"fig\|\d+\.\d+\.[a-z]+\.\d+"),
    "feature_id": re.compile(r"\b(?:PATRIC|RefSeq)\.\d+\.\d+\.[\w.\-]+"),
    "genome_id": re.compile(r"(?<![\w.])\d+\.\d+(?![\w.])"),
    "srr_accession": re.compile(r"\b[SED]RR\d{5,}\b"),
    "sra_accession": re.compile(r"\b[SED]RR\d{5,}\b"),
}


def _name(te: Any) -> str:
    tc = getattr(te, "tool_call", None)
    if tc is not None:
        return tc.name
    return te.get("tool") or "" if isinstance(te, dict) else ""


def _args(te: Any) -> dict:
    tc = getattr(te, "tool_call", None)
    if tc is not None:
        return tc.arguments or {}
    return (te.get("arguments") or {}) if isinstance(te, dict) else {}


def _result(te: Any) -> Any:
    return getattr(te, "result", None) if not isinstance(te, dict) else te.get("result")


def _error(te: Any) -> Any:
    return getattr(te, "error", None) if not isinstance(te, dict) else te.get("error")


def build_step_outputs(executions: Iterable[Any]) -> dict[str, Any]:
    """``step_outputs`` from a step's tool executions (``ToolExecution`` or dicts).

    Only successful calls count.  A blocked (plan-mode) call produced
    nothing yet: its outputs come from the replay that runs it.
    """
    out: dict[str, Any] = {}
    ids: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    paths: list[str] = []
    submission_ids: list[str] = []
    output_paths: list[str] = []

    for te in executions or []:
        name, args, result = _name(te), _args(te), _result(te)
        if _error(te) or not isinstance(result, dict) or result.get("error") or result.get("blocked_by_mode"):
            continue
        if name == "search_data":
            docs = result.get("results") or result.get("docs") or []
            field = next(
                (f for f in _ID_FIELDS if any(isinstance(d, dict) and d.get(f) for d in docs)), None
            )
            if field:
                values: list[str] = []
                for d in docs:
                    v = d.get(field) if isinstance(d, dict) else None
                    if v and str(v) not in values:
                        values.append(str(v))
                total = result.get("numFound")
                ids.append({
                    "collection": args.get("collection"),
                    "field": field,
                    "values": values[:MAX_IDS],
                    "total": int(total) if isinstance(total, (int, float)) else len(values),
                    "query": args.get("query"),
                })
        elif name == "create_group" and result.get("path"):
            selection = (
                {"ids_count": len(args.get("ids") or [])}
                if args.get("ids") else {"query": args.get("query"), "collection": args.get("collection")}
            )
            groups.append({
                "name": result.get("name"),
                "path": result["path"],
                "type": result.get("group_type") or args.get("group_type"),
                "count": result.get("count"),
                "selection": selection,
            })
        elif name == "submit_gowe_job" and result.get("submission_id"):
            submission_ids.append(str(result["submission_id"]))
            if result.get("output_path"):
                output_paths.append(str(result["output_path"]))
        elif name in ("workspace_browse", "get_file_metadata", "read_file_preview", "summarize_file", "search_file"):
            path = args.get("path")
            if isinstance(path, str) and path.startswith("/") and path not in paths:
                paths.append(path)

    if ids:
        out["ids"] = ids
    if groups:
        out["groups"] = groups
    if paths:
        out["paths"] = paths[:MAX_PATHS]
    if submission_ids:
        out["submission_ids"] = submission_ids
    if output_paths:
        out["output_paths"] = output_paths
    return out


def extract_values(answer: str, expects: Iterable[str] | None) -> dict[str, str]:
    """The first match in *answer* for each expected name with a known shape."""
    values: dict[str, str] = {}
    for name in expects or []:
        pattern = VALUE_PATTERNS.get(str(name).strip().lower())
        if pattern is None:
            continue
        m = pattern.search(answer or "")
        if m:
            values[str(name)] = m.group(0)
    return values


def merge_step_outputs(*parts: dict[str, Any] | None) -> dict[str, Any]:
    """Merge several step_outputs dicts (lists concatenated, values updated)."""
    out: dict[str, Any] = {}
    for part in parts:
        for key, val in (part or {}).items():
            if isinstance(val, list):
                existing = out.setdefault(key, [])
                for item in val:
                    if item not in existing:
                        existing.append(item)
            elif isinstance(val, dict):
                out.setdefault(key, {}).update(val)
    return out


def format_step_outputs(by_step: dict[str, dict[str, Any]]) -> str:
    """``=== PRIOR STEP OUTPUTS ===`` for a step's prompt, or ''.

    *by_step* maps a step label ("step_1: Find the PLFam") to its outputs.
    Ids are listed in full (they are capped at 200 when stored); every value
    here is copied from a tool result, so it can be used as given.
    """
    lines: list[str] = []
    for label, outputs in by_step.items():
        if not outputs:
            continue
        body: list[str] = []
        for v_name, v in (outputs.get("values") or {}).items():
            body.append(f"- {v_name}: {v}")
        for g in outputs.get("groups") or []:
            sel = g.get("selection") or {}
            how = f" (query: {sel['query']})" if sel.get("query") else (
                f" ({sel['ids_count']} listed ids)" if sel.get("ids_count") else "")
            body.append(f"- {g.get('type') or 'group'} {g.get('name') or ''}: path {g.get('path')}, "
                        f"{g.get('count')} members{how}")
        for i in outputs.get("ids") or []:
            vals = i.get("values") or []
            more = f" (showing {len(vals)} of {i.get('total')})" if i.get("total") and i["total"] > len(vals) else ""
            q = f", query {i['query']}" if i.get("query") else ""
            body.append(f"- {i.get('field')} from {i.get('collection')}{q}{more}: {', '.join(vals)}")
        if outputs.get("submission_ids"):
            body.append(f"- submitted jobs: {', '.join(outputs['submission_ids'])}")
        for p in outputs.get("output_paths") or []:
            body.append(f"- job output folder: {p}")
        for p in outputs.get("paths") or []:
            body.append(f"- workspace path used: {p}")
        if body:
            lines.append(f"{label}:")
            lines.extend(body)
    if not lines:
        return ""
    return (
        "=== PRIOR STEP OUTPUTS ===\n"
        "Exact results of earlier steps this one depends on, copied from their tool "
        "results. Use these values as given (paths, ids, group names); do not look "
        "them up again or retype them from memory. Where the request above names "
        "something differently (a planned group name, say), these values win: they "
        "are what the earlier step actually produced.\n" + "\n".join(lines)
    )
