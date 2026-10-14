"""
Shared tool infrastructure for BV-BRC agents.

Provides:
  - ``execute_tool``: Generic tool dispatcher with timeout and error handling.
  - ``truncate_result``: Smart result-to-JSON serializer with representative
    sampling across types, aggregate summaries, and binary-search sizing.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import traceback
from collections import Counter, defaultdict
from typing import Any, Dict, Optional


# Tools that legitimately run longer than the agent's default tool timeout.
TOOL_TIMEOUT_OVERRIDES: dict[str, float] = {
    "find_similar_genomes": 120.0,
    "search_file": 90.0,  # streams up to 25 MB from the workspace
    "summarize_file": 90.0,  # same scan
    "rerun_jobs": 90.0,  # recursive listing + query_tasks + workflow catalog
}

# Per-tool result character limits.  Tools that return bulk text (file
# reads, search results, summaries) need higher caps than the default
# 8000 chars — otherwise ``truncate_result`` silently cuts the data
# and re-creates the original "only see 7 KB" bug at a larger number.
TOOL_RESULT_CHAR_LIMITS: dict[str, int] = {
    "read_file_preview": 48_000,   # 32 KB data + JSON escaping + envelope
    "search_file":       16_000,   # self-bounded to ~14 000 by file_search.py
    "summarize_file":    12_000,   # self-bounded to ~11 000 by file_summary.py
    "search_literature": 24_000,   # self-bounded to ~20 000 by literature.py
    "workspace_browse":  16_000,   # ~45 compact items; job_result entries carry a job block
    "rerun_jobs":        24_000,   # up to 10 prepared jobs with their full inputs
}


def result_char_limit(tool_name: str, default: int = 8000) -> int:
    """Return the result char limit for *tool_name*.

    Returns the larger of *default* and the per-tool override from
    ``TOOL_RESULT_CHAR_LIMITS``, so a caller's explicit cap is never
    reduced by this function.
    """
    return max(default, TOOL_RESULT_CHAR_LIMITS.get(tool_name, 0))


# ---------------------------------------------------------------------------
# Execution mode gate
# ---------------------------------------------------------------------------

# Tools that create durable side effects (a GoWe submission, a workspace
# group).  They are only executed when the chat session is in EXECUTE
# mode.  In PLAN mode (the default) ``execute_tool`` refuses them and
# returns a structured ``blocked_by_mode`` result to the LLM instead.
# This is the single enforcement point — every agent loop dispatches
# through ``execute_tool``, so the gate holds regardless of which agent
# the router picked or what the LLM decided.
EXECUTE_ONLY_TOOLS: frozenset[str] = frozenset({"submit_gowe_job", "create_group"})

EXECUTION_MODES: frozenset[str] = frozenset({"plan", "execute"})

# Which agents may call each side-effecting tool.  Every agent has every tool
# schema, and on 2026-09-18 / 10-09 the data agent submitted jobs in answer
# to plain questions (parrello a05b7324, e88f5e12) and the helpdesk agent
# resubmitted a job when a user complained (Chris dae698e5).  Jobs from a
# non-service agent also carried no submission_ids, so no workflow watch and
# no completion message.  Groups stay with data too: "make a group of these"
# is a data request.
TOOL_AGENT_ALLOWLIST: Dict[str, frozenset[str]] = {
    "submit_gowe_job": frozenset({"service"}),
    "create_group": frozenset({"service", "data"}),
    # Prepares jobs (never submits), but the set it prepares is replayed by
    # the service agent, which owns the confirmation question and watches.
    "rerun_jobs": frozenset({"service"}),
}

PLAN_MODE_BLOCKED_MESSAGE = (
    "This chat session is in PLAN mode, so {tool} was NOT executed and "
    "nothing was submitted or created. Do not call it again in this turn "
    "and do not change any of the inputs. Instead, present exactly what you "
    "prepared (workflow or group, every input value, output folder name) as "
    "a 'Ready to submit' summary. The user is asked one Yes / No / Other "
    "question right under your reply, and Yes runs exactly what you "
    "prepared: end with 'Answer **Yes** below to run it, **No** to skip it, "
    "or tell me what to change.' Do not tell them to switch modes. Never say "
    "the job was submitted or the group was created."
)


def normalize_execution_mode(value: Any) -> str:
    """Coerce any value to ``"plan"`` or ``"execute"`` (default ``"plan"``)."""
    return "execute" if value == "execute" else "plan"


def execution_mode_of(config: Any) -> str:
    """Return the execution mode carried by *config* (``"plan"`` if absent)."""
    return normalize_execution_mode(getattr(config, "execution_mode", None))


def describe_blocked_action(tool_name: str, arguments: Dict[str, Any]) -> str:
    """One-line, user-facing description of a gated call that was refused."""
    args = _safe_args(arguments)
    if tool_name == "submit_gowe_job":
        workflow = (
            args.get("workflow_name")
            or args.get("workflow_id")
            or "workflow"
        )
        inputs = args.get("inputs") if isinstance(args.get("inputs"), dict) else {}
        output_path = args.get("output_path") or inputs.get("output_path") or ""
        folder = str(output_path).rstrip("/").rsplit("/", 1)[-1] if output_path else ""
        return f"{workflow} → {folder}" if folder else str(workflow)
    if tool_name == "create_group":
        group_type = args.get("group_type") or "group"
        name = args.get("group_name") or "unnamed"
        limit = args.get("limit")
        suffix = f" (up to {limit} ids)" if limit else ""
        return f"{group_type} '{name}'{suffix}"
    return tool_name


def blocked_by_mode_result(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Build the tool result returned in place of a gated call."""
    return {
        "error": PLAN_MODE_BLOCKED_MESSAGE.format(tool=tool_name),
        "blocked_by_mode": True,
        "execution_mode": "plan",
        "tool": tool_name,
        "arguments": _safe_args(arguments),
        "summary": describe_blocked_action(tool_name, arguments),
    }


def _add_output_location(result: Dict[str, Any], arguments: Dict[str, Any], config: Any) -> None:
    """Tell a plan-mode summary where the job's results will really go.

    The folder is placed at submission (place_output_path), so a "Ready to
    submit" summary only knew the bare name: "your Testing folder" (Oct 13).
    The card's arguments are left unchanged; the replayed submission places
    the folder the same way.
    """
    session_id = getattr(config, "session_id", None)
    workspace_path = getattr(config, "workspace_path", None)
    inputs = arguments.get("inputs") if isinstance(arguments, dict) else None
    if not (session_id and workspace_path and isinstance(inputs, dict)):
        return
    from shared.tools.gowe import place_output_path

    final, note = place_output_path(
        inputs.get("output_path"),
        f"{workspace_path}/chats/{session_id}",
        session_id,
        str(arguments.get("workflow_id") or "Job"),
    )
    result["output_location"] = final
    tell = f"In the summary, give the output folder as `{final}`."
    if note:
        result["output_location_note"] = note
        tell += f" Also tell the user: {note}"
    result["error"] += " " + tell


def prepared_actions_of(result: Any) -> list[Dict[str, Any]]:
    """Jobs a tool prepared without running them (D4), as blocked actions.

    ``rerun_jobs`` returns ``prepared_actions`` in the same ``{tool, summary,
    arguments}`` shape as a plan-mode refusal, so they reach the same
    confirmation question and the same verbatim replay.  ``origin``,
    ``original`` and ``dropped_inputs`` are kept (the gateway matches a replay on tool + arguments
    only, so extra keys do not change what may run).
    """
    if not isinstance(result, dict) or not isinstance(result.get("prepared_actions"), list):
        return []
    actions = []
    for action in result["prepared_actions"]:
        if not isinstance(action, dict) or action.get("tool") not in EXECUTE_ONLY_TOOLS:
            continue
        if not isinstance(action.get("arguments"), dict):
            continue
        entry = {
            "tool": action["tool"],
            "summary": action.get("summary") or action["tool"],
            "arguments": action["arguments"],
        }
        for key in ("origin", "original", "dropped_inputs"):
            if key in action:
                entry[key] = action[key]
        actions.append(entry)
    return actions


# Collections with one row per genome (or per feature for a feature group):
# their row count is the member count.  Others (genome_amr, sp_gene, ...)
# have several rows per member, so the count is an upper bound.
_ONE_ROW_PER_MEMBER = {"genome_group": {"genome"}, "feature_group": {"genome_feature"}}


async def _add_group_preview(
    result: Dict[str, Any], arguments: Dict[str, Any], config: Any, headers: Any
) -> None:
    """Put the member count and the selection on a plan-mode group question.

    D4 Phase 3: the user approves the actual selection ("genome_group
    'strep_tet_R' — 200 genomes (genome_amr: antibiotic:tetracycline AND
    ...)"), not just a name.  One count-only Solr request; on failure the
    summary keeps the name only.  The arguments are not changed, so the
    replay runs exactly what was shown.
    """
    from shared.tools.groups import count_group_members, selection_summary

    args = arguments if isinstance(arguments, dict) else {}
    group_type = args.get("group_type") or "group"
    name = args.get("group_name") or "unnamed"
    try:
        n = await asyncio.wait_for(count_group_members(args, config, headers), timeout=15)
    except Exception:
        n = None
    unit = "genomes" if group_type == "genome_group" else "features" if group_type == "feature_group" else "members"
    parts = [f"{group_type} '{name}'"]
    if n is not None:
        exact = bool(args.get("ids")) or args.get("collection") in _ONE_ROW_PER_MEMBER.get(group_type, set())
        parts.append(f"— {'' if exact else 'up to '}{n} {unit}")
        result["member_count"] = n
        result["member_count_exact"] = exact
    selection = selection_summary(args)
    if selection:
        parts.append(f"({selection})")
    result["summary"] = " ".join(parts)


def is_invalid_arguments_result(result: Any) -> bool:
    """True when the call was refused because its arguments were not JSON.

    Like a plan-mode block, this is NOT a tool or submission failure: the tool
    never ran and nothing changed. Loops that count failures must skip it, or a
    pair of malformed emissions aborts a turn that had nothing wrong with it.
    """
    return isinstance(result, dict) and bool(result.get("invalid_arguments"))


def wrong_agent_result(tool_name: str, agent_key: str) -> Dict[str, Any]:
    """Refusal for a gated tool called by an agent that may not use it."""
    allowed = " or ".join(sorted(TOOL_AGENT_ALLOWLIST.get(tool_name, ())))
    if tool_name == "submit_gowe_job":
        next_step = (
            "Tell the user which workflow and inputs you would use and that "
            "they can ask for it to be run as a job (for example \"run "
            "<workflow> on <input>\")."
        )
    elif tool_name == "rerun_jobs":
        next_step = (
            "Tell the user they can ask for the rerun directly (for example "
            "\"rerun the jobs in <folder>\")."
        )
    else:
        next_step = (
            "Tell the user what you would create and that they can ask for it "
            "directly (for example \"make a genome group of these\")."
        )
    return {
        "error": (
            f"{tool_name} is not available to the {agent_key} agent (only the "
            f"{allowed} agent can call it). Nothing was run. {next_step} Do "
            f"not call {tool_name} again."
        ),
        "wrong_agent": True,
        "tool": tool_name,
        "agent": agent_key,
    }


# F1 agent handoff: a call that belongs to another agent hands the turn to
# it instead of dead-ending (Oct 13 case 9: the data agent prepared BLAST,
# got the allowlist refusal, and could only tell the user to "switch to
# Execute mode", which never works for it).  One hop per turn: an agent that
# received a handoff (config.handoff_from) gets the plain refusal instead.
HANDOFF_TARGETS: Dict[str, str] = {
    "submit_gowe_job": "service",
    "create_group": "data",
    "create_plan": "planning",
    "rerun_jobs": "service",
}


def handoff_target(tool_name: str, config: Any) -> Optional[str]:
    """The agent this call should be handed to, or None to run it here."""
    target = HANDOFF_TARGETS.get(tool_name)
    agent_key = getattr(config, "agent_key", None)
    if not target or not agent_key or getattr(config, "handoff_from", None):
        return None
    if tool_name == "create_plan":
        return target if agent_key != "planning" else None
    allowed = TOOL_AGENT_ALLOWLIST.get(tool_name)
    return target if allowed is not None and agent_key not in allowed else None


def handoff_result(tool_name: str, arguments: Dict[str, Any], target: str) -> Dict[str, Any]:
    """Tool result for a call handed to *target*.  Not an error: nothing ran."""
    return {
        "handoff_to": target,
        "tool": tool_name,
        "arguments": _safe_args(arguments),
        "message": (
            f"Passed to the {target} agent, which will finish this request. "
            "Stop now: do not call more tools and do not write a reply."
        ),
    }


def is_handoff_result(result: Any) -> bool:
    """True when *result* hands the turn to another agent (F1)."""
    return isinstance(result, dict) and bool(result.get("handoff_to"))


def is_wrong_agent_result(result: Any) -> bool:
    """True when *result* is the agent-allowlist refusal."""
    return isinstance(result, dict) and bool(result.get("wrong_agent"))


def is_blocked_result(result: Any) -> bool:
    """True when *result* is the refusal produced by the execution-mode gate."""
    return isinstance(result, dict) and bool(result.get("blocked_by_mode"))


def is_needs_user_input_result(result: Any) -> bool:
    """True when submit_gowe_job stopped before submitting to ask the user something."""
    return isinstance(result, dict) and bool(result.get("needs_user_input"))


def is_already_submitted_result(result: Any) -> bool:
    """True when submit_gowe_job refused a repeat of a recent submission."""
    return isinstance(result, dict) and bool(result.get("already_submitted"))


async def execute_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    dispatch_table: Dict[str, Any],
    timeout_seconds: float = 30.0,
    config: Any = None,
    headers: Dict[str, str] | None = None,
    inject_config: bool = True,
    inject_headers: bool = True,
) -> Dict[str, Any]:
    """Execute a tool by name from the given dispatch table.

    Handles:
      - Looking up the tool function in *dispatch_table*
      - Optionally injecting *config* and *headers* into the call arguments
      - Timeout enforcement
      - Error wrapping (tool errors become structured dicts, not exceptions)

    Args:
        tool_name: Name of the tool to execute.
        arguments: Arguments from the LLM's ``tool_call``.
        dispatch_table: Mapping of ``{tool_name: async_callable}``.
        timeout_seconds: Maximum execution time.
        config: Agent config to inject (any Pydantic model).
        headers: HTTP headers to inject (e.g. auth).
        inject_config: Whether to inject ``config`` into arguments.
        inject_headers: Whether to inject ``headers`` into arguments.

    Returns:
        Dict with the tool's result, or an error dict if execution failed.
    """
    func = dispatch_table.get(tool_name)
    if func is None:
        return {
            "error": f"Unknown tool: '{tool_name}'",
            "available_tools": sorted(dispatch_table.keys()),
        }

    # Arguments that could not be parsed as JSON never reach the tool. Passing
    # them through produced a Python signature error -- "missing 2 required
    # positional arguments" -- which named the wrong problem, told the model
    # nothing about how to fix it, and counted against the service agent's
    # two-failure circuit breaker. Seen on 3 of 5 submissions on 2026-10-08.
    if "_unparseable_arguments" in arguments:
        raw = arguments.get("_unparseable_arguments")
        return {
            "error": (
                f"The arguments for '{tool_name}' were not valid JSON, so the "
                f"tool was not called. Nothing was submitted or changed. Send "
                f"the arguments again as a single JSON object."
            ),
            "invalid_arguments": True,
            "tool": tool_name,
            "received": (raw[:400] if isinstance(raw, str) else str(raw)[:400]),
        }

    # Agent gate, BEFORE the plan-mode gate: a data agent in plan mode would
    # otherwise produce a "Submit this job" card, and its replay would run
    # the job anyway.  A config without agent_key is not restricted.
    target = handoff_target(tool_name, config)
    if target:
        return handoff_result(tool_name, arguments, target)

    agent_key = getattr(config, "agent_key", None)
    allowed_agents = TOOL_AGENT_ALLOWLIST.get(tool_name)
    if agent_key and allowed_agents is not None and agent_key not in allowed_agents:
        return wrong_agent_result(tool_name, agent_key)

    # Execution-mode gate: side-effecting tools only run in EXECUTE mode.
    # A missing config (or a config without the attribute) is treated as
    # PLAN — the safe default.
    if tool_name in EXECUTE_ONLY_TOOLS and execution_mode_of(config) != "execute":
        result = blocked_by_mode_result(tool_name, arguments)
        if tool_name == "submit_gowe_job":
            _add_output_location(result, arguments, config)
        elif tool_name == "create_group":
            await _add_group_preview(result, arguments, config, headers)
        return result

    # Apply per-tool timeout overrides (use the larger of caller vs override)
    timeout_seconds = max(timeout_seconds, TOOL_TIMEOUT_OVERRIDES.get(tool_name, 0.0))

    # Inject config and headers if the tool accepts them
    if inject_config and config is not None and "config" not in arguments:
        arguments["config"] = config
    if inject_headers and headers is not None and "headers" not in arguments:
        arguments["headers"] = headers

    try:
        result = await asyncio.wait_for(
            func(**arguments),
            timeout=timeout_seconds,
        )
        return result

    except asyncio.TimeoutError:
        return {
            "error": f"Tool '{tool_name}' timed out after {timeout_seconds}s",
            "tool": tool_name,
            "arguments": _safe_args(arguments),
        }

    except TypeError as e:
        return {
            "error": f"Invalid arguments for tool '{tool_name}': {str(e)}",
            "tool": tool_name,
            "arguments": _safe_args(arguments),
        }

    except Exception as e:
        return {
            "error": f"Tool '{tool_name}' failed: {type(e).__name__}: {str(e)}",
            "tool": tool_name,
            "arguments": _safe_args(arguments),
            "traceback": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# Smart truncation with representative sampling
# ---------------------------------------------------------------------------


def _build_items_summary(items: list) -> Dict[str, Any]:
    """Build an aggregate summary of items for the LLM.

    Returns counts by type, folder distribution, size stats, date range, etc.
    so the LLM can answer aggregation questions even when items are truncated.
    """
    type_counts: Counter = Counter()
    folder_counts: Counter = Counter()
    total_size = 0
    dates: list[str] = []

    for item in items:
        if isinstance(item, dict):
            item_type = item.get("type", "unknown")
            item_path = item.get("path", "")
            item_size = item.get("size", 0)
            item_date = item.get("creation_time", "")
        elif isinstance(item, list) and len(item) >= 4:
            item_type = item[1] if len(item) > 1 else "unknown"
            item_path = item[2] if len(item) > 2 else ""
            item_size = item[6] if len(item) > 6 else 0
            item_date = item[3] if len(item) > 3 else ""
        else:
            continue

        type_counts[item_type] += 1

        parent = os.path.dirname(item_path.rstrip("/")) if item_path else ""
        if parent:
            folder_counts[parent] += 1

        if isinstance(item_size, (int, float)):
            total_size += item_size
        if item_date:
            dates.append(str(item_date))

    summary: Dict[str, Any] = {
        "total_items": len(items),
        "counts_by_type": dict(type_counts.most_common()),
        "unique_types": sorted(type_counts.keys()),
    }

    if folder_counts:
        summary["top_folders"] = dict(folder_counts.most_common(10))
    if total_size > 0:
        summary["total_size_bytes"] = total_size
    if dates:
        sorted_dates = sorted(dates)
        summary["date_range"] = {
            "earliest": sorted_dates[0],
            "latest": sorted_dates[-1],
        }

    return summary


def _slim_item(item: Any) -> Any:
    """Strip verbose metadata fields from a workspace item dict."""
    if not isinstance(item, dict):
        return item
    drop_keys = {
        "userMeta",
        "autoMeta",
        "link_reference",
        "user_permissions",
        "global_permission",
    }
    return {k: v for k, v in item.items() if k not in drop_keys}


def _sample_representative(items: list, budget: int) -> list:
    """Pick a representative sample of items, ensuring each type is included."""
    if len(items) <= budget:
        return [_slim_item(i) for i in items]

    by_type: dict[str, list] = defaultdict(list)
    for item in items:
        if isinstance(item, dict):
            t = item.get("type", "unknown")
        elif isinstance(item, list) and len(item) > 1:
            t = item[1]
        else:
            t = "unknown"
        by_type[t].append(item)

    sampled: list = []
    remaining_budget = budget

    # Phase 1: one item per type
    for t, group in by_type.items():
        if remaining_budget <= 0:
            break
        sampled.append(_slim_item(group[0]))
        remaining_budget -= 1

    # Phase 2: fill proportionally from each type
    if remaining_budget > 0:
        for t, group in by_type.items():
            share = max(0, math.floor((len(group) / len(items)) * remaining_budget))
            for item in group[1 : 1 + share]:
                sampled.append(_slim_item(item))

    # Fill remaining budget
    if len(sampled) < budget:
        seen = {id(s) for s in sampled}
        for item in items:
            if len(sampled) >= budget:
                break
            if id(item) not in seen:
                sampled.append(_slim_item(item))
                seen.add(id(item))

    return sampled[:budget]


def _strip_ui_grid(obj: Any) -> Any:
    """Recursively remove ``ui_grid`` keys from a result dict."""
    if isinstance(obj, dict):
        return {k: _strip_ui_grid(v) for k, v in obj.items() if k != "ui_grid"}
    if isinstance(obj, list):
        return [_strip_ui_grid(v) for v in obj]
    return obj


def truncate_result(result: Dict[str, Any], max_chars: int = 8000) -> str:
    """Serialize a tool result to JSON, truncating if too large.

    Smart truncation strategy:
      1. Strips ``ui_grid`` payloads (duplicate data for UI rendering).
      2. Catalog lists (``workflows``) are never sampled -- every entry is
         kept and descriptions are shortened instead, so the LLM always
         sees the full set of available workflows.
      3. Adds aggregate summary (type counts, folder distribution, etc.).
      4. Samples items representatively across types (not just first N).
      5. Strips verbose metadata from sampled items.
      6. Uses binary search to find the largest sample that fits.
      7. Falls back to hard truncation if nothing else works.

    Args:
        result: The tool result dict.
        max_chars: Maximum characters for the serialized output.

    Returns:
        JSON string of the result, possibly truncated.
    """
    result = _strip_ui_grid(result)
    serialized = json.dumps(result, indent=2, default=str)

    if len(serialized) <= max_chars:
        return serialized

    # Catalog lists must stay complete: dropping entries hides options from
    # the LLM (e.g. the BLAST workflow silently vanishing from discovery).
    catalog = _fit_catalog(result, "workflows", "description", max_chars)
    if catalog is not None:
        return catalog

    # Facet results: sample the bucket lists but always keep the per-field
    # bucket / distinct counts.  Hard-cutting the JSON mid-bucket left the
    # LLM with no total and thousands of ids to "count" (see the
    # penicillin-resistant S. pneumoniae 5-minute timeout).
    faceted = _fit_facets(result, max_chars)
    if faceted is not None:
        return faceted

    # Find the list-valued key to truncate (check nested "result" envelope too)
    for list_key in ("items", "results", "records", "files", "ids"):
        target = result
        if "result" in result and isinstance(result["result"], dict):
            target = result["result"]

        if list_key in target and isinstance(target[list_key], list):
            all_items = target[list_key]
            num_items = len(all_items)
            summary = _build_items_summary(all_items)

            truncated = dict(result)
            if "result" in truncated and isinstance(truncated["result"], dict):
                truncated["result"] = dict(truncated["result"])
                inner = truncated["result"]
            else:
                inner = truncated

            inner["_summary"] = summary

            # Binary search for the largest representative sample that fits
            lo, hi = 1, num_items
            best_n = 1
            while lo <= hi:
                mid = (lo + hi) // 2
                inner[list_key] = _sample_representative(all_items, mid)
                inner["_truncated"] = {
                    "total": num_items,
                    "shown": mid,
                    "note": (
                        f"Showing {mid} of {num_items} items "
                        f"(representative sample across types)."
                    ),
                }
                candidate = json.dumps(truncated, indent=2, default=str)
                if len(candidate) <= max_chars:
                    best_n = mid
                    serialized = candidate
                    lo = mid + 1
                else:
                    hi = mid - 1

            inner[list_key] = _sample_representative(all_items, best_n)
            inner["_truncated"] = {
                "total": num_items,
                "shown": best_n,
                "note": (
                    f"Showing {best_n} of {num_items} items "
                    f"(representative sample across types)."
                ),
            }
            return json.dumps(truncated, indent=2, default=str)

    # Fallback: hard truncate
    return serialized[:max_chars] + f"\n... [TRUNCATED at {max_chars} chars]"


def _fit_facets(result: Dict[str, Any], max_chars: int) -> Optional[str]:
    """Serialize a ``facet_query`` result within *max_chars*.

    Keeps ``numFound``, ``bucket_counts`` / ``distinct_counts`` and every
    other scalar key intact, then binary-searches the number of buckets
    kept per field.  Returns ``None`` when *result* has no ``facets`` dict.
    """
    facets = result.get("facets")
    if not isinstance(facets, dict) or not facets:
        return None
    lists = {f: v for f, v in facets.items() if isinstance(v, list)}
    if not lists:
        return None

    total = {f: len(v) for f, v in lists.items()}
    max_len = max(total.values())

    def render(n: int) -> str:
        trimmed = dict(result)
        trimmed["facets"] = {
            f: (v[:n] if isinstance(v, list) else v) for f, v in facets.items()
        }
        trimmed.setdefault("bucket_counts", total)
        trimmed["_truncated"] = {
            "buckets_total": total,
            "buckets_shown": {f: min(n, len(v)) for f, v in lists.items()},
            "note": (
                f"Showing the top {n} buckets per field. bucket_counts / "
                "distinct_counts give the full number of distinct values — "
                "do not count the buckets."
            ),
        }
        return json.dumps(trimmed, indent=2, default=str)

    lo, hi, best = 0, max_len, None
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = render(mid)
        if len(candidate) <= max_chars:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    # Even zero buckets did not fit (huge scalar payload) — let the generic
    # fallback handle it.
    return best


def _fit_catalog(
    result: Dict[str, Any],
    list_key: str,
    text_key: str,
    max_chars: int,
) -> Optional[str]:
    """Serialize ``result`` keeping every entry of ``result[list_key]``.

    Tries compact (no-indent) JSON first, then shrinks the ``text_key``
    field of each entry (binary search on its length) until the JSON fits
    ``max_chars``.  Returns ``None`` when ``list_key`` is absent or the
    entries cannot fit even with the text removed.
    """
    entries = result.get(list_key)
    if not isinstance(entries, list) or not entries:
        return None

    longest = max(
        (len(e.get(text_key, "")) for e in entries if isinstance(e, dict)),
        default=0,
    )

    def shrink_entry(e: Any, cap: int) -> Any:
        if not isinstance(e, dict) or not isinstance(e.get(text_key), str):
            return e
        if cap <= 0:
            return {k: v for k, v in e.items() if k != text_key}
        text = e[text_key]
        return {**e, text_key: text[:cap].rstrip() + ("..." if len(text) > cap else "")}

    def render(cap: int, indent: Optional[int] = 2) -> str:
        shrunk = dict(result)
        shrunk[list_key] = [shrink_entry(e, cap) for e in entries]
        if cap < longest:
            shrunk["_truncated"] = {
                "total": len(entries),
                "shown": len(entries),
                "note": (
                    f"All {len(entries)} entries kept; {text_key} fields "
                    + (f"shortened to {cap} chars to fit." if cap > 0
                       else "omitted to fit.")
                ),
            }
        return json.dumps(shrunk, indent=indent, default=str)

    # Indentation is the cheapest thing to give up -- keep full text if
    # compact JSON fits.
    compact = render(longest, indent=None)
    if len(compact) <= max_chars:
        return compact

    # Binary search for the longest text cap that fits (compact JSON).
    lo, hi = 0, longest
    best: Optional[str] = None
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = render(mid, indent=None)
        if len(candidate) <= max_chars:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _safe_args(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Strip config/headers from arguments for safe error reporting."""
    return {k: v for k, v in arguments.items() if k not in ("config", "headers")}
