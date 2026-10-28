"""
Shared tool infrastructure for BV-BRC agents.

Provides:
  - ``execute_tool``: the harness's tool dispatcher -- agent allowlist and
    handoffs, the group guard, the plan/execute gate and its questions --
    which then runs the tool through ``bvbrc_mcp.dispatch``.
  - ``truncate_result`` (from ``bvbrc_mcp.results``) and the per-tool result
    caps the agent loops use.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from typing import Any, Dict, Optional
import logging

from bvbrc_mcp.registry import dispatch
from bvbrc_mcp.results import (  # noqa: F401  (re-exported)
    is_already_submitted_result,
    is_invalid_arguments_result,
    is_needs_user_input_result,
)


# Tools that legitimately run longer than the agent's default tool timeout
# (defined with the tools, in bvbrc_mcp).
from bvbrc_mcp.registry import TOOL_TIMEOUT_OVERRIDES  # noqa: E402

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
    "list_groups":       12_000,   # up to 300 group names (bvbrc_mcp groups.MAX_LISTED_GROUPS)
    "list_jobs":         24_000,   # 20 compact jobs (paths + data inputs) ~16-21 KB
    "rerun_jobs":        24_000,   # up to 10 prepared jobs with their full inputs
    "prepare_batch":     24_000,   # same
    "list_samples":      24_000,   # up to ~100 samples with full read paths
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
logger = logging.getLogger(__name__)

EXECUTE_ONLY_TOOLS: frozenset[str] = frozenset({"submit_gowe_job", "create_group", "edit_group"})

# Gated tools that ask even in Execute mode: they change something that
# already exists (a group's members), so the user sees the exact change
# before it runs.  Only a replay of the approved question runs them.
ALWAYS_ASK_TOOLS: frozenset[str] = frozenset({"edit_group"})

# True while service_agent.phases.replay runs a set the user approved.
REPLAY_IN_PROGRESS: ContextVar[bool] = ContextVar("replay_in_progress", default=False)

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
    "edit_group": frozenset({"service", "data"}),
    # Prepares jobs (never submits), but the set it prepares is replayed by
    # the service agent, which owns the confirmation question and watches.
    "rerun_jobs": frozenset({"service"}),
    # D4 Phase 4: same reason; list_samples (read-only) stays with everyone.
    "prepare_batch": frozenset({"service"}),
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
    if tool_name == "edit_group":
        parts = []
        if args.get("remove_ids"):
            parts.append(f"remove {len(args['remove_ids'])}")
        if args.get("add_ids"):
            parts.append(f"add {len(args['add_ids'])}")
        return f"{args.get('group_type') or 'group'} '{args.get('group') or 'unnamed'}': {' and '.join(parts) or 'edit'}"
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
    from bvbrc_mcp.tools.gowe import place_output_path

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


async def _check_job_roles(arguments: Dict[str, Any], config: Any) -> tuple[Optional[str], list]:
    """``(error, lines)`` for a prepared job's role inputs (bvbrc_mcp.tools.roles).

    At the gate, so a reference or condition that cannot be what the user
    meant is refused before the question, not after the Yes.
    """
    from bvbrc_mcp.tools.gowe import workflow_name_of
    from bvbrc_mcp.tools.roles import check_roles

    args = arguments if isinstance(arguments, dict) else {}
    name = await workflow_name_of(args.get("workflow_id"), config)
    return check_roles(name, args.get("inputs"))


def _add_role_summary(result: Dict[str, Any], lines: list) -> None:
    """Name the reference / conditions in the question and tell the model to."""
    from bvbrc_mcp.tools.roles import with_roles

    if not lines:
        return
    result["roles"] = lines
    result["summary"] = with_roles(result["summary"], lines)
    result["error"] += " In the summary, state: " + "; ".join(lines) + "."


def prepared_actions_of(result: Any) -> list[Dict[str, Any]]:
    """Jobs a tool prepared without running them (D4), as blocked actions.

    ``rerun_jobs`` and ``prepare_batch`` return ``prepared_actions`` in the same ``{tool, summary,
    arguments}`` shape as a plan-mode refusal, so they reach the same
    confirmation question and the same verbatim replay.  ``origin``,
    ``original``, ``dropped_inputs`` and ``samples`` are kept (the gateway matches a replay on tool + arguments
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
        for key in ("origin", "original", "dropped_inputs", "samples"):
            if key in action:
                entry[key] = action[key]
        actions.append(entry)
    return actions


# Collections with one row per genome (or per feature for a feature group):
# their row count is the member count.  Others (genome_amr, sp_gene, ...)
# have several rows per member, so the count is an upper bound.
_ONE_ROW_PER_MEMBER = {"genome_group": {"genome"}, "feature_group": {"genome_feature"}}


async def _plan_group_edit_for_question(
    arguments: Dict[str, Any], config: Any, headers: Any
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """``(error, summary)`` for a gated edit_group call.

    The edit is planned (one Workspace.get) so the question reads
    "genome_group 'X': remove 1 (50 → 49): 1313.42242"; an edit that cannot
    be what the user meant is returned as the tool error instead of a
    question.  A read failure leaves the plain summary.
    """
    from bvbrc_mcp.tools.groups import edit_summary, plan_group_edit

    try:
        plan = await asyncio.wait_for(plan_group_edit(_safe_args(arguments), config, headers), timeout=20)
    except Exception as exc:
        logger.warning("edit_group preview failed: %s", exc)
        return None, None
    if "error" in plan:
        return {**plan, "tool": "edit_group", "arguments": _safe_args(arguments)}, None
    return None, edit_summary(plan)


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
    from bvbrc_mcp.tools.groups import count_group_members, selection_summary

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
        # Listed ids alone: exact.  A query on a one-row-per-member
        # collection: exact.  Both together: listed ids may also match the
        # query, so the count is an upper bound.
        if args.get("ids"):
            exact = not args.get("query")
        else:
            exact = args.get("collection") in _ONE_ROW_PER_MEMBER.get(group_type, set())
        parts.append(f"— {'' if exact else 'up to '}{n} {unit}")
        result["member_count"] = n
        result["member_count_exact"] = exact
    selection = selection_summary(args)
    if selection:
        parts.append(f"({selection})")
    result["summary"] = " ".join(parts)


def wrong_agent_result(tool_name: str, agent_key: str) -> Dict[str, Any]:
    """Refusal for a gated tool called by an agent that may not use it."""
    allowed = " or ".join(sorted(TOOL_AGENT_ALLOWLIST.get(tool_name, ())))
    if tool_name == "submit_gowe_job":
        next_step = (
            "Tell the user which workflow and inputs you would use and that "
            "they can ask for it to be run as a job (for example \"run "
            "<workflow> on <input>\")."
        )
    elif tool_name == "prepare_batch":
        next_step = (
            "Tell the user they can ask for the batch directly (for example "
            "\"run <workflow> on every sample in <folder>\")."
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
    "edit_group": "data",
    "create_plan": "planning",
    "rerun_jobs": "service",
    "prepare_batch": "service",
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

    # One group per plan step, and no group identical to one already made
    # (bvbrc_mcp.tools.groups.group_guard).  Before the plan-mode gate, so a
    # prepared group is refused too, not only a created one.
    if tool_name == "create_group":
        from bvbrc_mcp.tools.groups import group_guard

        refusal = await group_guard(_safe_args(arguments), config, headers)
        if refusal:
            return refusal

    # Execution-mode gate: side-effecting tools only run in EXECUTE mode.
    # A missing config (or a config without the attribute) is treated as
    # PLAN — the safe default.  ALWAYS_ASK_TOOLS are refused in Execute mode
    # too, except inside a replay of the question the user answered Yes to.
    gated = tool_name in EXECUTE_ONLY_TOOLS and (
        execution_mode_of(config) != "execute"
        or (tool_name in ALWAYS_ASK_TOOLS and not REPLAY_IN_PROGRESS.get())
    )
    if gated:
        if tool_name == "edit_group":
            # Read the group now: an impossible edit is refused here, and
            # the question shows the real before/after counts.
            edit_error, edit_summary_text = await _plan_group_edit_for_question(arguments, config, headers)
            if edit_error:
                return edit_error
        role_lines: list = []
        if tool_name == "submit_gowe_job":
            role_error, role_lines = await _check_job_roles(arguments, config)
            if role_error:
                return {"error": role_error, "tool": tool_name, "arguments": _safe_args(arguments)}
        result = blocked_by_mode_result(tool_name, arguments)
        if tool_name == "submit_gowe_job":
            _add_role_summary(result, role_lines)
            _add_output_location(result, arguments, config)
        elif tool_name == "create_group":
            await _add_group_preview(result, arguments, config, headers)
        elif tool_name == "edit_group" and edit_summary_text:
            result["summary"] = edit_summary_text
        return result

    # Inject config and headers if the tool accepts them.  In place, as
    # callers have always seen it; dispatch below adds nothing more.
    if inject_config and config is not None and "config" not in arguments:
        arguments["config"] = config
    if inject_headers and headers is not None and "headers" not in arguments:
        arguments["headers"] = headers

    # Running the tool -- the per-tool timeout (larger of caller vs override)
    # and turning exceptions into error dicts -- is bvbrc_mcp.dispatch.
    return await dispatch(
        tool_name,
        arguments,
        config,
        timeout_seconds=timeout_seconds,
        dispatch_table=dispatch_table,
        inject_config=False,
        inject_headers=False,
    )


# ---------------------------------------------------------------------------
# Smart truncation with representative sampling: bvbrc_mcp.results
# ---------------------------------------------------------------------------

from bvbrc_mcp.results import truncate_result  # noqa: E402,F401


def _safe_args(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Strip config/headers from arguments for safe error reporting."""
    return {k: v for k, v in arguments.items() if k not in ("config", "headers")}
