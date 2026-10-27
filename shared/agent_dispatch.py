"""Single entry point for running BV-BRC Copilot agents.

Used by:
  - InProcessAgentHandle (orchestrator in-process path)
  - agent_chat MCP tool wrapper (optional, for debugging)

Agents are imported lazily so the orchestrator can start even if an
individual agent has unmet dependencies.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# ---------------------------------------------------------------------------
# sys.path setup -- make agent packages, shared/ and config/
# importable from the orchestrator venv (or any other process).
#
# Layout relative to this file:
#   bvbrc-agents/
#     ├── shared/agent_dispatch.py  <-- HERE
#     ├── agents/data_agent/
#     ├── agents/service_agent/
#     ├── agents/workspace_agent/
#     ├── agents/helpdesk_agent/
#     ├── agents/analysis_agent/
#     ├── agents/planning_agent/
#     └── config/llm_config.py
#
# The BV-BRC tools (and the Solr client) come from the installed bvbrc-mcp
# package, not from a path.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent

for _subdir in ("agents", "config"):
    _path = str(_REPO_ROOT / _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

_root_path = str(_REPO_ROOT)
if _root_path not in sys.path:
    sys.path.insert(0, _root_path)

from shared.agent_utils import (  # noqa: E402
    CLARIFICATION_INTRO,
    format_clarification_questions,
    parse_clarification_questions,
)
from shared.answer_filter import collect_known_links, filter_answer  # noqa: E402
from shared.context_budget import resolve_context_window  # noqa: E402
from shared.tools import prepared_actions_of  # noqa: E402
from shared.step_outputs import build_step_outputs  # noqa: E402


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _normalize_bvbrc_token(token: Optional[str]) -> Optional[str]:
    """Return a raw PATRIC token, stripping a leading ``Bearer `` if present.

    Downstream BV-BRC services (workspace, literature RAG on :12006, etc.)
    expect ``un=...|tokenid=...``, not ``Bearer <token>``.
    """
    if not token:
        return None
    value = token.strip()
    if value.lower().startswith("bearer "):
        value = value[7:].strip()
    return value or None


def _resolve_auth_token(token: Optional[str]) -> Optional[str]:
    """Normalize a BV-BRC auth token string.

    Callers that have a TokenProvider (MCP wrapper) should resolve the
    token themselves and pass the raw string here.
    """
    return _normalize_bvbrc_token(token)


def _parse_context(context: Any) -> dict[str, Any]:
    """Parse context into a dict.

    Accepts a dict (in-process), a JSON string (MCP wire format), or None.
    """
    if not context:
        return {}
    if isinstance(context, dict):
        return dict(context)
    if isinstance(context, str):
        try:
            parsed = json.loads(context)
            return parsed if isinstance(parsed, dict) else {"raw_context": context}
        except (json.JSONDecodeError, TypeError):
            return {"raw_context": context}
    return {"raw_context": str(context)}


def _build_config_kwargs(
    auth_token: Optional[str],
    ctx: dict[str, Any],
) -> dict[str, Any]:
    """Build config kwargs from auth token and LLM override in context."""
    config_kwargs: dict[str, Any] = {"bvbrc_auth_token": auth_token}
    # Extract llm_override without mutating the caller's dict
    llm_override = ctx.get("llm_override")
    if llm_override and isinstance(llm_override, dict):
        if llm_override.get("base_url"):
            config_kwargs["llm_base_url"] = llm_override["base_url"]
        if llm_override.get("api_key"):
            config_kwargs["llm_api_key"] = llm_override["api_key"]
        if llm_override.get("model"):
            config_kwargs["llm_model"] = llm_override["model"]
        if llm_override.get("max_tokens"):
            config_kwargs["max_tokens"] = llm_override["max_tokens"]

    # LLM_ADMISSION_URL override: when the admission proxy is running on
    # holly, rewrite llm_base_url so every agent's create_client() routes
    # through the proxy instead of directly to the upstream vLLM. The model,
    # api_key, and max_tokens still come from the per-request llm_override.
    # Unset the env var to bypass the proxy.
    admission_url = os.environ.get("LLM_ADMISSION_URL", "").strip()
    if admission_url:
        config_kwargs["llm_base_url"] = admission_url

    # Execution mode ("plan" | "execute").  Anything else — including a
    # missing value — is plan, the safe default.  execute_tool reads this
    # off the config to gate submit_gowe_job / create_group.
    config_kwargs["execution_mode"] = (
        "execute" if ctx.get("execution_mode") == "execute" else "plan"
    )

    # Literature collections searched by search_literature (RAGStack ids,
    # comma-separated; "open-access" = PubMed Central OA, the default).
    lit_collections = [
        c.strip() for c in os.environ.get("LITERATURE_RAG_COLLECTIONS", "").split(",") if c.strip()
    ]
    if lit_collections:
        config_kwargs["literature_rag_collections"] = lit_collections

    # Forward GoWe URL override if provided
    gowe_url = ctx.get("gowe_url")
    if gowe_url:
        config_kwargs["gowe_url"] = gowe_url

    # Forward session context for session-based workspace output paths.
    # submit_gowe_job uses these to rewrite output_path under
    # /<workspace_path>/chats/<session_id>/<subfolder>
    session_id = ctx.get("session_id")
    if session_id:
        config_kwargs["session_id"] = session_id
    workspace_path = ctx.get("workspace_path")
    if workspace_path:
        config_kwargs["workspace_path"] = workspace_path

    # D1: thinking switch (agents.yaml -> executor -> context["llm_options"]).
    llm_options = ctx.get("llm_options")
    if isinstance(llm_options, dict):
        for key in ("disable_thinking", "disable_thinking_classifier"):
            if llm_options.get(key):
                config_kwargs[key] = True

    # A delegated plan step (create_group guard: one group per step).
    if isinstance(ctx.get("plan_step"), dict):
        config_kwargs["plan_step"] = ctx["plan_step"]

    # F1: an agent that received a handoff cannot hand off again.
    handoff = ctx.get("handoff")
    if isinstance(handoff, dict) and handoff.get("from"):
        config_kwargs["handoff_from"] = handoff["from"]

    return config_kwargs


# Result fields worth keeping in the trace: counts, links and errors.
# Bulk payloads (docs, file text, facet buckets) are left out on purpose:
# the trace is persisted with every assistant message.
_RESULT_SUMMARY_KEYS = (
    "numFound", "count", "total", "distinct_counts", "distinct_counts_capped",
    "viewer_url", "download_tsv_url", "download_fasta_dna_url",
    "download_fasta_protein_url", "workspace_browser_url", "jobs_page_url",
    "total_size", "is_complete", "error", "errorType",
)
_RESULT_SUMMARY_MAX_CHARS = 600


def _summarize_tool_result(result: Any) -> dict[str, Any] | None:
    """A small, persisted digest of a tool result.

    Lets synthesis and the orchestrator's fallback state what a tool found
    ("4932 records", a viewer link) when the agent's own write-up came back
    empty, instead of listing the tool's arguments.
    """
    if not isinstance(result, dict):
        return None
    summary: dict[str, Any] = {}
    for key in _RESULT_SUMMARY_KEYS:
        if key in result and result[key] not in (None, "", [], {}):
            summary[key] = result[key]
    docs = result.get("docs")
    if isinstance(docs, list):
        summary["docs_returned"] = len(docs)
    if not summary:
        return None
    if len(json.dumps(summary, default=str)) > _RESULT_SUMMARY_MAX_CHARS:
        # distinct_counts can be large; keep only scalars and links.
        summary = {k: v for k, v in summary.items()
                   if not isinstance(v, (dict, list))}
    return summary


def _progress_line(step: dict[str, Any]) -> str:
    """One readable line for a tool step done before a timeout."""
    tool = step.get("tool", "tool")
    args = step.get("arguments") or {}
    what = args.get("collection") or args.get("workflow_id") or args.get("path") or ""
    head = f"`{tool}`" + (f" ({what})" if what else "")
    result = step.get("result")
    if step.get("error"):
        return f"- {head}: failed — {str(step['error'])[:200]}"
    if tool == "submit_gowe_job" and isinstance(result, dict) and result.get("submission_id"):
        out = result.get("output_path")
        return f"- {head}: job submitted" + (f", results in `{out}`" if out else "")
    summary = _summarize_tool_result(result) or {}
    bits = []
    for key in ("numFound", "count", "total"):
        if key in summary:
            bits.append(f"{summary[key]:,} records" if isinstance(summary[key], int) else f"{summary[key]} records")
            break
    if summary.get("viewer_url"):
        bits.append(f"[view](<{summary['viewer_url']}>)")
    return f"- {head}: " + (", ".join(bits) if bits else "done")


def timeout_partial_result(
    agent_type: str,
    progress: list[dict[str, Any]],
    timeout_seconds: float,
) -> Dict[str, Any] | None:
    """The answer for an agent run stopped by the time limit, or None.

    Lists the steps that finished so the user sees how far it got. Jobs
    submitted before the cut-off keep their submission_ids (status
    "completed"), so the gateway still registers watches and completion
    messages; otherwise status "error" (the answer is shown verbatim).
    """
    if not progress:
        return None
    minutes = max(1, round(timeout_seconds / 60))
    lines = [_progress_line(step) for step in progress[:20]]
    if len(progress) > 20:
        lines.append(f"- … and {len(progress) - 20} more steps")
    submission_ids = [
        step["result"]["submission_id"]
        for step in progress
        if step.get("tool") == "submit_gowe_job"
        and isinstance(step.get("result"), dict)
        and step["result"].get("submission_id")
        and not step.get("error")
    ]
    # D4: jobs prepared under the plan gate before the cut-off still get
    # their confirmation question (same shape record_execution builds).
    blocked_actions = [
        {
            "tool": step["tool"],
            "summary": step["result"].get("summary") or step["tool"],
            "arguments": step["result"].get("arguments") or step.get("arguments") or {},
        }
        for step in progress
        if isinstance(step.get("result"), dict) and step["result"].get("blocked_by_mode")
    ]
    for step in progress:
        blocked_actions.extend(prepared_actions_of(step.get("result")))
    n_prepared = len(blocked_actions)
    answer = (
        f"I ran out of time (the {minutes}-minute limit) before I could finish. "
        f"Here is how far I got:\n\n" + "\n".join(lines) + "\n\n"
        + ("The submitted job keeps running; a completion message will appear in this "
           "chat. " if submission_ids else "")
        + (f"I had prepared {n_prepared} job{'s' if n_prepared != 1 else ''} before the "
           f"cut-off; they are listed below. Answer **Yes** to submit them, **No** to "
           f"submit nothing, or tell me what to change. " if blocked_actions else "")
        + "Ask me to continue from here, or narrow the request (for example one genus, "
          "one genome group, or a count first)."
    )
    answer, _ = filter_answer(
        answer, collect_known_links(step.get("result") for step in progress)
    )
    response: Dict[str, Any] = {
        "answer": answer,
        "status": "completed" if (submission_ids or blocked_actions) else "error",
        "timed_out": True,
        "sources": [],
        "elapsed_seconds": float(timeout_seconds),
        "tool_trace": [
            {"tool": s.get("tool"), "arguments": s.get("arguments"), "error": s.get("error"),
             **({"result_summary": rs} if (rs := _summarize_tool_result(s.get("result"))) else {})}
            for s in progress
        ],
    }
    if submission_ids:
        response["submission_ids"] = submission_ids
        response["submission_id"] = submission_ids[-1]
        response["auto_submitted"] = True
    if blocked_actions:
        response["blocked_actions"] = blocked_actions
    return response


def _build_tool_trace(result: Any) -> list[dict[str, Any]]:
    """Extract a serializable tool trace from an agent result."""
    trace = []
    for ex in result.tool_trace:
        entry = {
            "tool": ex.tool_call.name,
            "arguments": ex.tool_call.arguments,
            "error": ex.error,
            "duration_ms": ex.duration_ms,
        }
        summary = _summarize_tool_result(getattr(ex, "result", None))
        if summary:
            entry["result_summary"] = summary
        trace.append(entry)
    return trace


def _known_links(result: Any) -> list[str]:
    """URLs/DOIs the agent's tools returned this turn (for filter_answer)."""
    return collect_known_links(
        getattr(ex, "result", None) for ex in getattr(result, "tool_trace", None) or []
    )


def _blocked_actions_of(result: Any) -> list[dict[str, Any]]:
    """Gated tool calls refused in plan mode (empty when none)."""
    return list(getattr(result, "blocked_actions", None) or [])


def _with_blocked_actions(response: Dict[str, Any], result: Any) -> Dict[str, Any]:
    """Attach ``blocked_actions`` and ``step_outputs`` when the agent had any.

    Every agent's response passes through here, so this is where a plan step
    gets its structured outputs (D4 Phase 3), built from the tool results.
    """
    blocked = _blocked_actions_of(result)
    if blocked:
        response["blocked_actions"] = blocked
    _add_clarification_questions(response, result)
    outputs = build_step_outputs(getattr(result, "tool_trace", None) or [])
    if outputs:
        response["step_outputs"] = outputs
    return response


def _add_clarification_questions(response: Dict[str, Any], result: Any) -> None:
    """Pass any agent's ``ask_clarification`` questions on as the card's data.

    Planning sets ``clarification_questions`` itself; the other agents keep
    the questions as JSON in ``question``.  The orchestrator turns the list
    into the clickable questions card (ASK_QUESTIONS) for whichever agent
    asked.  ``question`` becomes readable text (the plan card's step
    question) and an answer that is empty or the raw JSON becomes a short
    intro, so JSON never reaches the user (2026-10-20: the data agent's two
    questions were dropped and the user saw only "I have some questions").
    """
    if response.get("status") != "needs_input" or response.get("clarification_questions"):
        return
    raw = getattr(result, "question", None)
    questions = parse_clarification_questions(raw)
    if not questions:
        return
    response["clarification_questions"] = questions
    response["question"] = format_clarification_questions(questions)
    answer = response.get("answer")
    if not answer or not str(answer).strip() or answer == raw:
        response["answer"] = CLARIFICATION_INTRO


def _error_response(message: str) -> Dict[str, Any]:
    """Build a standard error response dict."""
    return {
        "answer": message,
        "status": "error",
        "sources": [],
        "iterations_used": 0,
        "elapsed_seconds": 0.0,
        "tool_trace": [],
    }


def _classify_agent_error(error: Exception, agent_type: str) -> str:
    """Classify an agent error and return a user-friendly message.

    Detects common error patterns (context length, auth, rate limits,
    network) and provides actionable guidance instead of raw API errors.
    """
    error_str = str(error).lower()

    # Context length / token limit errors
    if any(
        phrase in error_str
        for phrase in [
            "maximum context length",
            "context_length_exceeded",
            "too many tokens",
            "prompt is too long",
            "token limit",
        ]
    ):
        # Reached only if the loop's overflow recovery itself failed: the
        # loops compact their results and answer from them first.
        return (
            "This request pulled in more data than fits in one turn, and I could "
            "not finish it. Please narrow the request (one organism, one genome "
            "group, or a count first) and ask again."
        )

    # Authentication errors
    if any(
        phrase in error_str
        for phrase in [
            "401",
            "unauthorized",
            "authentication",
            "invalid api key",
        ]
    ):
        return (
            "Authentication error. Your session may have expired. "
            "Please try logging in again."
        )

    # Rate limit errors
    if any(
        phrase in error_str
        for phrase in [
            "429",
            "rate_limit",
            "rate limit",
            "too many requests",
        ]
    ):
        return (
            "The AI service is temporarily overloaded. "
            "Please wait a moment and try again."
        )

    # Network / timeout errors
    if any(
        phrase in error_str
        for phrase in [
            "timeout",
            "connection",
            "network",
            "unreachable",
        ]
    ):
        return (
            f"The {agent_type} agent couldn't connect to a required service. "
            "This may be a temporary network issue. Please try again."
        )

    # Default: include the raw error but with a friendlier wrapper
    return f"The {agent_type} agent encountered an error: {error}"


# ---------------------------------------------------------------------------
# Per-agent dispatch helpers
# ---------------------------------------------------------------------------


async def _run_data_agent(
    query: str,
    config_kwargs: dict[str, Any],
    ctx: dict[str, Any],
    progress_callback,
) -> Dict[str, Any]:
    """Import and run the data agent, returning a response dict."""
    from data_agent.agent import run_agent
    from data_agent.models import AgentConfig

    config = AgentConfig(**config_kwargs)
    result = await run_agent(
        query=query,
        config=config,
        context=ctx,
        progress_callback=progress_callback,
    )

    response: Dict[str, Any] = {
        "answer": result.answer,
        "status": result.status,
        "sources": result.sources,
        "iterations_used": result.iterations_used,
        "elapsed_seconds": result.elapsed_seconds,
        "tool_trace": _build_tool_trace(result),
        "known_links": _known_links(result),
    }
    if result.structured_data:
        response["structured_data"] = result.structured_data
    return _with_blocked_actions(response, result)


async def _run_service_agent(
    query: str,
    config_kwargs: dict[str, Any],
    ctx: dict[str, Any],
    progress_callback,
) -> Dict[str, Any]:
    """Import and run the service agent, returning a response dict."""
    from service_agent.agent import run_agent
    from service_agent.models import AgentConfig

    config = AgentConfig(**config_kwargs)
    result = await run_agent(
        query=query,
        config=config,
        context=ctx,
        progress_callback=progress_callback,
    )

    tool_trace = _build_tool_trace(result)

    # The populate loop, replay and the lifecycle handlers (submit/status/
    # cancel) put the reply in operation_message; pretty() is the fallback.
    if result.operation_message:
        answer = result.operation_message
    elif result.status == "needs_input" and result.question:
        answer = result.question
    elif result.status == "error" and result.error_message:
        answer = result.error_message
    else:
        answer = result.pretty()

    response: Dict[str, Any] = {
        "answer": answer,
        "status": result.status,
        "sources": result.sources,
        "elapsed_seconds": result.elapsed_seconds,
        "tool_trace": tool_trace,
        "known_links": _known_links(result),
    }
    if result.question:
        response["question"] = result.question
    # Workflow engine persistence metadata
    if result.workflow_id:
        response["workflow_id"] = result.workflow_id
    response["persisted"] = result.persisted
    if result.auto_submitted:
        response["auto_submitted"] = True
    _with_blocked_actions(response, result)
    # GoWe-specific fields
    if result.submission_id:
        response["submission_id"] = result.submission_id
    if result.submission_ids:
        response["submission_ids"] = result.submission_ids
    return response


async def _run_workspace_agent(
    query: str,
    config_kwargs: dict[str, Any],
    ctx: dict[str, Any],
    progress_callback,
) -> Dict[str, Any]:
    """Import and run the workspace agent, returning a response dict."""
    from workspace_agent.agent import run_agent
    from workspace_agent.models import AgentConfig

    config = AgentConfig(**config_kwargs)
    result = await run_agent(
        query=query,
        config=config,
        context=ctx,
        progress_callback=progress_callback,
    )

    tool_trace = _build_tool_trace(result)

    return _with_blocked_actions({
        "answer": result.answer,
        "status": result.status,
        # Keep fields aligned with other agent_chat responses
        "sources": [],
        "iterations_used": result.iterations_used,
        "elapsed_seconds": result.elapsed_seconds,
        "tool_trace": tool_trace,
        "known_links": _known_links(result),
        # Small structured extras.  The full listings and UI grids are not
        # passed on (2026-10-26): nothing rendered them, and one recursive
        # listing made this response 7.9 MB.
        "files_found": getattr(result, "files_found", 0),
        "metadata": getattr(result, "metadata", []),
        "previews": getattr(result, "previews", []),
        "paths_explored": getattr(result, "paths_explored", []),
    }, result)


async def _run_helpdesk_agent(
    query: str,
    config_kwargs: dict[str, Any],
    ctx: dict[str, Any],
    progress_callback,
) -> Dict[str, Any]:
    """Import and run the helpdesk agent, returning a response dict."""
    from helpdesk_agent.agent import run_agent
    from helpdesk_agent.models import AgentConfig

    config = AgentConfig(**config_kwargs)
    result = await run_agent(
        query=query,
        config=config,
        context=ctx,
        progress_callback=progress_callback,
    )

    return _with_blocked_actions({
        "answer": result.answer,
        "status": result.status,
        "sources": result.sources,
        "iterations_used": result.iterations_used,
        "elapsed_seconds": result.elapsed_seconds,
        "tool_trace": _build_tool_trace(result),
        "known_links": _known_links(result),
    }, result)


async def _run_analysis_agent(
    query: str,
    config_kwargs: dict[str, Any],
    ctx: dict[str, Any],
    progress_callback,
) -> Dict[str, Any]:
    """Import and run the analysis agent, returning a response dict."""
    from analysis_agent.agent import run_agent
    from analysis_agent.models import AgentConfig


    config = AgentConfig(**config_kwargs)
    result = await run_agent(
        query=query,
        config=config,
        context=ctx,
        progress_callback=progress_callback,
    )

    tool_trace = _build_tool_trace(result)

    return _with_blocked_actions({
        "answer": result.answer,
        "status": result.status,
        "sources": result.sources,
        "iterations_used": result.iterations_used,
        "elapsed_seconds": result.elapsed_seconds,
        "tool_trace": tool_trace,
        "known_links": _known_links(result),
        # Analysis agent structured payload for rich UI rendering
        "output_files": result.output_files,
        "metrics": result.metrics,
        "previews": result.previews,
        "report_links": result.report_links,
        "step_summaries": result.step_summaries,
    }, result)


async def _run_planning_agent(
    query: str,
    config_kwargs: dict[str, Any],
    ctx: dict[str, Any],
    progress_callback,
) -> Dict[str, Any]:
    """Import and run the planning agent, returning a response dict."""
    from planning_agent.agent import run_agent
    from planning_agent.models import AgentConfig


    config = AgentConfig(**config_kwargs)
    result = await run_agent(
        query=query,
        config=config,
        context=ctx,
        progress_callback=progress_callback,
    )

    tool_trace = _build_tool_trace(result)

    response: Dict[str, Any] = {
        "answer": result.answer,
        "status": result.status,
        "sources": result.sources,
        "iterations_used": result.iterations_used,
        "elapsed_seconds": result.elapsed_seconds,
        "tool_trace": tool_trace,
        "known_links": _known_links(result),
    }

    # Planning-specific fields
    if result.plan:
        response["plan"] = result.plan
    if result.clarification_questions:
        response["clarification_questions"] = result.clarification_questions
    if result.step_execution:
        response["step_execution"] = result.step_execution

    return _with_blocked_actions(response, result)


def _handoff_response(agent_type: str, slot: dict[str, Any], response: Any) -> Dict[str, Any]:
    """Result for a run that handed the turn to another agent (F1).

    The answer is empty on purpose: the executor runs ``handoff.to`` next and
    the user sees that agent's answer.  ``findings`` (the source agent's
    bounded trace) tells the receiving agent what was already looked up.
    """
    trace = response.get("tool_trace", []) if isinstance(response, dict) else []
    findings = [step for step in trace if step.get("tool") != slot.get("tool")]
    return {
        "answer": "",
        "status": "handoff",
        "handoff": {
            "from": agent_type,
            "to": slot.get("to"),
            "tool": slot.get("tool"),
            "arguments": slot.get("arguments") or {},
            "findings": findings[-15:],
        },
        "sources": [],
        "elapsed_seconds": response.get("elapsed_seconds", 0.0) if isinstance(response, dict) else 0.0,
        "tool_trace": trace,
    }


_AGENT_RUNNERS: dict[str, Callable] = {
    "data": _run_data_agent,
    "service": _run_service_agent,
    "workspace": _run_workspace_agent,
    "helpdesk": _run_helpdesk_agent,
    "analysis": _run_analysis_agent,
    "planning": _run_planning_agent,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def dispatch_agent(
    agent_type: str,
    query: str,
    context: dict | str | None,
    token: str,
    progress_callback: Callable | None = None,
) -> dict:
    """Dispatch to the appropriate agent. Returns a result dict.

    This is the single entry point for running agents, used by:
    - InProcessAgentHandle (orchestrator in-process path)
    - agent_chat MCP tool wrapper (optional, for debugging)

    Args:
        agent_type: One of "data", "service", "workspace", "helpdesk",
                    "analysis", "planning".
        query: The user's query string.
        context: Dict with recent_messages, workspace_path, session_id,
                 llm_override, images, workflow_context, page_context, etc.
                 A JSON string is also accepted (MCP wire format).
        token: BV-BRC auth token (raw, no "Bearer " prefix).
        progress_callback: Optional async callable(progress, total, message).

    Returns:
        Dict with at minimum: status, answer, tool_trace, iterations_used.
        Agent-specific keys vary by agent_type.
    """
    auth_token = _resolve_auth_token(token)
    ctx = _parse_context(context)
    config_kwargs = _build_config_kwargs(auth_token, ctx)
    config_kwargs["agent_key"] = agent_type
    # The model's context window, for the loops' prompt budget
    # (shared/context_budget.py); cached per model server.
    window = await resolve_context_window(
        config_kwargs.get("llm_base_url"), config_kwargs.get("llm_api_key"), config_kwargs.get("llm_model"),
    )
    if window:
        config_kwargs["context_window_tokens"] = window

    runner = _AGENT_RUNNERS.get(agent_type)
    if runner is None:
        return _error_response(f"Unknown agent type: {agent_type}")

    from shared.models import HANDOFF
    from shared.usage import CURRENT_AGENT

    # LLM calls in this agent run are attributed to it (shared.usage).
    agent_token = CURRENT_AGENT.set(agent_type)
    from bvbrc_mcp.tools.groups import GROUP_RUN
    # Groups this run prepares or creates, for create_group's guard.
    group_token = GROUP_RUN.set({})

    handoff_slot: dict[str, Any] = {}
    handoff_token = HANDOFF.set(handoff_slot)
    try:
        response = await runner(query, config_kwargs, ctx, progress_callback)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return _error_response(_classify_agent_error(e, agent_type))
    finally:
        HANDOFF.reset(handoff_token)
        CURRENT_AGENT.reset(agent_token)
        GROUP_RUN.reset(group_token)

    if handoff_slot:
        return _handoff_response(agent_type, handoff_slot, response)

    # Unwrap links no tool returned this turn (models invent BV-BRC URLs).
    # known_links stays on the response so the orchestrator can filter a
    # synthesized answer the same way.
    if isinstance(response, dict) and isinstance(response.get("answer"), str):
        response["answer"], removed = filter_answer(
            response["answer"], response.get("known_links") or []
        )
        if removed:
            response["links_removed"] = removed
    return response
