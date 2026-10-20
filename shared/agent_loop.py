"""Generic agent LLM loop shared by all standard BV-BRC agents.

Extracts the common plan-execute-evaluate loop that was previously
duplicated across data, workspace, helpdesk, and analysis agents.

Usage::

    from shared.agent_loop import run_agent_loop

    result = await run_agent_loop(
        state=state,
        config=cfg,
        client=client,
        system_prompt=SYSTEM_PROMPT,
        tool_schemas=TOOL_SCHEMAS,
        progress_callback=progress_callback,
        progress_messages={"workspace_browse": "Browsing workspace..."},
        done_message="Workspace exploration complete.",
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict

from shared.agent_utils import (
    CLARIFICATION_INTRO,
    build_user_content,
    is_context_window_error,
    call_fingerprint,
    format_attached_documents,
    format_recent_messages,
    format_execution_mode,
    format_session_workspace,
    format_handoff,
    parse_tool_calls as _parse_tool_calls_raw,
    get_response_content,
    build_tool_calls_message,
    emit_progress,
    trim_messages_to_fit,
)
from shared.agent_messages import (
    BUDGET_ANSWER_NOW,
    BUDGET_FALLBACK,
    DUPLICATE_CALL_WARNING,
    EMPTY_ANSWER_NUDGE,
    OVERFLOW_ANSWER_NOW,
    MAX_ITERATIONS_SYNTHESIS,
    MAX_ITERATIONS_FALLBACK,
    STUCK_ANSWER_NOW,
    STUCK_FALLBACK,
)
from shared.context_budget import ContextBudget
from shared.models import BaseAgentState, BaseAgentConfig, ToolCall
from shared.tools import execute_tool, is_handoff_result, truncate_result, result_char_limit

logger = logging.getLogger(__name__)


ProgressCallback = Any  # async (progress: float, total: float|None, message: str) -> None

# Type for the progress message builder function
ProgressMessageFn = Callable[[ToolCall], str]


def _check_cancelled() -> None:
    """Raise CancelledError if the current asyncio task has been cancelled.

    Called at natural checkpoints in the agent loop (between LLM calls
    and tool executions) so that a cancelled task stops promptly
    instead of continuing to the next iteration.
    """
    task = asyncio.current_task()
    if task is not None and task.cancelled():
        raise asyncio.CancelledError()


def _default_progress_message(tc: ToolCall) -> str:
    """Default progress message: just show the tool name."""
    return f"Calling {tc.name}..."


def _default_result_message(tc: ToolCall, result: Any) -> str:
    """Default result message after a tool completes."""
    msg = f"Tool {tc.name} completed."
    if isinstance(result, dict):
        nf = result.get("numFound") or result.get("count")
        if nf is not None:
            msg = f"Found {nf} records."
        items = result.get("items")
        if isinstance(items, list):
            msg = f"Found {len(items)} items."
        if result.get("error"):
            msg = "Query returned an error, adjusting approach..."
    return msg


async def force_answer(
    state: BaseAgentState,
    client: Any,
    config: BaseAgentConfig,
    chat_completion_fn: Callable,
    tool_schemas: list[dict] | None = None,
    trim_messages: bool = False,
    nudge: str = STUCK_ANSWER_NOW,
    fallback: str = STUCK_FALLBACK,
    budget: ContextBudget | None = None,
    compact_all: bool = False,
) -> None:
    """End the loop with an answer: one nudge, one call with tools off, then stop.

    Used when the model is stuck, when the context budget is spent, and
    after a context overflow (``compact_all`` first shrinks every tool
    result).  If the call itself overflows, every result is compacted and it
    is tried once more.  Sets ``state.final_answer`` and ``state.status =
    "completed"``; an empty or failed answer falls back to *fallback*, which
    names the tools that ran.
    """
    if compact_all and budget is not None:
        budget.compact_all(state.messages)
    state.add_user_message(nudge)
    content = ""
    for attempt in range(2):
        messages = state.messages
        if trim_messages:
            messages = trim_messages_to_fit(state.messages, tool_schemas, config.max_context_tokens)
        try:
            response = await chat_completion_fn(
                client=client, messages=messages, tools=tool_schemas, config=config, tool_choice="none",
            )
            content = (get_response_content(response) or "").strip()
            break
        except Exception as exc:  # the fallback below still answers
            if attempt == 0 and budget is not None and is_context_window_error(exc) and not compact_all:
                logger.warning("Forced answer overflowed the context; compacting every result and retrying")
                budget.compact_all(state.messages)
                compact_all = True
                continue
            logger.warning("Forced answer failed: %s", exc)
            break
    if not content:
        steps = "\n".join(
            f"- `{te.tool_call.name}`" + (f": failed — {str(te.error)[:120]}" if te.error else "")
            for te in state.tool_executions[-10:]
        ) or "- (no tool ran)"
        content = fallback.format(steps=steps)
    state.final_answer = content
    state.status = "completed"


async def run_agent_loop(
    *,
    state: BaseAgentState,
    config: BaseAgentConfig,
    client: Any,
    system_prompt: str,
    tool_schemas: list[dict] | None = None,
    chat_completion_fn: Any = None,
    chat_completion_stream_fn: Any = None,
    progress_callback: ProgressCallback | None = None,
    progress_message_fn: ProgressMessageFn | None = None,
    result_message_fn: Callable[[ToolCall, Any], str] | None = None,
    on_tool_result: Callable[[ToolCall, Any, BaseAgentState], str | None] | None = None,
    trim_messages: bool = False,
    max_tool_result_chars: int = 8000,
    stuck_threshold: int = 2,
    start_message: str = "Analyzing your question...",
    done_message: str = "Done.",
    excluded_context_keys: set[str] | None = None,
) -> None:
    """Run the standard plan-execute-evaluate agent loop.

    This function mutates ``state`` in place (adds messages, records tool
    executions, sets ``final_answer`` and ``status``). The caller is
    responsible for calling ``state.to_result()`` after this returns.

    Args:
        state: Agent state object (must be a BaseAgentState subclass).
        config: Agent config (must be a BaseAgentConfig subclass).
        client: OpenAI-compatible client (from ``create_client``).
        system_prompt: The agent's system prompt string.
        tool_schemas: Tool schemas to pass to the LLM. Defaults to ALL_TOOL_SCHEMAS.
        chat_completion_fn: Async function for non-streaming LLM calls.
        chat_completion_stream_fn: Async generator for streaming LLM calls.
        progress_callback: Optional async callback for progress updates.
        progress_message_fn: Function(tc) -> str for tool-specific progress messages.
        result_message_fn: Function(tc, result) -> str for result-specific messages.
        on_tool_result: Optional hook called after each tool execution.
            If it returns a non-None string, the loop exits immediately
            (used by planning agent for early returns on create_plan, etc.).
        trim_messages: Whether to trim message history before each LLM call.
        max_tool_result_chars: Max chars for truncated tool results.
        stuck_threshold: Number of duplicate calls before breaking the loop.
        start_message: Progress message shown at the start.
        done_message: Progress message shown at the end.
        excluded_context_keys: Context keys to exclude from the JSON dump
            (defaults to ``{"page_context", "images"}``).
    """
    from shared.tools.schemas import ALL_TOOL_SCHEMAS
    from shared.tools.registry import TOOL_DISPATCH

    if tool_schemas is None:
        tool_schemas = ALL_TOOL_SCHEMAS
    if progress_message_fn is None:
        progress_message_fn = _default_progress_message
    if result_message_fn is None:
        result_message_fn = _default_result_message
    if excluded_context_keys is None:
        excluded_context_keys = {
            "page_context", "images",
            "conversation_summary", "recent_messages",
            "parsed_documents", "attached_files", "handoff", "llm_override", "llm_options",
        }

    # Build initial messages if the caller hasn't already set them up.
    # Some agents (e.g., analysis) need to inject extra user messages
    # (workflow context) before the query, so they pre-populate messages.
    if not state.messages:
        system_content = system_prompt
        context = state.context
        if context:
            page_context = context.get("page_context", "")
            if page_context:
                system_content += (
                    f"\n\n=== PAGE CONTEXT ===\n"
                    f"The user is currently viewing the following page:\n"
                    f"{page_context}"
                )

            # Inject attached document excerpts (PDFs + text uploads)
            docs_section = format_attached_documents(
                context.get("parsed_documents")
            )
            if docs_section:
                system_content += f"\n\n{docs_section}"

            # Inject session workspace path for chat file discovery
            session_ws = format_session_workspace(context)
            if session_ws:
                system_content += f"\n\n{session_ws}"
            # Another agent passed this turn on (F1 handoff)
            handoff_section = format_handoff(context)
            if handoff_section:
                system_content += f"\n\n{handoff_section}"

            # Inject bounded conversation context from recent_messages.
            recent_msgs = context.get("recent_messages")
            if recent_msgs:
                formatted = format_recent_messages(recent_msgs)
                if formatted:
                    system_content += (
                        f"\n\n=== CONVERSATION CONTEXT ===\n{formatted}"
                    )

            ctx_for_prompt = {
                k: v for k, v in context.items()
                if k not in excluded_context_keys
            }
            if ctx_for_prompt:
                system_content += (
                    f"\n\n=== ADDITIONAL CONTEXT ===\n"
                    f"{json.dumps(ctx_for_prompt, default=str)}"
                )

        # Tell the LLM which side-effecting tools the gate will refuse.
        # Always injected — the gate applies even without gateway context.
        system_content += f"\n\n{format_execution_mode(config)}"

        state.add_system_message(system_content)
        images = context.get("images", []) if context else []
        state.add_user_message(build_user_content(state.query, images))

    await emit_progress(progress_callback, 0, None, start_message)

    # Build auth headers
    headers: Dict[str, str] | None = None
    if config.bvbrc_auth_token:
        headers = {"Authorization": config.bvbrc_auth_token}

    # Track fingerprints for duplicate detection
    executed_fingerprints: set[str] = set()
    empty_answer_nudged = False
    duplicate_count = 0
    # Prompt budget: compacts older results past the soft line, forces the
    # answer past the hard line, recovers from an overflow (#12, 2026-10-20).
    budget = ContextBudget.for_config(config)

    for iteration in range(config.max_iterations):
        # --- Cancel checkpoint: top of each iteration ---
        _check_cancelled()

        state.iteration = iteration + 1

        await emit_progress(
            progress_callback,
            iteration,
            config.max_iterations,
            "Planning next step...",
        )

        # 1. PLAN -- Ask the LLM what to do next
        prep = budget.prepare(state.messages, tool_schemas)
        if prep["compacted"]:
            logger.info("Compacted %d older tool result(s); %s", prep["compacted"], budget.describe())
        if prep["exhausted"]:
            logger.warning("Context budget spent after %d tool call(s); forcing an answer (%s)",
                           len(state.tool_executions), budget.describe())
            await force_answer(
                state, client, config, chat_completion_fn, tool_schemas,
                trim_messages=trim_messages, nudge=BUDGET_ANSWER_NOW, fallback=BUDGET_FALLBACK, budget=budget,
            )
            break

        messages_for_llm = state.messages
        if trim_messages:
            messages_for_llm = trim_messages_to_fit(
                state.messages, tool_schemas, config.max_context_tokens
            )

        try:
            response = await chat_completion_fn(
                client=client,
                messages=messages_for_llm,
                tools=tool_schemas,
                config=config,
            )
        except Exception as exc:
            if not is_context_window_error(exc):
                raise
            # The work so far is kept: every result is compacted and the
            # model answers from it (before, the turn was discarded with a
            # "start a new conversation" message).
            logger.warning("Context overflow after %d tool call(s); compacting and answering (%s)",
                           len(state.tool_executions), budget.describe())
            await force_answer(
                state, client, config, chat_completion_fn, tool_schemas,
                trim_messages=trim_messages, nudge=OVERFLOW_ANSWER_NOW, fallback=BUDGET_FALLBACK,
                budget=budget, compact_all=True,
            )
            break
        budget.record(response, messages_for_llm, tool_schemas)

        tool_calls = _parse_tool_calls_raw(response, ToolCall)
        content = get_response_content(response)

        # --- Cancel checkpoint: after LLM call, before tool execution ---
        _check_cancelled()

        # 2. CHECK -- If no tool calls, the LLM produced a final answer
        if (
            not tool_calls
            and not (content or "").strip()
            and state.tool_executions
            and not empty_answer_nudged
        ):
            # Empty answer after tools ran: ask once for the write-up
            # rather than ending the turn with nothing to show.
            empty_answer_nudged = True
            logger.warning(
                "Empty final answer after %d tool call(s); nudging once",
                len(state.tool_executions),
            )
            state.add_user_message(EMPTY_ANSWER_NUDGE)
            continue

        if not tool_calls:
            await emit_progress(
                progress_callback,
                iteration + 1,
                config.max_iterations,
                "Composing answer...",
            )
            state.final_answer = content or ""
            state.status = "completed"
            break

        # Add assistant message with tool_calls to conversation
        state.add_assistant_message(
            content=content,
            tool_calls=build_tool_calls_message(tool_calls),
        )

        # 3. EXECUTE -- Run each tool call and feed results back
        should_exit = False
        stuck = False
        for i, tc in enumerate(tool_calls):
            fp = call_fingerprint(tc)

            # --- Duplicate detection ---
            if fp in executed_fingerprints:
                duplicate_count += 1
                state.add_tool_result(
                    tc.id,
                    json.dumps(
                        {"_duplicate": True, "_message": DUPLICATE_CALL_WARNING}
                    ),
                )
                if duplicate_count >= stuck_threshold:
                    # Every call in the response needs a result before the
                    # next LLM call.
                    for rest in tool_calls[i + 1:]:
                        state.add_tool_result(
                            rest.id, json.dumps({"_skipped": True, "_message": STUCK_ANSWER_NOW})
                        )
                    stuck = True
                    break
                continue

            # --- Cancel checkpoint: before each tool execution ---
            _check_cancelled()

            # Progress message
            tool_msg = progress_message_fn(tc)
            await emit_progress(
                progress_callback, iteration, config.max_iterations, tool_msg
            )

            # Execute the tool
            start = time.time()
            result = await execute_tool(
                tool_name=tc.name,
                arguments=dict(tc.arguments),  # copy to avoid mutation
                dispatch_table=TOOL_DISPATCH,
                timeout_seconds=config.tool_timeout_seconds,
                config=config,
                headers=headers,
            )
            duration_ms = (time.time() - start) * 1000

            # Record the execution
            executed_fingerprints.add(fp)
            error = result.get("error") if isinstance(result, dict) else None
            state.record_execution(
                tc=tc,
                result=result,
                error=error,
                duration_ms=duration_ms,
            )

            # Result progress message
            res_msg = result_message_fn(tc, result)
            await emit_progress(
                progress_callback, iteration, config.max_iterations, res_msg
            )

            # --- F1 handoff: the call belongs to another agent ---
            # record_execution has filled the dispatcher's handoff slot;
            # stop without writing a reply (the user sees the next agent's).
            if is_handoff_result(result):
                state.status = "handoff"
                state.final_answer = ""
                state.add_tool_result(tc.id, json.dumps(result, default=str))
                should_exit = True
                break

            # --- ask_clarification detection ---
            # When an agent explicitly calls ask_clarification with a
            # valid result, the agent is requesting user input.  Exit
            # the loop with status "needs_input" so the orchestrator
            # can pause plan execution and surface the question to the
            # user.  This mirrors the planning agent's handling in
            # agents/planning_agent/agent.py.
            if (
                tc.name == "ask_clarification"
                and isinstance(result, dict)
                and result.get("status") == "valid"
            ):
                questions = result.get("questions", [])
                state.status = "needs_input"
                state.question = json.dumps(questions)
                state.final_answer = content or CLARIFICATION_INTRO
                result_str = truncate_result(result, max_chars=result_char_limit(tc.name, max_tool_result_chars))
                state.add_tool_result(tc.id, result_str)
                should_exit = True
                break

            # Optional hook for special tool result handling
            if on_tool_result is not None:
                action = on_tool_result(tc, result, state)
                if action is not None:
                    # Hook wants to exit the loop (e.g., plan created)
                    should_exit = True
                    # Still need to add the tool result to messages
                    result_str = truncate_result(result, max_chars=result_char_limit(tc.name, max_tool_result_chars))
                    state.add_tool_result(tc.id, result_str)
                    break

            # Serialize and truncate for the LLM context
            result_str = truncate_result(result, max_chars=result_char_limit(tc.name, max_tool_result_chars))
            state.add_tool_result(tc.id, result_str)

        if should_exit:
            break

        if stuck:
            # Before 2026-10-14 the break above only left this response's
            # tool calls: the next LLM call repeated the same call, ~90
            # times in one benchmark task (2.4 M prompt tokens, 300 s cap).
            logger.warning(
                "Agent repeated a tool call %d time(s); forcing an answer", duplicate_count
            )
            await force_answer(
                state, client, config, chat_completion_fn, tool_schemas,
                trim_messages=trim_messages,
            )
            break

    else:
        # Hit max iterations -- force synthesis
        state.status = "max_iterations"
        await emit_progress(
            progress_callback,
            config.max_iterations,
            config.max_iterations,
            f"Synthesizing answer from {len(state.tool_executions)} tool calls...",
        )

        if chat_completion_stream_fn is not None:
            try:
                state.add_user_message(MAX_ITERATIONS_SYNTHESIS)
                messages_for_synth = state.messages
                if trim_messages:
                    messages_for_synth = trim_messages_to_fit(
                        state.messages, tool_schemas, config.max_context_tokens
                    )
                synthesis_content = ""
                async for chunk in chat_completion_stream_fn(
                    client=client,
                    messages=messages_for_synth,
                    config=config,
                    tool_choice="none",
                ):
                    synthesis_content += chunk
                state.final_answer = synthesis_content or (
                    MAX_ITERATIONS_FALLBACK.format(n=len(state.tool_executions))
                )
            except Exception:
                state.final_answer = MAX_ITERATIONS_FALLBACK.format(
                    n=len(state.tool_executions)
                )
        else:
            state.final_answer = MAX_ITERATIONS_FALLBACK.format(
                n=len(state.tool_executions)
            )

    await emit_progress(
        progress_callback,
        config.max_iterations,
        config.max_iterations,
        done_message,
    )
