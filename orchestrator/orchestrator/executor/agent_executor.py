"""Single agent step executor.

Executes one step of a plan by calling the agent's chat tool through
the registry. Yields Event objects for progress and results.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from orchestrator.events.events import (
    Event,
    EventType,
    agent_start_event,
    agent_result_event,
    error_event,
)
from orchestrator.models import OrchestratorRequest
from orchestrator.registry.agent_handle import AgentHandle, InProcessAgentHandle
from orchestrator.router.models import Step

# Plain-language names for agent keys in user-facing error text.
_AGENT_LABELS = {
    "data": "data search",
    "service": "analysis-service",
    "workspace": "workspace",
    "helpdesk": "help",
    "analysis": "results-analysis",
    "planning": "planning",
}


def user_facing_agent_error(agent_key: str, exc: BaseException, timeout_seconds: int | None) -> str:
    """The message a user sees when an agent run raises.

    The exception text is for logs, not users: an asyncio timeout has an
    empty str(), so the old f"Agent '{key}' execution failed: {e}" showed
    users a dangling "execution failed: " (42 hackathon turns).
    """
    label = _AGENT_LABELS.get(agent_key, agent_key)
    if isinstance(exc, TimeoutError):
        minutes = max(1, round((timeout_seconds or 300) / 60))
        return (
            f"Sorry — the {label} step took longer than the {minutes}-minute limit "
            "and was stopped before it could answer. This usually means the question "
            "needs a lot of data at once. Try narrowing it (for example, one genus, "
            "one genome group, or a smaller time range), or ask for a count first."
        )
    return (
        f"Sorry — the {label} step ran into an unexpected problem and could not "
        "finish. Please try again; if it keeps happening, try rephrasing the question."
    )

logger = logging.getLogger(__name__)

# Default MCP tool name for the agent's chat entry point
DEFAULT_CHAT_TOOL = "agent_chat"


async def execute_agent_step(
    step: Step,
    agent: AgentHandle | InProcessAgentHandle,
    request: OrchestratorRequest,
    step_index: int = 0,
    upstream_results: dict[str, Any] | None = None,
    handoff: dict[str, Any] | None = None,
) -> AsyncGenerator[Event, None]:
    """Execute a single agent step via the agent's chat tool.

    Calls agent_chat on the target agent with the user's query, yielding
    events for progress tracking.

    Args:
        step: The plan step to execute.
        agent: Handle for the target agent (in-process or MCP).
        request: The original orchestrator request (for auth token, context).
        step_index: Index of this step in the plan (for event tracing).
        upstream_results: Results from upstream pipeline steps to thread
            into this agent's context. Keys are like "step_0_data" and
            values are the result_for_ui dicts from those steps.
        handoff: Set when another agent handed this turn over (F1):
            ``{from, to, tool, arguments, findings}``, forwarded as
            ``context["handoff"]``.

    Yields:
        Event objects: AGENT_START, AGENT_TOOL_CALL, AGENT_TOOL_RESULT,
        AGENT_RESULT, or AGENT_ERROR.
    """
    # Determine the chat tool name from agent config or use default
    chat_tool = getattr(agent.config, "chat_tool", DEFAULT_CHAT_TOOL)

    yield agent_start_event(agent_name=agent.key, task=step.task)

    # Build arguments for the agent_chat tool
    arguments: dict[str, Any] = {"query": step.task}

    # Inject agent-specific chat tool params (e.g., agent_type)
    if agent.config.chat_tool_params:
        arguments.update(agent.config.chat_tool_params)

    # Pass context if available.
    # There is no full-history summary: the gateway's unbounded
    # conversation_summary bloated agent prompts and caused LLM timeouts, and
    # was removed (2026-10-27).  Agents receive the bounded recent_messages.
    # For single-agent routing, step.task is the original user query
    # (router does not rewrite it). Pipeline steps still use per-step
    # task text. recent_messages is token-budgeted to ~4000 tokens by
    # the gateway for conversation context.
    context_data: dict[str, Any] = {}
    if request.recent_messages:
        context_data["recent_messages"] = request.recent_messages[-5:]
    if request.workspace_path:
        context_data["workspace_path"] = request.workspace_path
    if request.selected_items:
        context_data["selected_items"] = request.selected_items

    # Pass session_id so agents can thread it to the workflow engine
    if request.session_id:
        context_data["session_id"] = request.session_id

    # Execution mode gates submit_gowe_job / create_group in execute_tool.
    context_data["execution_mode"] = request.execution_mode
    if request.replay_actions:
        context_data["replay_actions"] = request.replay_actions
    if request.prepared_actions:
        context_data["prepared_actions"] = request.prepared_actions

    # Forward page context (description of the page the user is viewing)
    if request.page_context:
        context_data["page_context"] = request.page_context

    # Thread upstream results into context for pipeline steps
    if upstream_results:
        context_data["upstream_results"] = upstream_results

    # Forward LLM override so agents can use the user-selected model
    if request.llm_override:
        llm_override_data: dict[str, Any] = {
            "base_url": request.llm_override.base_url,
            "api_key": request.llm_override.api_key,
            "model": request.llm_override.model,
        }
        if request.llm_override.max_tokens is not None:
            llm_override_data["max_tokens"] = request.llm_override.max_tokens
        context_data["llm_override"] = llm_override_data

    # NOTE: attached_files is intentionally NOT forwarded to agents.
    # Full file content must not reach agent prompts (could be up to 10 MB).
    # Agents receive the processed excerpts + workspace paths via
    # parsed_documents instead.

    # Forward images for multimodal content (base64 data URIs)
    if request.images:
        context_data["images"] = request.images

    # Forward parsed documents (PDFs + text uploads: excerpts + metadata)
    if request.parsed_documents:
        context_data["parsed_documents"] = request.parsed_documents

    # Forward workflow context for the analysis agent
    if request.workflow_context:
        context_data["workflow_context"] = request.workflow_context

    # Forward batch_mode flag so the service agent knows whether
    # multiple submissions are permitted (planning-delegated steps only)
    if step.batch_mode:
        context_data["batch_mode"] = True
    if step.plan_step:
        context_data["plan_step"] = step.plan_step

    if handoff:
        context_data["handoff"] = handoff

    # D1: per-agent thinking switch from agents.yaml.
    llm_options = {
        key: True
        for key in ("disable_thinking", "disable_thinking_classifier")
        if getattr(agent.config, key, False)
    }
    if llm_options:
        context_data["llm_options"] = llm_options

    if context_data:
        arguments["context"] = context_data

    # Pass auth token if available
    if request.auth_token:
        arguments["token"] = request.auth_token

    # Emit tool call event
    yield Event(
        type=EventType.AGENT_TOOL_CALL,
        agent_name=agent.key,
        step_index=step_index,
        data={
            "agent": agent.key,
            "tool": chat_tool,
            "arguments": {"query": step.task},  # Don't log token
            "mcp_server_name": agent.config.mcp_server_name,
        },
    )

    start_time = time.monotonic()

    try:
        # Check that the agent has the chat tool
        if chat_tool not in agent.tool_names:
            yield error_event(
                message=(
                    f"Agent '{agent.key}' does not have the '{chat_tool}' "
                    f"tool. Available tools: {', '.join(agent.tool_names)}"
                ),
                agent_name=agent.key,
            )
            return

        # --- Progress notification bridge ---
        # Progress callbacks arrive via an async callable, but
        # execute_agent_step is an async generator that yields Events.
        # Use an asyncio.Queue to bridge the two: the callback puts
        # AGENT_PROGRESS Events into the queue, and we yield them while
        # waiting for the agent call to finish.
        progress_queue: asyncio.Queue[Event] = asyncio.Queue()

        # Structured steps (shared.progress): the sink installed for this run
        # receives every step the agent reports — tool calls, model calls,
        # retries, compaction — and the gateway forwards them as `progress`
        # SSE events with the step fields on the payload.
        from shared import progress as _progress

        async def _emit_step(step: dict) -> None:
            event = Event(
                type=EventType.AGENT_PROGRESS,
                agent_name=agent.key,
                step_index=step_index,
                data={
                    "agent": agent.key,
                    "message": step.get("label") or "",
                    "progress": step.get("ts_ms") or 0,
                    "total": None,
                    **step,
                },
            )
            await progress_queue.put(event)

        async def _on_progress(
            progress: float, total: float | None, message: Any
        ) -> None:
            # Legacy callback shape (message may be a plain string).
            if isinstance(message, dict):
                await _emit_step(message)
                return
            await _emit_step(_progress.public(_progress.finish(
                _progress.new_step("note", str(message or ""), transient=True)
            )))

        run_tokens = _progress.start_run(_emit_step, agent.key)

        # Launch the agent call as a task so we can yield progress in parallel.
        # If the caller abandons this generator (e.g. gateway aborts the SSE
        # stream), the finally block cancels the task so the LLM/tool work
        # stops promptly.  The task copies this context, so it reports into
        # the sink above.
        call_task = asyncio.create_task(
            agent.call_tool(chat_tool, arguments, progress_handler=_on_progress)
        )

        try:
            # Yield progress events as they arrive until the call completes
            while not call_task.done():
                try:
                    event = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                    yield event
                except asyncio.TimeoutError:
                    continue

            # Drain any remaining queued progress events
            while not progress_queue.empty():
                yield await progress_queue.get()
        except (asyncio.CancelledError, GeneratorExit):
            # Generator was abandoned (SSE disconnect / Stop).
            # Cancel the agent task so it stops doing LLM/tool work.
            if not call_task.done():
                call_task.cancel()
                try:
                    await call_task
                except (asyncio.CancelledError, Exception):
                    pass
            logger.info(f"Agent step for '{agent.key}' cancelled")
            raise
        finally:
            _progress.end_run(run_tokens)

        # Get the result (raises if the task failed). Both handle types
        # return a plain dict.
        result_data = call_task.result()
        elapsed_ms = (time.monotonic() - start_time) * 1000

        # Emit tool result event
        yield Event(
            type=EventType.AGENT_TOOL_RESULT,
            agent_name=agent.key,
            step_index=step_index,
            data={
                "agent": agent.key,
                "tool": chat_tool,
                "elapsed_ms": round(elapsed_ms, 1),
                "status": result_data.get("status", "unknown"),
                "mcp_server_name": agent.config.mcp_server_name,
                "tool_trace": result_data.get("tool_trace", []),
            },
        )

        # Build result_for_llm and result_for_ui
        answer = result_data.get("answer", "")
        result_for_llm = answer
        result_for_ui = {
            "agent": agent.key,
            "answer": answer,
            "status": result_data.get("status", "unknown"),
            "sources": result_data.get("sources", []),
            "elapsed_seconds": result_data.get("elapsed_seconds", 0.0),
            "iterations_used": result_data.get("iterations_used", 0),
            "tool_trace": result_data.get("tool_trace", []),
            "mcp_server_name": agent.config.mcp_server_name,
        }

        # Preserve agent-specific fields (e.g., the service agent's
        # submission_ids, question) so they are available downstream
        # in the pipeline and in the synthesizer.
        _STANDARD_KEYS = {
            "answer", "status", "sources", "elapsed_seconds",
            "iterations_used", "tool_trace", "known_links",
        }
        for key, value in result_data.items():
            if key not in _STANDARD_KEYS and key not in result_for_ui:
                result_for_ui[key] = value

        # Check for errors from the agent
        if result_data.get("status") == "error":
            # The agent's error answer is already written for the user
            # (e.g. the service agent's submission-failure explanation);
            # no "Agent 'x' returned an error:" prefix in front of it.
            yield error_event(
                message=answer or user_facing_agent_error(agent.key, RuntimeError(), None),
                agent_name=agent.key,
                details=result_for_ui,
            )
            return

        yield agent_result_event(
            agent_name=agent.key,
            result_for_llm=result_for_llm,
            result_for_ui=result_for_ui,
            known_links=result_data.get("known_links"),
        )

    except Exception as e:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.error(
            f"Agent step execution failed for '{agent.key}' after "
            f"{elapsed_ms / 1000:.0f}s: {type(e).__name__}: {e}"
        )
        yield error_event(
            message=user_facing_agent_error(
                agent.key, e, getattr(agent.config, "timeout_seconds", None)
            ),
            agent_name=agent.key,
            details={
                "elapsed_ms": round(elapsed_ms, 1),
                "internal_error": f"{type(e).__name__}: {e}",
            },
        )
