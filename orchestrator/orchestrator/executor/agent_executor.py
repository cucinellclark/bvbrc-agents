"""Single agent step executor.

Executes one step of a plan by calling the agent's chat tool via MCP
through the registry. Yields Event objects for progress and results.
"""

from __future__ import annotations

import asyncio
import json
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
from orchestrator.registry.agent_handle import AgentHandle
from orchestrator.router.models import Step

logger = logging.getLogger(__name__)

# Default MCP tool name for the agent's chat entry point
DEFAULT_CHAT_TOOL = "agent_chat"


async def execute_agent_step(
    step: Step,
    agent: AgentHandle,
    request: OrchestratorRequest,
    step_index: int = 0,
    upstream_results: dict[str, Any] | None = None,
) -> AsyncGenerator[Event, None]:
    """Execute a single agent step via the agent's chat tool.

    Calls agent_chat on the target agent with the user's query, yielding
    events for progress tracking.

    Args:
        step: The plan step to execute.
        agent: The AgentHandle for the target agent.
        request: The original orchestrator request (for auth token, context).
        step_index: Index of this step in the plan (for event tracing).
        upstream_results: Results from upstream pipeline steps to thread
            into this agent's context. Keys are like "step_0_data" and
            values are the result_for_ui dicts from those steps.

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

    # Pass context if available
    context_data: dict[str, Any] = {}
    if request.conversation_summary:
        context_data["conversation_summary"] = request.conversation_summary
    if request.recent_messages:
        context_data["recent_messages"] = request.recent_messages[-5:]
    if request.workspace_path:
        context_data["workspace_path"] = request.workspace_path
    if request.selected_items:
        context_data["selected_items"] = request.selected_items

    # Pass session_id so agents can thread it to the workflow engine
    if request.session_id:
        context_data["session_id"] = request.session_id

    # Pass auto-submit preference so the service agent can auto-submit
    if request.auto_submit_preference:
        context_data["auto_submit_preference"] = request.auto_submit_preference

    # Thread upstream results into context for pipeline steps
    if upstream_results:
        context_data["upstream_results"] = upstream_results

    # Forward LLM override so agents can use the user-selected model
    if request.llm_override:
        context_data["llm_override"] = {
            "base_url": request.llm_override.base_url,
            "api_key": request.llm_override.api_key,
            "model": request.llm_override.model,
        }

    # Forward attached files so agents can use file content
    if request.attached_files:
        context_data["attached_files"] = request.attached_files

    if context_data:
        arguments["context"] = json.dumps(context_data)

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
        # MCP progress notifications arrive via an async callback, but
        # execute_agent_step is an async generator that yields Events.
        # Use an asyncio.Queue to bridge the two: the callback puts
        # AGENT_PROGRESS Events into the queue, and we yield them while
        # waiting for the MCP call to finish.
        progress_queue: asyncio.Queue[Event] = asyncio.Queue()

        async def _on_progress(
            progress: float, total: float | None, message: str | None
        ) -> None:
            event = Event(
                type=EventType.AGENT_PROGRESS,
                agent_name=agent.key,
                step_index=step_index,
                data={
                    "agent": agent.key,
                    "message": message or "",
                    "progress": progress,
                    "total": total,
                },
            )
            await progress_queue.put(event)

        # Launch the MCP call as a task so we can yield progress in parallel
        call_task = asyncio.create_task(
            agent.call_tool(chat_tool, arguments, progress_handler=_on_progress)
        )

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

        # Get the result (raises if the task failed)
        mcp_result = call_task.result()
        elapsed_ms = (time.monotonic() - start_time) * 1000

        # Parse the MCP result
        result_data = _parse_mcp_result(mcp_result)

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

        # Preserve agent-specific fields (e.g., Service2's manifest,
        # workflow_plan, question) so they are available downstream
        # in the pipeline and in the synthesizer.
        _STANDARD_KEYS = {
            "answer", "status", "sources", "elapsed_seconds",
            "iterations_used", "tool_trace",
        }
        for key, value in result_data.items():
            if key not in _STANDARD_KEYS and key not in result_for_ui:
                result_for_ui[key] = value

        # Check for errors from the agent
        is_error = getattr(mcp_result, "isError", False)
        if is_error or result_data.get("status") == "error":
            yield error_event(
                message=f"Agent '{agent.key}' returned an error: {answer}",
                agent_name=agent.key,
                details=result_for_ui,
            )
            return

        yield agent_result_event(
            agent_name=agent.key,
            result_for_llm=result_for_llm,
            result_for_ui=result_for_ui,
        )

    except Exception as e:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.error(f"Agent step execution failed for '{agent.key}': {e}")
        yield error_event(
            message=f"Agent '{agent.key}' execution failed: {e}",
            agent_name=agent.key,
            details={"elapsed_ms": round(elapsed_ms, 1)},
        )


def _parse_mcp_result(mcp_result: Any) -> dict[str, Any]:
    """Parse an MCP CallToolResult into a dict.

    The agent_chat tool returns a JSON dict, but MCP wraps it in
    content blocks. Extract and parse the actual data.
    """
    if not hasattr(mcp_result, "content") or not mcp_result.content:
        return {"answer": "(empty result from agent)", "status": "error"}

    for block in mcp_result.content:
        if hasattr(block, "text"):
            try:
                return json.loads(block.text)
            except (json.JSONDecodeError, TypeError):
                return {"answer": block.text, "status": "completed"}

    return {"answer": str(mcp_result.content), "status": "completed"}
