"""
Core agent loop for the BV-BRC Data Retrieval Agent.

Supports two modes:
  - plan_only(): Sends the query to the LLM with tool schemas and captures the
    planned tool_calls WITHOUT executing them. For testing/inspecting.
  - run_agent(): Full plan-execute-evaluate loop using the shared agent loop.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ensure repo root and shared/config dirs are on sys.path
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
_SHARED_DIR = str(Path(__file__).resolve().parent.parent.parent / "shared")
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent.parent / "config")
for _p in (_REPO_ROOT, _SHARED_DIR, _CONFIG_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Shared utilities
from shared.agent_utils import (
    build_user_content,
    format_attached_documents,
    format_recent_messages,
    format_execution_mode,
    format_session_workspace,
    parse_tool_calls as _parse_tool_calls_raw,
    get_response_content,
    build_tool_calls_message,
    emit_progress,
)
from shared.agent_messages import (
    MAX_PLANNING_ITERATIONS_FALLBACK,
)
from shared.agent_loop import run_agent_loop
from shared.models import ToolCall
from shared.tools.schemas import ALL_TOOL_SCHEMAS as TOOL_SCHEMAS

from data_agent.llm_client import chat_completion, chat_completion_stream, create_client
from data_agent.models import AgentConfig, AgentResult, AgentState
from data_agent.prompts.simulated_results import build_simulated_result
from data_agent.prompts.system import PLAN_ONLY_ADDENDUM, SYSTEM_PROMPT


def _parse_tool_calls(response: Any) -> list[ToolCall]:
    """Extract ToolCall objects from an OpenAI ChatCompletion response."""
    return _parse_tool_calls_raw(response, ToolCall)


async def plan_only(
    query: str,
    config: AgentConfig | None = None,
    context: dict[str, Any] | None = None,
) -> AgentResult:
    """
    Plan-only mode: send the query to the LLM with tool schemas and capture
    the planned tool calls without executing any of them.

    For multi-step plans, the agent loops: after each batch of planned tool_calls,
    it feeds back a simulated "pending execution" result so the LLM can plan
    subsequent steps.

    Args:
        query: Natural language data retrieval question.
        config: Agent configuration. Uses defaults if not provided.
        context: Optional additional context to include in the prompt.

    Returns:
        AgentResult with planned_tool_calls and answer populated.
    """
    cfg = config or AgentConfig()
    state = AgentState(query=query, context=context or {})
    client = create_client(cfg)

    # Build initial messages -- include planning-mode instructions
    system_content = SYSTEM_PROMPT + PLAN_ONLY_ADDENDUM
    images: list[str] = []
    if context:
        page_context = context.get("page_context", "")
        if page_context:
            system_content += (
                f"\n\n=== PAGE CONTEXT ===\n"
                f"The user is currently viewing the following page:\n"
                f"{page_context}"
            )
        images = context.get("images", []) or []

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

        system_content += f"\n\n{format_execution_mode(config or context)}"

        # Inject bounded conversation context from recent_messages
        recent_msgs = context.get("recent_messages")
        if recent_msgs:
            formatted = format_recent_messages(recent_msgs)
            if formatted:
                system_content += (
                    f"\n\n=== CONVERSATION CONTEXT ===\n{formatted}"
                )

        ctx_for_prompt = {
            k: v for k, v in context.items()
            if k not in (
                "page_context", "images",
                "conversation_summary", "recent_messages",
                "parsed_documents", "attached_files",
            )
        }
        if ctx_for_prompt:
            system_content += f"\n\n=== ADDITIONAL CONTEXT ===\n{json.dumps(ctx_for_prompt)}"

    state.add_system_message(system_content)
    state.add_user_message(build_user_content(query, images))

    for iteration in range(cfg.max_iterations):
        state.iteration = iteration + 1

        # Ask the LLM for its next action
        response = await chat_completion(
            client=client,
            messages=state.messages,
            tools=TOOL_SCHEMAS,
            config=cfg,
        )

        # Parse tool calls from the response
        tool_calls = _parse_tool_calls(response)
        content = get_response_content(response)

        # If no tool calls, the LLM produced a final text response
        if not tool_calls:
            state.final_answer = content or ""
            state.status = "completed"
            break

        # Record the planned calls
        for tc in tool_calls:
            state.record_planned_call(tc)

        # Add assistant message with tool_calls to conversation history
        state.add_assistant_message(
            content=content,
            tool_calls=build_tool_calls_message(tool_calls),
        )

        # Feed back simulated results so the LLM can plan next steps
        for tc in tool_calls:
            simulated_result = json.dumps(
                build_simulated_result(tc),
                indent=2,
            )
            state.add_tool_result(tc.id, simulated_result)

    else:
        # Hit max iterations
        state.status = "max_iterations"
        state.final_answer = state.final_answer or (
            MAX_PLANNING_ITERATIONS_FALLBACK.format(n=len(state.planned_calls))
        )

    return state.to_result()


ProgressCallback = Any


def _data_progress_message(tc: ToolCall) -> str:
    """Build a human-readable progress message for data tool calls."""
    args = tc.arguments
    if tc.name in ("search_data", "facet_query", "probe_data"):
        coll = args.get("collection", "")
        if coll:
            return f"Querying BV-BRC {coll.replace('_', ' ')}..."
    elif tc.name == "list_collections":
        return "Listing available data collections..."
    elif tc.name == "get_collection_fields":
        return f"Looking up fields for {args.get('collection', 'collection')}..."
    elif tc.name == "get_sra_metadata":
        return "Fetching SRA metadata from NCBI..."
    elif tc.name == "create_group":
        gname = args.get("group_name", "group")
        return f"Creating group '{gname}'..."
    return f"Calling {tc.name}..."


def _data_result_message(tc: ToolCall, result: Any) -> str:
    """Build a progress message after a data tool completes."""
    msg = f"Tool {tc.name} completed."
    if isinstance(result, dict):
        nf = result.get("numFound") or result.get("count")
        if nf is not None:
            msg = f"Found {nf} records."
        elif result.get("error"):
            msg = "Query returned an error, adjusting approach..."
    return msg


async def run_agent(
    query: str,
    config: AgentConfig | None = None,
    context: dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AgentResult:
    """
    Full agent loop: plan, execute, evaluate, repeat.

    Args:
        query: Natural language data retrieval question.
        config: Agent configuration.
        context: Optional additional context.
        progress_callback: Optional async callback for progress updates.

    Returns:
        AgentResult with answer, execution trace, and sources.
    """
    cfg = config or AgentConfig()
    state = AgentState(query=query, context=context or {})
    client = create_client(cfg)

    await run_agent_loop(
        state=state,
        config=cfg,
        client=client,
        system_prompt=SYSTEM_PROMPT,
        chat_completion_fn=chat_completion,
        chat_completion_stream_fn=chat_completion_stream,
        progress_callback=progress_callback,
        progress_message_fn=_data_progress_message,
        result_message_fn=_data_result_message,
        start_message="Analyzing your question...",
        done_message="Data retrieval complete.",
    )

    return state.to_result()
