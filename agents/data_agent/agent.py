"""
Core agent loop for the BV-BRC Data Retrieval Agent.

Supports two modes:
  - plan_only(): Sends the query to the LLM with tool schemas and captures the
    planned tool_calls WITHOUT executing them. This is for testing/inspecting
    what the LLM would do.
  - run_agent(): Full plan-execute-evaluate loop (tool execution is stubbed for
    now; will be wired to real functions in Phase 1 build-out).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Shared utilities -- deduplicated across all agents
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))
from agent_utils import (
    call_fingerprint,
    parse_tool_calls as _parse_tool_calls_raw,
    get_response_content,
    build_tool_calls_message,
    emit_progress,
)
from agent_messages import (
    DUPLICATE_CALL_WARNING,
    MAX_ITERATIONS_SYNTHESIS,
    MAX_ITERATIONS_FALLBACK,
    MAX_PLANNING_ITERATIONS_FALLBACK,
)

from data_agent.llm_client import chat_completion, chat_completion_stream, create_client
from data_agent.models import AgentConfig, AgentResult, AgentState, ToolCall
from data_agent.prompts.simulated_results import build_simulated_result
from data_agent.prompts.system import PLAN_ONLY_ADDENDUM, SYSTEM_PROMPT
from data_agent.tool_registry import TOOL_SCHEMAS


def _parse_tool_calls(response: Any) -> list[ToolCall]:
    """Extract ToolCall objects from an OpenAI ChatCompletion response."""
    return _parse_tool_calls_raw(response, ToolCall)



# _build_simulated_result was moved to data_agent/prompts/simulated_results.py
# so the self-evolving prompt system can discover and target its LLM-facing
# _note strings. It is imported above as build_simulated_result.


async def plan_only(
    query: str,
    config: AgentConfig | None = None,
    context: dict[str, Any] | None = None,
) -> AgentResult:
    """
    Plan-only mode: send the query to the LLM with tool schemas and capture
    the planned tool calls without executing any of them.

    The LLM may produce a single batch of tool_calls, or it may produce a text
    response explaining its plan. Either way, the result captures what the LLM
    decided to do.

    For multi-step plans, the agent loops: after each batch of planned tool_calls,
    it feeds back a simulated "pending execution" result so the LLM can plan
    subsequent steps. This lets you see the full multi-step plan.

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
    if context:
        system_content += f"\n\n=== ADDITIONAL CONTEXT ===\n{json.dumps(context)}"

    state.add_system_message(system_content)
    state.add_user_message(query)

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

        # Feed back simulated results so the LLM can plan next steps.
        # Each tool_call requires a corresponding tool result message.
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


ProgressCallback = Any  # async (progress: float, total: float|None, message: str) -> None


async def run_agent(
    query: str,
    config: AgentConfig | None = None,
    context: dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AgentResult:
    """
    Full agent loop: plan, execute, evaluate, repeat.

    On each iteration the LLM either:
      - Returns tool_calls -> execute them, feed results back, loop
      - Returns text (no tool_calls) -> done, that's the final answer

    Args:
        query: Natural language data retrieval question.
        config: Agent configuration.
        context: Optional additional context.
        progress_callback: Optional async callback for progress updates.
            Signature: async (progress, total, message) -> None.

    Returns:
        AgentResult with answer, execution trace, and sources.
    """
    from data_agent.tools import execute_tool, truncate_result

    cfg = config or AgentConfig()
    state = AgentState(query=query, context=context or {})
    client = create_client(cfg)

    # Build initial messages (no plan-only addendum)
    system_content = SYSTEM_PROMPT
    if context:
        system_content += f"\n\n=== ADDITIONAL CONTEXT ===\n{json.dumps(context)}"

    state.add_system_message(system_content)
    state.add_user_message(query)

    await emit_progress(progress_callback, 0, None, "Analyzing your question...")

    # Build auth headers if token is available
    headers: dict[str, str] | None = None
    if cfg.bvbrc_auth_token:
        headers = {"Authorization": cfg.bvbrc_auth_token}

    # Track executed call fingerprints for duplicate detection
    executed_fingerprints: set[str] = set()
    duplicate_count = 0

    for iteration in range(cfg.max_iterations):
        state.iteration = iteration + 1

        await emit_progress(
            progress_callback, iteration, cfg.max_iterations,
            f"Planning next step ({state.iteration}/{cfg.max_iterations})...",
        )

        # 1. PLAN -- Ask the LLM what to do next
        response = await chat_completion(
            client=client,
            messages=state.messages,
            tools=TOOL_SCHEMAS,
            config=cfg,
        )

        tool_calls = _parse_tool_calls(response)
        content = get_response_content(response)

        # 2. CHECK -- If no tool calls, the LLM produced a final answer
        if not tool_calls:
            await emit_progress(progress_callback, iteration + 1, cfg.max_iterations, "Composing answer...")
            state.final_answer = content or ""
            state.status = "completed"
            break

        # Add assistant message with tool_calls to conversation
        state.add_assistant_message(
            content=content,
            tool_calls=build_tool_calls_message(tool_calls),
        )

        # 3. EXECUTE -- Run each tool call and feed results back
        for tc in tool_calls:
            fp = call_fingerprint(tc)

            # --- Duplicate detection ---
            if fp in executed_fingerprints:
                duplicate_count += 1
                state.add_tool_result(tc.id, json.dumps({"_duplicate": True, "_message": DUPLICATE_CALL_WARNING}))

                # If we've seen 2+ duplicates total, break the inner loop
                # to let the outer loop re-prompt the LLM
                if duplicate_count >= 2:
                    break
                continue

            # Build a human-readable progress message for this tool call
            _tool_msg = f"Calling {tc.name}..."
            _tc_args = dict(tc.arguments)
            if tc.name in ("search_data", "facet_query", "probe_data"):
                _coll = _tc_args.get("collection", "")
                if _coll:
                    _tool_msg = f"Querying BV-BRC {_coll.replace('_', ' ')}..."
            elif tc.name == "list_collections":
                _tool_msg = "Listing available data collections..."
            elif tc.name == "get_collection_fields":
                _tool_msg = f"Looking up fields for {_tc_args.get('collection', 'collection')}..."
            await emit_progress(progress_callback, iteration, cfg.max_iterations, _tool_msg)

            import time as _time

            start = _time.time()
            result = await execute_tool(
                tool_name=tc.name,
                arguments=dict(tc.arguments),  # copy to avoid mutation
                timeout_seconds=cfg.tool_timeout_seconds,
                base_url=cfg.bvbrc_api_url,
                headers=headers,
            )
            duration_ms = (_time.time() - start) * 1000

            # Record the execution and its fingerprint
            executed_fingerprints.add(fp)
            error = result.get("error") if isinstance(result, dict) else None
            state.record_execution(
                tc=tc,
                result=result,
                error=error,
                duration_ms=duration_ms,
            )

            # Emit result summary
            _result_msg = f"Tool {tc.name} completed."
            if isinstance(result, dict):
                _nf = result.get("numFound") or result.get("count")
                if _nf is not None:
                    _result_msg = f"Found {_nf} records."
                elif result.get("error"):
                    _result_msg = f"Query returned an error, adjusting approach..."
            await emit_progress(progress_callback, iteration, cfg.max_iterations, _result_msg)

            # Serialize and truncate for the LLM context
            result_str = truncate_result(result)
            state.add_tool_result(tc.id, result_str)

    else:
        # Hit max iterations without a final text response.
        # Make one final LLM call with tool_choice="none" to force a
        # synthesis from collected results.  Use streaming to reduce
        # time-to-first-token for remote endpoints (Argo).
        state.status = "max_iterations"
        await emit_progress(
            progress_callback, cfg.max_iterations, cfg.max_iterations,
            f"Synthesizing answer from {len(state.tool_calls_executed)} queries...",
        )
        try:
            state.add_system_message(MAX_ITERATIONS_SYNTHESIS)
            synthesis_content = ""
            async for chunk in chat_completion_stream(
                client=client,
                messages=state.messages,
                config=cfg,
                tool_choice="none",
            ):
                synthesis_content += chunk
            state.final_answer = synthesis_content or (
                MAX_ITERATIONS_FALLBACK.format(n=len(state.tool_calls_executed))
            )
        except Exception:
            # If the synthesis call fails, fall back to the generic message
            state.final_answer = (
                MAX_ITERATIONS_FALLBACK.format(n=len(state.tool_calls_executed))
            )

    await emit_progress(progress_callback, cfg.max_iterations, cfg.max_iterations, "Data retrieval complete.")
    return state.to_result()
