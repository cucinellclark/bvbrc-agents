"""
Core agent loop for the BV-BRC Helpdesk Agent.

Full plan-execute-evaluate loop:
  - run_agent(): Sends the query to the LLM with tool schemas. The LLM
    can query the helpdesk RAG, list services, and look up service schemas.
    The LLM reasons over raw documents and produces a grounded answer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Shared utilities -- deduplicated across all agents
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))
# Also add repo root so `shared` package imports work (shared.tools, shared.prompts)
if str(Path(__file__).resolve().parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
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
)

from helpdesk_agent.llm_client import (
    chat_completion,
    chat_completion_stream,
    create_client,
)
from helpdesk_agent.models import AgentConfig, AgentResult, AgentState, ToolCall
from helpdesk_agent.prompts.system import SYSTEM_PROMPT
from helpdesk_agent.tool_registry import TOOL_SCHEMAS


def _parse_tool_calls(response: Any) -> list[ToolCall]:
    """Extract ToolCall objects from an OpenAI ChatCompletion response."""
    return _parse_tool_calls_raw(response, ToolCall)


ProgressCallback = (
    Any  # async (progress: float, total: float|None, message: str) -> None
)


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
        query: Natural language question about using BV-BRC.
        config: Agent configuration.
        context: Optional additional context.
        progress_callback: Optional async callback for progress updates.

    Returns:
        AgentResult with answer, execution trace, and sources.
    """
    from helpdesk_agent.tools import execute_tool, truncate_result

    cfg = config or AgentConfig()
    state = AgentState(query=query, context=context or {})
    client = create_client(cfg)

    # Build initial messages
    system_content = SYSTEM_PROMPT
    if context:
        page_context = context.get("page_context", "")
        if page_context:
            system_content += (
                f"\n\n=== PAGE CONTEXT ===\n"
                f"The user is currently viewing the following page:\n"
                f"{page_context}"
            )
        ctx_for_prompt = {k: v for k, v in context.items() if k != "page_context"}
        if ctx_for_prompt:
            system_content += f"\n\n=== ADDITIONAL CONTEXT ===\n{json.dumps(ctx_for_prompt)}"

    state.add_system_message(system_content)
    state.add_user_message(query)

    await emit_progress(progress_callback, 0, None, "Analyzing your question...")

    # Track executed call fingerprints for duplicate detection
    executed_fingerprints: set[str] = set()
    duplicate_count = 0

    for iteration in range(cfg.max_iterations):
        state.iteration = iteration + 1

        await emit_progress(
            progress_callback,
            iteration,
            cfg.max_iterations,
            "Searching helpdesk knowledge base...",
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
            await emit_progress(
                progress_callback,
                iteration + 1,
                cfg.max_iterations,
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
        for tc in tool_calls:
            fp = call_fingerprint(tc)

            # --- Duplicate detection ---
            if fp in executed_fingerprints:
                duplicate_count += 1
                state.add_tool_result(
                    tc.id,
                    json.dumps(
                        {
                            "_duplicate": True,
                            "_message": DUPLICATE_CALL_WARNING,
                        }
                    ),
                )

                if duplicate_count >= 2:
                    break
                continue

            # Build a human-readable progress message
            _tool_msg = f"Calling {tc.name}..."
            if tc.name == "query_helpdesk":
                _q = tc.arguments.get("query", "")
                _tool_msg = f"Searching helpdesk: {_q[:60]}..."
            elif tc.name == "list_services":
                _tool_msg = "Listing available BV-BRC services..."
            elif tc.name == "get_service_schema":
                _svc = tc.arguments.get("service_name", "service")
                _tool_msg = f"Looking up {_svc} parameters..."
            await emit_progress(
                progress_callback,
                iteration,
                cfg.max_iterations,
                _tool_msg,
            )

            import time as _time

            start = _time.time()
            # Build headers for shared tools that need auth
            _headers = (
                {"Authorization": cfg.bvbrc_auth_token}
                if cfg.bvbrc_auth_token
                else None
            )
            result = await execute_tool(
                tool_name=tc.name,
                arguments=dict(tc.arguments),  # copy to avoid mutation
                timeout_seconds=cfg.tool_timeout_seconds,
                config=cfg,
                headers=_headers,
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
                _count = result.get("count")
                if _count is not None:
                    _result_msg = f"Found {_count} relevant documents."
                elif result.get("error"):
                    _result_msg = "Query returned an error, adjusting approach..."
            await emit_progress(
                progress_callback,
                iteration,
                cfg.max_iterations,
                _result_msg,
            )

            # Serialize and truncate for the LLM context
            result_str = truncate_result(result)
            state.add_tool_result(tc.id, result_str)

    else:
        # Hit max iterations without a final text response.
        # Force synthesis with tool_choice="none".
        state.status = "max_iterations"
        await emit_progress(
            progress_callback,
            cfg.max_iterations,
            cfg.max_iterations,
            f"Synthesizing answer from {len(state.tool_executions)} tool calls...",
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
                MAX_ITERATIONS_FALLBACK.format(n=len(state.tool_executions))
            )
        except Exception:
            state.final_answer = MAX_ITERATIONS_FALLBACK.format(
                n=len(state.tool_executions)
            )

    await emit_progress(
        progress_callback,
        cfg.max_iterations,
        cfg.max_iterations,
        "Helpdesk response complete.",
    )
    return state.to_result()
