"""Phase 1: Decompose -- analyze user request and produce a WorkflowPlan.

Runs the LLM with Phase 1 tools (create_workflow_plan, list_services,
get_sra_metadata) to decompose the user's request into a structured
DAG of service steps.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Shared utilities -- deduplicated across all agents
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "shared"))
from agent_utils import (
    call_fingerprint,
    parse_tool_calls as _parse_tool_calls_raw,
    get_response_content,
    build_tool_calls_message,
    emit_progress,
)
from agent_messages import (
    DUPLICATE_CALL_WARNING_SERVICE,
    STUCK_IN_LOOP_DECOMPOSE,
)

from service_agent.llm_client import chat_completion, create_client
from service_agent.models import (
    AgentConfig,
    AgentState,
    ToolCall,
    WorkflowPlan,
)
from service_agent.prompts.phase1 import build_phase1_prompt
from service_agent.tool_registry import PHASE_1_TOOLS
from service_agent.tools import execute_tool, truncate_result

logger = logging.getLogger(__name__)


def _parse_tool_calls(response: Any) -> list[ToolCall]:
    """Extract ToolCall objects from an OpenAI ChatCompletion response."""
    return _parse_tool_calls_raw(response, ToolCall)


async def decompose(
    query: str,
    config: AgentConfig,
    state: AgentState,
    progress_callback=None,
) -> AgentState:
    """
    Phase 1: Decompose the user's request into a WorkflowPlan.

    Runs the LLM with Phase 1 tools. On success, sets state.workflow_plan.
    On needs_input, sets state.status and state.question.

    Args:
        query: The user's natural language request.
        config: Agent configuration.
        state: Current agent state (will be mutated).
        progress_callback: Optional async callback for progress updates.

    Returns:
        Updated AgentState with either:
          - workflow_plan set (success)
          - status="needs_input" and question set (needs user input)
          - status="error" (failure)
    """
    client = create_client(config)

    # Build Phase 1 messages
    system_prompt = build_phase1_prompt()
    if state.context:
        system_prompt += (
            f"\n\n=== ADDITIONAL CONTEXT ===\n"
            f"{json.dumps(state.context)}"
        )

    state.reset_messages()
    state.add_system_message(system_prompt)
    state.add_user_message(query)

    # Build auth headers if token is available
    headers: dict[str, str] | None = None
    if config.bvbrc_auth_token:
        headers = {"Authorization": config.bvbrc_auth_token}

    # Track duplicates
    executed_fingerprints: set[str] = set()
    duplicate_count = 0
    consecutive_failures = 0

    for iteration in range(config.max_iterations):
        # If stuck in a loop, force a text response
        if duplicate_count >= 3 or consecutive_failures >= 5:
            state.add_system_message(STUCK_IN_LOOP_DECOMPOSE)
            try:
                response = await chat_completion(
                    client=client,
                    messages=state.messages,
                    tools=PHASE_1_TOOLS,
                    config=config,
                    tool_choice="none",
                )
                content = get_response_content(response)
                if content:
                    state.status = "needs_input"
                    state.question = content
                else:
                    state.status = "error"
                    state.error_message = (
                        "Phase 1 failed: LLM could not produce a valid "
                        "workflow plan after multiple attempts."
                    )
            except Exception:
                state.status = "error"
                state.error_message = (
                    "Phase 1 failed after repeated failures."
                )
            return state

        response = await chat_completion(
            client=client,
            messages=state.messages,
            tools=PHASE_1_TOOLS,
            config=config,
        )

        tool_calls = _parse_tool_calls(response)
        content = get_response_content(response)

        # No tool calls -> LLM produced a text response
        if not tool_calls:
            if content:
                # The LLM is asking a question or explaining something
                state.status = "needs_input"
                state.question = content
            else:
                state.status = "error"
                state.error_message = (
                    "Phase 1 failed: LLM produced no tool calls and no text."
                )
            return state

        # Add assistant message with tool calls
        state.add_assistant_message(
            content=content,
            tool_calls=build_tool_calls_message(tool_calls),
        )

        # Execute each tool call
        iteration_had_failure = False
        for tc in tool_calls:
            _tool_msg = f"Decompose: Calling {tc.name}..."
            if tc.name == "list_services":
                _tool_msg = "Discovering available BV-BRC services..."
            elif tc.name == "create_workflow_plan":
                _tool_msg = "Validating workflow plan structure..."
            elif tc.name == "submit_workflow":
                _tool_msg = "Submitting workflow for execution..."
            elif tc.name == "get_sra_metadata":
                _tool_msg = "Retrieving SRA metadata..."
            await emit_progress(progress_callback, iteration, config.max_iterations, _tool_msg)
            fp = call_fingerprint(tc)

            if fp in executed_fingerprints:
                duplicate_count += 1
                state.add_tool_result(
                    tc.id,
                    json.dumps({
                        "_duplicate": True,
                        "_message": DUPLICATE_CALL_WARNING_SERVICE,
                    }),
                )
                continue

            start = time.time()
            result = await execute_tool(
                tool_name=tc.name,
                arguments=dict(tc.arguments),
                timeout_seconds=config.tool_timeout_seconds,
                config=config,
                headers=headers,
            )
            duration_ms = (time.time() - start) * 1000

            executed_fingerprints.add(fp)
            error = result.get("error") if isinstance(result, dict) else None
            state.record_execution(
                tc=tc,
                result=result,
                error=error,
                duration_ms=duration_ms,
                iteration=iteration,
            )

            # Track failures
            if error or (isinstance(result, dict) and result.get("status") == "invalid"):
                iteration_had_failure = True

            # Check if this was create_workflow_plan and it succeeded
            if (
                tc.name == "create_workflow_plan"
                and isinstance(result, dict)
                and result.get("status") == "valid"
            ):
                # Extract the validated plan
                plan_data = result["plan"]
                plan = WorkflowPlan(**plan_data)
                state.workflow_plan = plan
                state.current_phase = "build"
                state.status = "in_progress"
                await emit_progress(
                    progress_callback, iteration + 1, config.max_iterations,
                    f"Workflow plan created with {len(plan.steps)} step(s).",
                )

                # Feed success back to LLM conversation for clean exit
                result_str = truncate_result(result)
                state.add_tool_result(tc.id, result_str)
                return state

            # Check if this was submit_workflow — short-circuit the
            # 3-phase pipeline and return the submission result directly.
            if tc.name == "submit_workflow" and isinstance(result, dict):
                result_str = truncate_result(result)
                state.add_tool_result(tc.id, result_str)

                if result.get("error"):
                    state.status = "error"
                    state.error_message = result["error"]
                else:
                    state.status = "completed"
                    state.workflow_id = result.get("workflow_id")
                    state.current_phase = "done"
                return state

            # Feed result back for LLM to continue
            result_str = truncate_result(result)
            state.add_tool_result(tc.id, result_str)

        if iteration_had_failure:
            consecutive_failures += 1
        else:
            consecutive_failures = 0

    # Max iterations reached
    state.status = "error"
    state.error_message = (
        f"Phase 1 (Decompose) reached maximum iterations ({config.max_iterations}) "
        f"without producing a valid workflow plan."
    )
    return state
