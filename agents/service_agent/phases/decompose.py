"""Phase 1: Decompose -- analyze user request and produce a WorkflowPlan.

Runs the LLM with Phase 1 tools (create_workflow_plan, list_services,
get_sra_metadata) to decompose the user's request into a structured
DAG of service steps.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

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


def _normalize_arguments(args: dict[str, Any]) -> dict[str, Any]:
    """Normalize LLM tool call arguments (string bools/ints -> proper types)."""
    normalized = {}
    for key, value in args.items():
        if isinstance(value, str):
            lower = value.lower()
            if lower == "true":
                normalized[key] = True
            elif lower == "false":
                normalized[key] = False
            elif lower.isdigit() or (lower.startswith("-") and lower[1:].isdigit()):
                normalized[key] = int(value)
            else:
                normalized[key] = value
        else:
            normalized[key] = value
    return normalized


def _parse_tool_calls(response: Any) -> list[ToolCall]:
    """Extract ToolCall objects from an OpenAI ChatCompletion response."""
    choice = response.choices[0]
    if not choice.message.tool_calls:
        return []

    calls = []
    for tc in choice.message.tool_calls:
        try:
            args = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                "Failed to parse tool call arguments as JSON for %s "
                "(strict mode should prevent this for most tools): %s",
                tc.function.name,
                tc.function.arguments[:200],
            )
            args = {"_raw": tc.function.arguments}

        args = _normalize_arguments(args)

        calls.append(
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=args,
            )
        )
    return calls


def _get_response_content(response: Any) -> str | None:
    """Extract text content from the response, if any."""
    choice = response.choices[0]
    return choice.message.content


def _build_tool_calls_message(tool_calls: list[ToolCall]) -> list[dict[str, Any]]:
    """Build the tool_calls list in OpenAI message format."""
    return [
        {
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.name,
                "arguments": json.dumps(tc.arguments),
            },
        }
        for tc in tool_calls
    ]


async def _emit(cb, progress: float, total: float | None, message: str) -> None:
    """Fire progress callback if provided, swallowing errors."""
    if cb is not None:
        try:
            await cb(progress, total, message)
        except Exception:
            pass


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
            state.add_system_message(
                "You are stuck in a loop. STOP calling tools. Instead, "
                "provide your plan as a text response, explaining what "
                "services are needed and their dependencies. If you need "
                "information from the user, ask a clear question."
            )
            try:
                response = await chat_completion(
                    client=client,
                    messages=state.messages,
                    tools=PHASE_1_TOOLS,
                    config=config,
                    tool_choice="none",
                )
                content = _get_response_content(response)
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
        content = _get_response_content(response)

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
            tool_calls=_build_tool_calls_message(tool_calls),
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
            await _emit(progress_callback, iteration, config.max_iterations, _tool_msg)
            fp = f"{tc.name}::{json.dumps(tc.arguments, sort_keys=True)}"

            if fp in executed_fingerprints:
                duplicate_count += 1
                state.add_tool_result(
                    tc.id,
                    json.dumps({
                        "_duplicate": True,
                        "_message": (
                            "DUPLICATE CALL DETECTED: You already called this "
                            "tool with these exact arguments. Use the results "
                            "you already have. Do NOT repeat this call."
                        ),
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
                await _emit(
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
