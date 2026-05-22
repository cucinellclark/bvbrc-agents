"""Phase 2: Build Steps -- gather inputs and validate each step incrementally.

For each step in the WorkflowPlan (in topological order), runs a focused
LLM sub-loop to gather inputs, resolve parameters, and validate through
plan_service. Each step has access to upstream outputs from already-built steps.
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
    StepPlan,
    ToolCall,
    ValidatedStep,
)
from service_agent.prompts.phase2 import build_phase2_prompt
from service_agent.tool_registry import PHASE_2_TOOLS
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
                "(expected for non-strict tools like plan_service): %s",
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


def _extract_validated_step(
    step_plan: StepPlan,
    plan_service_result: dict[str, Any],
) -> ValidatedStep:
    """Extract a ValidatedStep from a successful plan_service result."""
    return ValidatedStep(
        step_id=step_plan.step_id,
        service_name=step_plan.service_name,
        api_name=plan_service_result.get("api_name", ""),
        params=plan_service_result.get("params", {}),
        output_patterns=plan_service_result.get("output_patterns", {}),
        depends_on=step_plan.depends_on,
        auto_corrections=plan_service_result.get("auto_corrections", []),
        warnings=plan_service_result.get("warnings", []),
    )


async def _emit(cb, progress: float, total: float | None, message: str) -> None:
    """Fire progress callback if provided, swallowing errors."""
    if cb is not None:
        try:
            await cb(progress, total, message)
        except Exception:
            pass


async def build_step(
    step_id: str,
    config: AgentConfig,
    state: AgentState,
    progress_callback=None,
) -> AgentState:
    """
    Phase 2: Build a single step of the workflow.

    Runs a focused LLM sub-loop for the given step, using Phase 2 tools
    to gather inputs and validate parameters via plan_service.

    Args:
        step_id: The step to build.
        config: Agent configuration.
        state: Current agent state (will be mutated).
        progress_callback: Optional async callback for progress updates.

    Returns:
        Updated AgentState with either:
          - completed_steps[step_id] set (success)
          - status="needs_input" and question set (needs user input)
          - status="error" (failure)
    """
    if not state.workflow_plan:
        state.status = "error"
        state.error_message = "Cannot build step: no workflow plan available."
        return state

    step_plan = state.workflow_plan.get_step(step_id)
    upstream_outputs = state.get_upstream_outputs(step_id)

    state.current_step_id = step_id

    # Build step-specific system prompt
    system_prompt = build_phase2_prompt(
        step_id=step_plan.step_id,
        service_name=step_plan.service_name,
        intent=step_plan.intent,
        depends_on=step_plan.depends_on,
        input_sources=step_plan.input_sources,
        upstream_outputs=upstream_outputs,
    )

    # Add context about the overall plan and user's original query
    plan_summary = (
        f"Overall workflow: {state.workflow_plan.workflow_name}\n"
        f"Description: {state.workflow_plan.description}\n"
        f"Steps: {', '.join(s.step_id for s in state.workflow_plan.steps)}\n"
        f"Build order: {' -> '.join(state.workflow_plan.topological_order)}"
    )

    if state.context:
        system_prompt += (
            f"\n\n=== ADDITIONAL CONTEXT ===\n"
            f"{json.dumps(state.context)}"
        )

    # Reset messages for this step's sub-loop
    state.reset_messages()
    state.add_system_message(system_prompt)

    # User message = the original query + plan context
    user_msg = (
        f"User's original request: {state.query}\n\n"
        f"Workflow plan context:\n{plan_summary}\n\n"
        f"Build step '{step_id}' now."
    )
    state.add_user_message(user_msg)

    client = create_client(config)

    # Build auth headers
    headers: dict[str, str] | None = None
    if config.bvbrc_auth_token:
        headers = {"Authorization": config.bvbrc_auth_token}

    # Track duplicates
    executed_fingerprints: set[str] = set()
    duplicate_count = 0

    for iteration in range(config.max_iterations):
        # If stuck in a loop, force a text response
        if duplicate_count >= 3:
            state.add_system_message(
                "You are stuck in a loop. STOP calling tools. If you need "
                "information from the user to complete this step, ask a "
                "clear question in your text response."
            )
            try:
                response = await chat_completion(
                    client=client,
                    messages=state.messages,
                    tools=PHASE_2_TOOLS,
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
                        f"Phase 2 failed for step '{step_id}': "
                        f"stuck in a loop."
                    )
            except Exception:
                state.status = "error"
                state.error_message = (
                    f"Phase 2 failed for step '{step_id}' after "
                    f"repeated failures."
                )
            return state

        response = await chat_completion(
            client=client,
            messages=state.messages,
            tools=PHASE_2_TOOLS,
            config=config,
        )

        tool_calls = _parse_tool_calls(response)
        content = _get_response_content(response)

        # No tool calls -> LLM produced a text response (question or error)
        if not tool_calls:
            if content:
                state.status = "needs_input"
                state.question = content
            else:
                state.status = "error"
                state.error_message = (
                    f"Phase 2 failed for step '{step_id}': "
                    f"LLM produced no tool calls and no text."
                )
            return state

        # Add assistant message with tool calls
        state.add_assistant_message(
            content=content,
            tool_calls=_build_tool_calls_message(tool_calls),
        )

        # Execute each tool call
        for tc in tool_calls:
            _tool_msg = f"Build '{step_id}': Calling {tc.name}..."
            if tc.name == "get_service_schema":
                _svc = tc.arguments.get("service_name", "")
                _tool_msg = f"Build '{step_id}': Fetching schema for '{_svc}'..."
            elif tc.name == "plan_service":
                _tool_msg = f"Build '{step_id}': Validating service parameters..."
            elif tc.name in ("workspace_browse", "read_file_info"):
                _tool_msg = f"Build '{step_id}': Browsing workspace for inputs..."
            elif tc.name in ("search_data", "get_genome_group", "get_feature_group"):
                _tool_msg = f"Build '{step_id}': Querying BV-BRC data..."
            await _emit(progress_callback, iteration, config.max_iterations, _tool_msg)

            fp = f"{tc.name}::{json.dumps(tc.arguments, sort_keys=True)}"

            if fp in executed_fingerprints:
                duplicate_count += 1
                state.add_tool_result(
                    tc.id,
                    json.dumps({
                        "_duplicate": True,
                        "_message": (
                            "DUPLICATE CALL DETECTED. Use the results you "
                            "already have. Do NOT repeat this call."
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

            # Check if this was a successful plan_service call
            if (
                tc.name == "plan_service"
                and isinstance(result, dict)
                and result.get("status") in ("valid", "planned")
                and "error" not in result
            ):
                # Extract validated step
                validated = _extract_validated_step(step_plan, result)
                state.mark_step_complete(step_id, validated)
                state.status = "in_progress"
                await _emit(progress_callback, iteration + 1, config.max_iterations,
                            f"Step '{step_id}' validated successfully.")

                # Feed success back
                result_str = truncate_result(result)
                state.add_tool_result(tc.id, result_str)
                return state

            # Feed result back for LLM to continue
            result_str = truncate_result(result)
            state.add_tool_result(tc.id, result_str)

    # Max iterations reached for this step
    state.status = "error"
    state.error_message = (
        f"Phase 2 reached maximum iterations ({config.max_iterations}) "
        f"for step '{step_id}' without producing a valid plan_service result."
    )
    return state
