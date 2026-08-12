"""Workflow populate phase -- GoWe-first workflow execution.

Single LLM loop that:
1. Lists available workflows from GoWe
2. Selects the right one for the user's request
3. Gets its input schema
4. Populates inputs using workspace/data tools
5. Submits the job

Replaces the 3-phase (Decompose -> Build -> Compose) pipeline for
pre-registered GoWe workflows.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "shared"))
# Also add repo root so `shared` package imports work
if str(Path(__file__).resolve().parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from agent_utils import (
    call_fingerprint,
    parse_tool_calls as _parse_tool_calls_raw,
    get_response_content,
    build_tool_calls_message,
    emit_progress,
)
from agent_messages import (
    DUPLICATE_CALL_WARNING_SHORT,
)

from service_agent.llm_client import chat_completion, create_client
from service_agent.models import AgentConfig, AgentState, ToolCall
from service_agent.prompts.populate import build_populate_prompt
from service_agent.tool_registry import POPULATE_TOOLS
from service_agent.tools import execute_tool, truncate_result

logger = logging.getLogger(__name__)

STUCK_MESSAGE = (
    "You seem to be stuck in a loop.  Please provide a text response "
    "summarizing what you need from the user, or explain the problem."
)


def _parse_tool_calls(response: Any) -> list[ToolCall]:
    return _parse_tool_calls_raw(response, ToolCall)


async def populate_and_submit(
    query: str,
    config: AgentConfig,
    state: AgentState,
    progress_callback: Any | None = None,
) -> AgentState:
    """
    Run the GoWe-first workflow populate + submit flow.

    The LLM has access to GoWe discovery tools (list workflows, get inputs),
    workspace/data tools, and a submit tool. It handles the entire flow in
    a single loop.

    Args:
        query: User's natural language request.
        config: Agent configuration.
        state: Agent state (will be mutated).
        progress_callback: Optional progress callback.

    Returns:
        Updated AgentState with either:
          - status="completed", submission_id set (job submitted)
          - status="needs_input", question set (needs user clarification)
          - status="error" (failure)
    """
    state.current_phase = "populate"

    # Build system prompt
    attached_files = state.context.get("attached_files", []) if state.context else []
    system_prompt = build_populate_prompt(attached_files=attached_files)

    if state.context:
        system_prompt += f"\n\n=== ADDITIONAL CONTEXT ===\n{json.dumps(state.context)}"

    # Initialize messages
    state.reset_messages()
    state.add_system_message(system_prompt)
    state.add_user_message(query)

    client = create_client(config)

    # Build auth headers
    headers: dict[str, str] | None = None
    if config.bvbrc_auth_token:
        headers = {"Authorization": config.bvbrc_auth_token}

    # Track duplicates
    executed_fingerprints: set[str] = set()
    duplicate_count = 0

    for iteration in range(config.max_iterations):
        await emit_progress(
            progress_callback,
            iteration,
            config.max_iterations,
            f"Workflow planning step {iteration + 1}...",
        )

        # If stuck in a loop, force text response
        if duplicate_count >= 3:
            state.add_system_message(STUCK_MESSAGE)
            try:
                response = await chat_completion(
                    client=client,
                    messages=state.messages,
                    tools=POPULATE_TOOLS,
                    config=config,
                    tool_choice="none",
                )
                content = get_response_content(response)
                if content:
                    state.status = "needs_input"
                    state.question = content
                else:
                    state.status = "error"
                    state.error_message = "Stuck in a loop without resolution."
            except Exception:
                state.status = "error"
                state.error_message = "Failed after repeated attempts."
            return state

        response = await chat_completion(
            client=client,
            messages=state.messages,
            tools=POPULATE_TOOLS,
            config=config,
        )

        tool_calls = _parse_tool_calls(response)
        content = get_response_content(response)

        # No tool calls -> LLM produced text (question, answer, or done message)
        if not tool_calls:
            if content:
                # Check if the LLM is done (already submitted) or asking a question
                if state.submission_ids:
                    # One or more jobs submitted; this is the summary message
                    state.status = "completed"
                    state.current_phase = "done"
                    state.operation_message = content
                else:
                    state.status = "needs_input"
                    state.question = content
            else:
                state.status = "error"
                state.error_message = "LLM produced no tool calls and no text."
            return state

        # Add assistant message with tool calls
        state.add_assistant_message(
            content=content,
            tool_calls=build_tool_calls_message(tool_calls),
        )

        # Execute each tool call
        for tc in tool_calls:
            # Progress message
            _msgs = {
                "list_gowe_workflows": "Discovering available workflows...",
                "get_workflow_inputs": "Getting workflow input schema...",
                "submit_gowe_job": (
                    f"Submitting job {len(state.submission_ids) + 1} to GoWe..."
                    if state.submission_ids
                    else "Submitting job to GoWe..."
                ),
                "workspace_browse": "Browsing workspace for inputs...",
                "read_file_info": "Reading file metadata...",
                "search_data": "Querying BV-BRC data...",
                "get_sra_metadata": "Fetching SRA metadata...",
            }
            _msg = _msgs.get(tc.name, f"Calling {tc.name}...")
            await emit_progress(
                progress_callback,
                iteration,
                config.max_iterations,
                _msg,
            )

            fp = call_fingerprint(tc)

            if fp in executed_fingerprints:
                duplicate_count += 1
                state.add_tool_result(
                    tc.id,
                    json.dumps(
                        {
                            "_duplicate": True,
                            "_message": DUPLICATE_CALL_WARNING_SHORT,
                        }
                    ),
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

            # Check if this was a successful submission
            if (
                tc.name == "submit_gowe_job"
                and isinstance(result, dict)
                and "submission_id" in result
                and "error" not in result
            ):
                sub_id = result.get("submission_id")
                state.workflow_id = result.get("workflow_id")
                state.submission_id = sub_id  # backward compat: last submitted
                state.submission_ids.append(sub_id)
                state.auto_submitted = True
                state.persisted = True

                logger.info(
                    "Job submitted (%d so far): workflow_id=%s, submission_id=%s",
                    len(state.submission_ids),
                    state.workflow_id,
                    sub_id,
                )

                # Feed result back so LLM can continue (more samples) or summarize
                result_str = truncate_result(result)
                state.add_tool_result(tc.id, result_str)
                # Do NOT return here -- let the LLM continue the loop.
                # It may have more samples to submit. The loop exits when
                # the LLM produces a text response (no tool calls) after
                # all submissions are done.
                continue

            # Feed result back for LLM to continue
            result_str = truncate_result(result)
            state.add_tool_result(tc.id, result_str)

    # Max iterations reached
    state.status = "error"
    state.error_message = (
        f"Reached maximum iterations ({config.max_iterations}) "
        f"without completing the workflow."
    )
    return state
