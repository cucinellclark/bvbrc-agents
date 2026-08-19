"""
Core agent loop for the BV-BRC Planning Agent.

Operates in three modes based on context:
  - _analyze_and_plan(): Analyze user query, optionally ask clarification
    questions, then create a plan. (Default mode)
  - _plan_with_answers(): Create a plan using clarification answers
    provided by the user. (Triggered by plan_action="answer_questions")
  - _execute_step(): Formulate a focused query for the next step's
    target agent. (Triggered by plan_action="execute_next")
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Shared utilities -- deduplicated across all agents
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))
# Also add repo root so `shared` package imports work (shared.tools, shared.prompts)
if str(Path(__file__).resolve().parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from agent_utils import (  # noqa: E402
    build_user_content,
    call_fingerprint,
    parse_tool_calls as _parse_tool_calls_raw,
    get_response_content,
    build_tool_calls_message,
    emit_progress,
)
from agent_messages import (  # noqa: E402
    DUPLICATE_CALL_WARNING,
    MAX_ITERATIONS_SYNTHESIS,
    MAX_ITERATIONS_FALLBACK,
)

from planning_agent.llm_client import (  # noqa: E402
    chat_completion,
    chat_completion_stream,
    create_client,
)
from planning_agent.models import (  # noqa: E402
    AgentConfig,
    AgentResult,
    AgentState,
    ClarificationQuestion,
    Plan,
    PlanStep,
    ToolCall,
)
from planning_agent.prompts.system import build_system_prompt  # noqa: E402
from planning_agent.prompts.step_query import (  # noqa: E402
    build_step_execution_prompt,
)
from shared.tools.schemas import ALL_TOOL_SCHEMAS as TOOL_SCHEMAS  # noqa: E402

logger = logging.getLogger(__name__)

ProgressCallback = Any


def _parse_tool_calls(response: Any) -> list[ToolCall]:
    """Extract ToolCall objects from an OpenAI ChatCompletion response."""
    return _parse_tool_calls_raw(response, ToolCall)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_agent(
    query: str,
    config: AgentConfig | None = None,
    context: dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AgentResult:
    """Main entry point for the planning agent.

    Dispatches to one of three modes based on the context:
    - Default: analyze and plan (or ask clarification questions)
    - answer_questions: plan with user's clarification answers
    - execute_next: formulate query for the next step's target agent

    Args:
        query: User's natural language request.
        config: Agent configuration.
        context: Additional context (conversation history, plan state, etc.)
        progress_callback: Async progress callback.

    Returns:
        AgentResult with plan, clarification questions, or step execution info.
    """
    cfg = config or AgentConfig()
    ctx = context or {}

    # Extract plan_action from workflow_context (set by gateway)
    workflow_ctx = ctx.get("workflow_context", {})
    plan_action = workflow_ctx.get("plan_action", "")

    if plan_action == "execute_next":
        return await _execute_step(query, cfg, ctx, progress_callback)
    elif plan_action == "answer_questions":
        return await _plan_with_answers(query, cfg, ctx, progress_callback)
    elif plan_action == "continue_review":
        return await _continue_review(query, cfg, ctx, progress_callback)
    else:
        return await _analyze_and_plan(query, cfg, ctx, progress_callback)


# ---------------------------------------------------------------------------
# Mode 1: Analyze and Plan (default)
# ---------------------------------------------------------------------------


async def _analyze_and_plan(
    query: str,
    config: AgentConfig,
    context: dict[str, Any],
    progress_callback: ProgressCallback | None,
) -> AgentResult:
    """Analyze the user's query and either ask questions or create a plan.

    The LLM decides whether to:
    1. Call ask_clarification -> returns needs_input with questions
    2. Call create_plan -> returns needs_approval with plan
    3. Respond with text -> returns completed with direct answer
    """
    state = AgentState(query=query, context=context)
    client = create_client(config)

    # Build system prompt with agent catalog from context
    agent_catalog_text = _build_agent_catalog_text(context)
    system_prompt = build_system_prompt(agent_catalog_text)

    # Add page context if available
    images: list[str] = []
    if context:
        page_context = context.get("page_context", "")
        if page_context:
            system_prompt += (
                f"\n\n=== PAGE CONTEXT ===\n"
                f"The user is currently viewing the following page:\n"
                f"{page_context}"
            )
        images = context.get("images", []) or []

    # Add conversation context if available
    if context:
        conv_summary = context.get("conversation_summary", "")
        if conv_summary:
            system_prompt += f"\n\n=== CONVERSATION CONTEXT ===\n{conv_summary}"

    state.add_system_message(system_prompt)
    state.add_user_message(build_user_content(query, images))

    await emit_progress(progress_callback, 0, None, "Analyzing your request...")

    return await _run_planning_loop(state, client, config, progress_callback)


# ---------------------------------------------------------------------------
# Mode 2: Plan with Answers
# ---------------------------------------------------------------------------


async def _plan_with_answers(
    query: str,
    config: AgentConfig,
    context: dict[str, Any],
    progress_callback: ProgressCallback | None,
) -> AgentResult:
    """Create a plan using the user's clarification answers.

    The clarification answers are prepended to the conversation so the
    LLM has the extra context it asked for.
    """
    state = AgentState(query=query, context=context)
    client = create_client(config)

    # Build system prompt
    agent_catalog_text = _build_agent_catalog_text(context)
    system_prompt = build_system_prompt(agent_catalog_text)

    images: list[str] = []
    if context:
        page_context = context.get("page_context", "")
        if page_context:
            system_prompt += (
                f"\n\n=== PAGE CONTEXT ===\n"
                f"The user is currently viewing the following page:\n"
                f"{page_context}"
            )
        conv_summary = context.get("conversation_summary", "")
        if conv_summary:
            system_prompt += f"\n\n=== CONVERSATION CONTEXT ===\n{conv_summary}"
        images = context.get("images", []) or []

    state.add_system_message(system_prompt)

    # Add the original query — prefer the explicit original_query from
    # workflow_context over the generic placeholder that the gateway
    # may have set (e.g. "Submitted clarification responses.").
    workflow_ctx = context.get("workflow_context", {})
    original_query = workflow_ctx.get("original_query", "") or query
    state.add_user_message(build_user_content(original_query, images))

    # Add clarification answers as assistant/user exchange
    answers = workflow_ctx.get("clarification_answers", [])
    if answers:
        # Format answers as a user message with the Q&A
        answer_lines = []
        for ans in answers:
            q = ans.get("question", "")
            a = ans.get("answer", "")
            if q and a:
                answer_lines.append(f"Q: {q}\nA: {a}")

        if answer_lines:
            answers_text = "Here are my answers to your questions:\n\n" + "\n\n".join(
                answer_lines
            )
            # Add as if the assistant asked and user answered
            state.add_assistant_message(
                "I had some clarification questions about your request."
            )
            state.add_user_message(answers_text)

    await emit_progress(
        progress_callback, 0, None, "Creating plan with your clarifications..."
    )

    return await _run_planning_loop(state, client, config, progress_callback)


# ---------------------------------------------------------------------------
# Mode 3: Execute Step
# ---------------------------------------------------------------------------


async def _execute_step(
    query: str,
    config: AgentConfig,
    context: dict[str, Any],
    progress_callback: ProgressCallback | None,
) -> AgentResult:
    """Formulate a focused query for the next step's target agent.

    Reads the plan and step index from workflow_context, uses the LLM
    to combine the step description with prior results into a
    self-contained query for the target agent.

    For 'direct' agent steps, the planning agent answers the step itself.
    """
    workflow_ctx = context.get("workflow_context", {})
    plan_data = workflow_ctx.get("plan", {})
    current_index = workflow_ctx.get("current_step_index", 0)
    completed_results = workflow_ctx.get("completed_step_results", {})

    # Extract plan info
    steps = plan_data.get("steps", [])
    if current_index >= len(steps):
        return AgentResult(
            answer="All steps in the plan have been completed.",
            status="completed",
        )

    step = steps[current_index]
    step_id = step.get("step_id", f"step_{current_index}")
    step_description = step.get("description", "")
    target_agent = step.get("agent", "data")
    step_reasoning = step.get("reasoning", "")

    await emit_progress(
        progress_callback,
        current_index,
        len(steps),
        f"Preparing step {current_index + 1}: {step_description[:60]}...",
    )

    # For 'direct' steps, the planning agent answers itself
    if target_agent == "direct":
        return await _execute_direct_step(
            config,
            context,
            plan_data,
            step,
            current_index,
            completed_results,
            progress_callback,
        )

    # For 'review' steps, gather source data and return for user review
    if target_agent == "review":
        return await _execute_review_step(
            config,
            context,
            plan_data,
            step,
            current_index,
            completed_results,
            progress_callback,
        )

    # For agent-delegated steps, formulate the query
    client = create_client(config)

    # Gather original query from plan description or conversation context
    original_query = plan_data.get("description", query)
    conv_summary = context.get("conversation_summary", "")
    if conv_summary and not original_query:
        original_query = conv_summary[:500]

    # Build the step execution prompt
    step_prompt = build_step_execution_prompt(
        plan_title=plan_data.get("title", ""),
        plan_description=plan_data.get("description", ""),
        original_query=original_query,
        step_number=current_index + 1,
        total_steps=len(steps),
        step_description=step_description,
        target_agent=target_agent,
        step_reasoning=step_reasoning,
        prior_results=completed_results if completed_results else None,
    )

    messages = [
        {"role": "system", "content": step_prompt},
        {
            "role": "user",
            "content": (
                f"Formulate the query for the {target_agent} agent to "
                f"execute this step: {step_description}"
            ),
        },
    ]

    # Use the LLM to formulate a focused query (no tools needed)
    response = await chat_completion(
        client=client,
        messages=messages,
        config=config,
        tool_choice="none",
    )

    formulated_query = get_response_content(response) or step_description

    await emit_progress(
        progress_callback,
        current_index + 1,
        len(steps),
        f"Delegating to {target_agent} agent...",
    )

    # Return step_ready status with the formulated query
    state = AgentState(query=query, context=context)
    state.status = "step_ready"
    state.final_answer = f"Executing step {current_index + 1}: {step_description}"
    state.step_execution = {
        "agent": target_agent,
        "task": formulated_query,
        "plan_id": plan_data.get("plan_id", ""),
        "step_id": step_id,
        "step_index": current_index,
    }

    return state.to_result()


async def _execute_direct_step(
    config: AgentConfig,
    context: dict[str, Any],
    plan_data: dict[str, Any],
    step: dict[str, Any],
    step_index: int,
    completed_results: dict[str, dict],
    progress_callback: ProgressCallback | None,
) -> AgentResult:
    """Execute a 'direct' step — the planning agent answers it itself.

    Uses the LLM with context from prior steps to answer the step's
    question directly.
    """
    client = create_client(config)

    step_description = step.get("description", "")
    step_id = step.get("step_id", f"step_{step_index}")

    # Build context from prior results
    prior_context = ""
    if completed_results:
        lines = []
        for sid, result in completed_results.items():
            summary = result.get("answer", result.get("result_summary", ""))
            if summary:
                if len(summary) > 1500:
                    summary = summary[:1500] + "... [truncated]"
                lines.append(f"## {sid}\n{summary}")
        if lines:
            prior_context = "\n\n".join(lines)

    system_prompt = (
        "You are completing a step in a multi-step analysis plan. "
        "Use the information from prior steps to answer the current step.\n\n"
        f"Plan: {plan_data.get('title', '')}\n"
        f"Description: {plan_data.get('description', '')}\n"
    )

    if prior_context:
        system_prompt += f"\n=== RESULTS FROM PRIOR STEPS ===\n{prior_context}\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": step_description},
    ]

    await emit_progress(
        progress_callback,
        step_index,
        len(plan_data.get("steps", [])),
        f"Working on: {step_description[:60]}...",
    )

    # Stream the response for better UX
    answer = ""
    async for chunk in chat_completion_stream(
        client=client,
        messages=messages,
        config=config,
        tool_choice="none",
    ):
        answer += chunk

    state = AgentState(query=step_description, context=context)
    state.status = "completed"
    state.final_answer = answer

    # For direct steps, we return completed (not step_ready) because
    # we already have the answer — no need to delegate to another agent.
    # But we still need the orchestrator to know this was a plan step.
    state.step_execution = {
        "agent": "direct",
        "task": step_description,
        "plan_id": plan_data.get("plan_id", ""),
        "step_id": step_id,
        "step_index": step_index,
        "direct_answer": answer,
    }

    return state.to_result()


async def _execute_review_step(
    config: AgentConfig,
    context: dict[str, Any],
    plan_data: dict[str, Any],
    step: dict[str, Any],
    step_index: int,
    completed_results: dict[str, dict],
    progress_callback: ProgressCallback | None,
) -> AgentResult:
    """Execute a 'review' step -- gather source data for user review.

    Looks up the data_source_step in completed_results, extracts
    structured data, and returns step_ready with agent="review" so
    the orchestrator can emit a PLAN_REVIEW_READY event.
    """
    step_id = step.get("step_id", f"step_{step_index}")
    step_description = step.get("description", "")
    review_config = step.get("review_config", {})
    data_source_step = review_config.get("data_source_step", "")

    await emit_progress(
        progress_callback,
        step_index,
        len(plan_data.get("steps", [])),
        f"Preparing review: {step_description[:60]}...",
    )

    # Gather source data from the referenced prior step
    source_data: dict[str, Any] = {}
    if data_source_step and data_source_step in completed_results:
        source_result = completed_results[data_source_step]
        source_data = {
            "answer": source_result.get("answer", ""),
            "structured_data": source_result.get("structured_data"),
            "status": source_result.get("status", ""),
        }
    elif completed_results:
        # Fallback: use the most recent completed step's data
        last_key = list(completed_results.keys())[-1]
        source_result = completed_results[last_key]
        source_data = {
            "answer": source_result.get("answer", ""),
            "structured_data": source_result.get("structured_data"),
            "status": source_result.get("status", ""),
        }

    state = AgentState(query=step_description, context=context)
    state.status = "step_ready"
    # No visible chat message -- the frontend PlanCard renders an
    # interactive review panel via the PLAN_REVIEW_READY SSE event.
    state.final_answer = ""
    state.step_execution = {
        "agent": "review",
        "task": step_description,
        "plan_id": plan_data.get("plan_id", ""),
        "step_id": step_id,
        "step_index": step_index,
        "review_config": review_config,
        "source_data": source_data,
    }

    return state.to_result()


# ---------------------------------------------------------------------------
# Mode 4: Continue after Review
# ---------------------------------------------------------------------------


async def _continue_review(
    query: str,
    config: AgentConfig,
    context: dict[str, Any],
    progress_callback: ProgressCallback | None,
) -> AgentResult:
    """Complete a review step with the user's selections.

    The user's review decisions (selected IDs, chosen workflow, parameters)
    are stored as this step's result, which downstream steps can reference
    via completed_step_results.
    """
    workflow_ctx = context.get("workflow_context", {})
    plan_data = workflow_ctx.get("plan", {})
    current_index = workflow_ctx.get("current_step_index", 0)
    review_selections = workflow_ctx.get("review_selections", {})

    steps = plan_data.get("steps", [])
    step = steps[current_index] if current_index < len(steps) else {}
    step_id = step.get("step_id", f"step_{current_index}")
    step_description = step.get("description", "Review completed")

    await emit_progress(
        progress_callback,
        current_index,
        len(steps),
        "Processing your review selections...",
    )

    state = AgentState(query=query, context=context)
    state.status = "completed"

    # Build a summary of the user's selections for downstream context
    selection_parts: list[str] = []
    if review_selections.get("selected_ids"):
        ids = review_selections["selected_ids"]
        selection_parts.append(f"Selected {len(ids)} items")
    if review_selections.get("chosen_workflow"):
        selection_parts.append(
            f"Chose workflow: {review_selections['chosen_workflow']}"
        )
    if review_selections.get("filters"):
        selection_parts.append(
            f"Applied filters: {json.dumps(review_selections['filters'])}"
        )
    # Group management selections
    if review_selections.get("group_path"):
        selection_parts.append(f"Genome group path: {review_selections['group_path']}")
    if review_selections.get("group_name"):
        selection_parts.append(f"Group name: {review_selections['group_name']}")
    if review_selections.get("group_type"):
        selection_parts.append(f"Group type: {review_selections['group_type']}")
    if review_selections.get("group_action"):
        selection_parts.append(f"Action: {review_selections['group_action']}")
    selection_summary = (
        ". ".join(selection_parts) if selection_parts else "Review completed"
    )

    # No visible chat message -- the PlanCard already shows the review
    # step as completed with the selection summary in its result row.
    state.final_answer = ""
    state.step_execution = {
        "agent": "direct",
        "task": step_description,
        "plan_id": plan_data.get("plan_id", ""),
        "step_id": step_id,
        "step_index": current_index,
        "direct_answer": selection_summary,
        "review_selections": review_selections,
    }

    return state.to_result()


# ---------------------------------------------------------------------------
# Shared planning loop (used by Mode 1 and Mode 2)
# ---------------------------------------------------------------------------


async def _run_planning_loop(
    state: AgentState,
    client: Any,
    config: AgentConfig,
    progress_callback: ProgressCallback | None,
) -> AgentResult:
    """Run the LLM loop for plan creation.

    The LLM can:
    - Call ask_clarification -> sets needs_input status
    - Call create_plan -> sets needs_approval status
    - Respond with text -> sets completed status (simple request)
    """
    from shared.tools import execute_tool
    from shared.tools.registry import TOOL_DISPATCH

    executed_fingerprints: set[str] = set()
    duplicate_count = 0

    for iteration in range(config.max_iterations):
        state.iteration = iteration + 1

        await emit_progress(
            progress_callback,
            iteration,
            config.max_iterations,
            "Thinking about your request...",
        )

        # Ask the LLM what to do
        response = await chat_completion(
            client=client,
            messages=state.messages,
            tools=TOOL_SCHEMAS,
            config=config,
        )

        tool_calls = _parse_tool_calls(response)
        content = get_response_content(response)

        # No tool calls -> the LLM produced a text response
        if not tool_calls:
            state.final_answer = content or ""
            state.status = "completed"
            break

        # Add assistant message with tool_calls
        state.add_assistant_message(
            content=content,
            tool_calls=build_tool_calls_message(tool_calls),
        )

        # Execute each tool call
        for tc in tool_calls:
            fp = call_fingerprint(tc)

            # Duplicate detection
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

            # Build headers for shared tools that need auth
            _headers = (
                {"Authorization": config.bvbrc_auth_token}
                if config.bvbrc_auth_token
                else None
            )
            start = time.time()
            result = await execute_tool(
                tool_name=tc.name,
                arguments=dict(tc.arguments),
                dispatch_table=TOOL_DISPATCH,
                timeout_seconds=config.tool_timeout_seconds,
                config=config,
                headers=_headers,
            )
            duration_ms = (time.time() - start) * 1000

            executed_fingerprints.add(fp)
            error = result.get("error") if isinstance(result, dict) else None
            state.record_execution(
                tc=tc,
                result=result,
                error=error,
                duration_ms=duration_ms,
            )

            # Handle special tool results
            if tc.name == "ask_clarification" and result.get("status") == "valid":
                # Successfully validated questions — return them
                state.clarification_questions = [
                    ClarificationQuestion(**q) for q in result["questions"]
                ]
                state.status = "needs_input"
                state.final_answer = "I have some questions before I can create a plan."
                await emit_progress(
                    progress_callback,
                    config.max_iterations,
                    config.max_iterations,
                    "Waiting for your input...",
                )
                return state.to_result()

            elif tc.name == "create_plan" and result.get("status") == "valid":
                # Successfully validated plan — return it
                plan_dict = result["plan"]
                state.plan = Plan(
                    plan_id=plan_dict["plan_id"],
                    title=plan_dict["title"],
                    description=plan_dict["description"],
                    steps=[PlanStep(**s) for s in plan_dict["steps"]],
                    status="draft",
                    current_step_index=0,
                )
                state.status = "needs_approval"
                state.final_answer = (
                    f"I've created a plan: {plan_dict['title']}. "
                    f"It has {len(plan_dict['steps'])} steps. "
                    f"Please review and approve it."
                )
                await emit_progress(
                    progress_callback,
                    config.max_iterations,
                    config.max_iterations,
                    "Plan created — waiting for your review.",
                )
                return state.to_result()

            # For all tools, feed the result back to the LLM (truncated)
            from shared.tools import truncate_result

            result_str = (
                truncate_result(result, max_chars=8000)
                if isinstance(result, dict)
                else json.dumps(result, default=str)
            )
            state.add_tool_result(tc.id, result_str)

    else:
        # Max iterations reached
        state.status = "max_iterations"
        if not state.final_answer:
            state.final_answer = (
                "I was unable to create a plan within the allowed iterations. "
                "Please try rephrasing your request or breaking it down "
                "into simpler steps."
            )

    await emit_progress(
        progress_callback,
        config.max_iterations,
        config.max_iterations,
        "Planning complete.",
    )
    return state.to_result()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_agent_catalog_text(context: dict[str, Any]) -> str:
    """Build agent catalog text from context for the system prompt.

    The orchestrator may pass agent catalog info in the context.
    If not available, returns empty string (prompt will use fallback).
    """
    # Check if the orchestrator passed agent catalog info
    agent_catalog = context.get("agent_catalog", [])
    if not agent_catalog:
        return ""

    lines = []
    for agent in agent_catalog:
        name = agent.get("name", agent.get("key", "unknown"))
        key = agent.get("key", "")
        desc = agent.get("description", "")
        caps = agent.get("capabilities", [])
        lines.append(f"### {name} (key: `{key}`)")
        if desc:
            lines.append(f"{desc}")
        if caps:
            lines.append(f"Capabilities: {', '.join(caps)}")
        lines.append("")

    return "\n".join(lines)
