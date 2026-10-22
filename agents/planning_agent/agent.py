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

import asyncio
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
    format_attached_documents,
    format_recent_messages,
    format_execution_mode,
    format_session_workspace,
    format_handoff,
    parse_tool_calls as _parse_tool_calls_raw,
    get_response_content,
    build_tool_calls_message,
    emit_progress,
)
from agent_messages import (  # noqa: E402
    DUPLICATE_CALL_WARNING,
    MAX_ITERATIONS_SYNTHESIS,
    MAX_ITERATIONS_FALLBACK,
    STUCK_ANSWER_NOW,
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
from shared.prompts.progress_skill import PROGRESS_SKILL_PROMPT
from shared.step_outputs import format_step_outputs  # noqa: E402
from shared.agent_utils import is_context_window_error  # noqa: E402
from shared.context_budget import ContextBudget  # noqa: E402
from shared import progress as prog  # noqa: E402

RECON_CAP_DEFAULT = 4
PLAN_TOOL_NAMES = frozenset({"create_plan", "ask_clarification"})
PLAN_ONLY_SCHEMAS = [s for s in TOOL_SCHEMAS if s["function"]["name"] in PLAN_TOOL_NAMES]
RECON_CAP_NUDGE = (
    "[SYSTEM CHECK — not from the user. Do not acknowledge or quote this note.] "
    "You have looked up enough to plan. Call create_plan now (or ask_clarification "
    "if a choice only the user can make is still open). Do not look anything else "
    "up: each step's agent finds the details it needs when the step runs."
)

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

        # Inject attached document excerpts (PDFs + text uploads)
        docs_section = format_attached_documents(
            context.get("parsed_documents")
        )
        if docs_section:
            system_prompt += f"\n\n{docs_section}"

        # Inject session workspace path for chat file discovery
        session_ws = format_session_workspace(context)
        if session_ws:
            system_prompt += f"\n\n{session_ws}"
        # Another agent passed this turn on (F1 handoff)
        handoff_section = format_handoff(context)
        if handoff_section:
            system_prompt += f"\n\n{handoff_section}"

        system_prompt += f"\n\n{format_execution_mode(config)}"
        system_prompt += f"\n\n{PROGRESS_SKILL_PROMPT}"

    # Add bounded conversation context from recent_messages
    if context:
        recent_msgs = context.get("recent_messages")
        if recent_msgs:
            formatted = format_recent_messages(recent_msgs)
            if formatted:
                system_prompt += f"\n\n=== CONVERSATION CONTEXT ===\n{formatted}"

        # Inject filtered context JSON dump (session_id, workspace_path, etc.)
        ctx_for_prompt = {
            k: v for k, v in context.items()
            if k not in {
                "page_context", "images",
                "conversation_summary", "recent_messages",
                "parsed_documents", "attached_files", "handoff", "llm_override", "llm_options",
            }
        }
        if ctx_for_prompt:
            system_prompt += (
                f"\n\n=== ADDITIONAL CONTEXT ===\n"
                f"{json.dumps(ctx_for_prompt, default=str)}"
            )

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
        # Inject attached document excerpts (PDFs + text uploads)
        docs_section = format_attached_documents(
            context.get("parsed_documents")
        )
        if docs_section:
            system_prompt += f"\n\n{docs_section}"

        # Inject session workspace path for chat file discovery
        session_ws = format_session_workspace(context)
        if session_ws:
            system_prompt += f"\n\n{session_ws}"
        # Another agent passed this turn on (F1 handoff)
        handoff_section = format_handoff(context)
        if handoff_section:
            system_prompt += f"\n\n{handoff_section}"

        system_prompt += f"\n\n{format_execution_mode(config)}"
        system_prompt += f"\n\n{PROGRESS_SKILL_PROMPT}"

        recent_msgs = context.get("recent_messages")
        if recent_msgs:
            formatted = format_recent_messages(recent_msgs)
            if formatted:
                system_prompt += f"\n\n=== CONVERSATION CONTEXT ===\n{formatted}"
        images = context.get("images", []) or []

        # Inject filtered context JSON dump (session_id, workspace_path, etc.)
        ctx_for_prompt = {
            k: v for k, v in context.items()
            if k not in {
                "page_context", "images",
                "conversation_summary", "recent_messages",
                "parsed_documents", "attached_files", "handoff", "llm_override", "llm_options",
            }
        }
        if ctx_for_prompt:
            system_prompt += (
                f"\n\n=== ADDITIONAL CONTEXT ===\n"
                f"{json.dumps(ctx_for_prompt, default=str)}"
            )

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

    await prog.note(
        f"Preparing step {current_index + 1} of {len(steps)}", step_description[:80], cb=progress_callback
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

    # Legacy 'review' steps are no longer supported — skip them
    if target_agent == "review":
        state = AgentState(query=query, context=context)
        state.status = "completed"
        state.final_answer = ""
        step_id = step.get("step_id", f"step_{current_index}")
        state.step_execution = {
            "agent": "direct",
            "task": step.get("description", "Skipped review step"),
            "plan_id": plan_data.get("plan_id", ""),
            "step_id": step_id,
            "step_index": current_index,
            "direct_answer": "Review step skipped — plans no longer use review checkpoints.",
        }
        return state.to_result()

    # For agent-delegated steps, formulate the query
    client = create_client(config)

    # Gather original query from plan description or conversation context
    original_query = plan_data.get("description", query)
    if not original_query:
        recent_msgs = context.get("recent_messages")
        if recent_msgs:
            original_query = format_recent_messages(recent_msgs, max_per_message=200, max_messages=3)

    # D4 Phase 3: only the steps this one depends on (transitively) feed it;
    # their structured outputs go first, verbatim.
    relevant = _relevant_prior_results(steps, current_index, completed_results)
    outputs_block = format_step_outputs({
        f"{sid}: {_step_description(steps, sid)[:80]}": (res or {}).get("step_outputs") or {}
        for sid, res in relevant.items()
    })

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
        prior_results=relevant or None,
        prior_outputs_block=outputs_block,
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
    # The block is also appended to the delegated agent's task, so ids and
    # paths reach it even if the formulated query paraphrased them away.
    if outputs_block:
        formulated_query = f"{formulated_query}\n\n{outputs_block}"

    await prog.note(f"Handing step {current_index + 1} to: {prog.agent_label(target_agent)}", cb=progress_callback)

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
        "expects": list(step.get("expects") or []),
        # Every group earlier steps made (create_group guard: a step may not
        # rebuild one of them under a new name).
        "prior_groups": _prior_groups(completed_results),
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

    await prog.note(
        f"Working on step {step_index + 1} of {len(plan_data.get('steps', []))}", step_description[:80], cb=progress_callback
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
    # D4 Phase 3 recon cap: after this many lookups the model may only plan
    # or ask.  The planner used to spend its whole 300 s on reconnaissance
    # (mshukla, Oct 9); the steps look things up themselves.
    recon_cap = max(0, int(getattr(config, "max_recon_tools", RECON_CAP_DEFAULT) or 0))
    recon_count = 0
    capped = False
    # Prompt budget (shared/context_budget.py, #12): past the hard line the
    # model may only plan or ask (the recon cap's state), and an overflow
    # compacts every result and retries once instead of failing the turn.
    budget = ContextBudget.for_config(config)
    overflow_recovered = False

    def _check_cancelled() -> None:
        """Raise CancelledError if the current asyncio task has been cancelled."""
        task = asyncio.current_task()
        if task is not None and task.cancelled():
            raise asyncio.CancelledError()

    for iteration in range(config.max_iterations):
        _check_cancelled()  # checkpoint 1: top of iteration

        if recon_cap and recon_count >= recon_cap and not capped:
            capped = True
            logger.info("Planning recon cap reached (%d lookups): plan or ask now", recon_count)
            state.add_user_message(RECON_CAP_NUDGE)

        state.iteration = iteration + 1

        prep = budget.prepare(state.messages, PLAN_ONLY_SCHEMAS if capped else TOOL_SCHEMAS)
        if prep["compacted"]:
            logger.info("Compacted %d older tool result(s); %s", prep["compacted"], budget.describe())
            await prog.note(prog.NOTE_COMPACTED, cb=progress_callback)
        if prep["exhausted"] and not capped:
            capped = True
            logger.warning("Context budget spent after %d lookup(s): plan or ask now (%s)",
                           recon_count, budget.describe())
            await prog.note(prog.NOTE_BUDGET_SPENT, cb=progress_callback)
            state.add_user_message(RECON_CAP_NUDGE)

        thinking = await prog.start(
            "thinking",
            "Working out the steps" if recon_count else "Reading your request and deciding what to look up",
            f"step {iteration + 1}",
            cb=progress_callback,
        )

        # Ask the LLM what to do
        try:
            response = await chat_completion(
                client=client,
                messages=state.messages,
                tools=PLAN_ONLY_SCHEMAS if capped else TOOL_SCHEMAS,
                config=config,
            )
        except Exception as exc:
            if not is_context_window_error(exc) or overflow_recovered:
                raise
            logger.warning("Context overflow after %d lookup(s); compacting and retrying once (%s)",
                           recon_count, budget.describe())
            await prog.done(thinking, status="error", result=prog.RESULT_TOO_LARGE, cb=progress_callback)
            await prog.note(prog.NOTE_OVERFLOW, cb=progress_callback)
            budget.compact_all(state.messages)
            overflow_recovered = True
            if not capped:
                capped = True
                state.add_user_message(RECON_CAP_NUDGE)
            continue
        budget.record(response, state.messages, PLAN_ONLY_SCHEMAS if capped else TOOL_SCHEMAS)

        _check_cancelled()  # checkpoint 2: after LLM call

        tool_calls = _parse_tool_calls(response)
        content = get_response_content(response)

        # No tool calls -> the LLM produced a text response
        if not tool_calls:
            await prog.done(thinking, label=prog.THINKING_ANSWER, cb=progress_callback)
            state.final_answer = content or ""
            state.status = "completed"
            break
        intent = prog.intent_line(content)
        await prog.done(thinking, cb=progress_callback)

        # Add assistant message with tool_calls
        state.add_assistant_message(
            content=content,
            tool_calls=build_tool_calls_message(tool_calls),
        )

        # Execute each tool call
        stuck = False
        for i, tc in enumerate(tool_calls):
            fp = call_fingerprint(tc)

            # Duplicate detection
            if fp in executed_fingerprints:
                duplicate_count += 1
                await prog.note(prog.NOTE_REPEAT_SKIPPED, cb=progress_callback)
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
                    for rest in tool_calls[i + 1:]:
                        state.add_tool_result(
                            rest.id, json.dumps({"_skipped": True, "_message": STUCK_ANSWER_NOW})
                        )
                    stuck = True
                    break
                continue

            _check_cancelled()  # checkpoint 3: before tool execution

            # Recon cap within one response: several lookups can arrive in a
            # single model reply (live, Oct 14: five), so the cap is applied
            # per call, not only between replies.
            if recon_cap and recon_count >= recon_cap and tc.name not in PLAN_TOOL_NAMES:
                state.add_tool_result(tc.id, json.dumps({"_skipped": True, "_message": RECON_CAP_NUDGE}))
                continue

            # Build headers for shared tools that need auth
            _headers = (
                {"Authorization": config.bvbrc_auth_token}
                if config.bvbrc_auth_token
                else None
            )
            tool_step = await prog.tool_start(tc.name, tc.arguments, intent, cb=progress_callback)
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
            if tc.name not in PLAN_TOOL_NAMES:
                recon_count += 1
            error = result.get("error") if isinstance(result, dict) else None
            state.record_execution(
                tc=tc,
                result=result,
                error=error,
                duration_ms=duration_ms,
            )
            await prog.tool_done(tool_step, tc.name, tc.arguments, result, cb=progress_callback)

            # F1 handoff (submit_gowe_job -> service, create_group -> data):
            # stop; the executor runs that agent next.
            from shared.tools import is_handoff_result

            if is_handoff_result(result):
                state.status = "handoff"
                state.final_answer = ""
                return state.to_result()

            # Handle special tool results
            if tc.name == "ask_clarification" and result.get("status") == "valid":
                # Successfully validated questions — return them
                state.clarification_questions = [
                    ClarificationQuestion(**q) for q in result["questions"]
                ]
                state.status = "needs_input"
                state.final_answer = "I have some questions before I can create a plan."
                await prog.note("Waiting for your answer", transient=True, cb=progress_callback)
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
                await prog.note("Plan ready, waiting for your review", transient=True, cb=progress_callback)
                return state.to_result()

            # For all tools, feed the result back to the LLM (truncated)
            from shared.tools import truncate_result, result_char_limit

            result_str = (
                truncate_result(result, max_chars=result_char_limit(tc.name, 8000))
                if isinstance(result, dict)
                else json.dumps(result, default=str)
            )
            state.add_tool_result(tc.id, result_str)

        if stuck:
            # The break above used to leave only this response's tool calls;
            # the next LLM call repeated the same call (see shared/agent_loop.py).
            from shared.agent_loop import force_answer

            logger.warning("Planning agent repeated a tool call; forcing an answer")
            await force_answer(state, client, config, chat_completion, TOOL_SCHEMAS)
            break

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


def _step_description(steps: list[dict[str, Any]], step_id: str) -> str:
    for s_ in steps:
        if s_.get("step_id") == step_id:
            return s_.get("description", "")
    return ""


def _prior_groups(completed_results: Any) -> list[dict[str, Any]]:
    """``[{name, path}]`` of the groups in any completed step's outputs."""
    groups: list[dict[str, Any]] = []
    for res in (completed_results or {}).values() if isinstance(completed_results, dict) else []:
        for g in ((res or {}).get("step_outputs") or {}).get("groups") or []:
            if isinstance(g, dict) and g.get("path") and all(g["path"] != x["path"] for x in groups):
                groups.append({"name": g.get("name"), "path": g["path"]})
    return groups


def _relevant_prior_results(
    steps: list[dict[str, Any]],
    current_index: int,
    completed: dict[str, Any] | None,
) -> dict[str, Any]:
    """Completed results of the steps the current step depends on, transitively.

    A step that declares no ``depends_on`` gets every earlier completed step:
    planners often leave the field empty on a step that plainly uses the one
    before it, and dropping its outputs would lose the id the chain needs.
    """
    completed = completed or {}
    if current_index >= len(steps):
        return {}
    by_id = {s_.get("step_id"): s_ for s_ in steps}
    direct = list(steps[current_index].get("depends_on") or [])
    if not direct:
        earlier = [s_.get("step_id") for s_ in steps[:current_index]]
        return {sid: completed[sid] for sid in earlier if sid in completed}
    wanted: list[str] = []
    stack = list(direct)
    while stack:
        sid = stack.pop()
        if sid in wanted or sid not in by_id:
            continue
        wanted.append(sid)
        stack.extend(by_id[sid].get("depends_on") or [])
    order = [s_.get("step_id") for s_ in steps]
    return {sid: completed[sid] for sid in order if sid in wanted and sid in completed}


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
