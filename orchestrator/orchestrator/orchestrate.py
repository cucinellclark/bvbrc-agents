"""Core orchestration loop.

The main orchestrate() async generator wires together:
  route -> execute -> synthesize

It yields Event objects throughout, making the entire pipeline
streamable to the gateway.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from orchestrator.documents import prepare_attached_documents
from orchestrator.events.events import Event, EventType, error_event
from orchestrator.executor.executor import execute_plan
from orchestrator.llm.client import LLMClient
from orchestrator.models import OrchestratorRequest, OrchestratorResponse
from orchestrator.registry.agent_registry import AgentRegistry
from orchestrator.router.models import Plan, Step
from orchestrator.router.router import route
from orchestrator.synthesizer.synthesizer import synthesize

logger = logging.getLogger(__name__)


async def orchestrate(
    request: OrchestratorRequest,
    registry: AgentRegistry,
    llm: LLMClient,
    routing_llm: LLMClient | None = None,
) -> AsyncGenerator[Event, None]:
    """Main orchestration loop.

    1. Route: LLM decides which agent to invoke (or to respond directly)
    2. Execute: Run the agent via MCP through the registry
    3. Synthesize: Generate final response from agent results

    Yields Event objects at every stage for streaming.

    Args:
        request: The inbound request from the gateway.
        registry: Agent registry with discovered agents.
        llm: LLM client for synthesis (and routing if routing_llm is None).
        routing_llm: Optional faster LLM client dedicated to routing.
            Falls back to ``llm`` when not provided.

    Yields:
        Event objects throughout the orchestration pipeline.
        The final event is ORCHESTRATOR_DONE with the complete response.
    """
    start_time = time.monotonic()

    yield Event(
        type=EventType.ORCHESTRATOR_START,
        data={"query": request.query, "session_id": request.session_id},
    )

    try:
        # ------------------------------------------------------------------
        # 0. PREPROCESS — extract/persist attached documents before routing
        # ------------------------------------------------------------------
        if request.pdfs or request.attached_files or (
            request.selected_items
            and any(
                (item.get("path", "") if isinstance(item, dict) else "")
                .lower()
                .endswith(".pdf")
                for item in request.selected_items
            )
        ):
            await prepare_attached_documents(request)

        # ------------------------------------------------------------------
        # 1. ROUTE
        # ------------------------------------------------------------------
        yield Event(
            type=EventType.ROUTING_START,
            data={"query": request.query},
        )

        decision = await route(request, registry, routing_llm or llm)

        # Build routing decision event data
        routing_event_data: dict[str, Any] = {
            "decision": decision.decision,
            "confidence": decision.confidence,
            "reasoning": (decision.plan.reasoning if decision.plan else None),
        }
        if decision.plan and decision.plan.steps:
            if decision.decision == "pipeline":
                routing_event_data["steps"] = [
                    {
                        "agent_key": s.agent_key,
                        "task": s.task,
                        "depends_on": s.depends_on,
                    }
                    for s in decision.plan.steps
                ]
            else:
                routing_event_data["agent_key"] = decision.plan.steps[0].agent_key

        yield Event(
            type=EventType.ROUTING_DECISION,
            data=routing_event_data,
        )

        # ------------------------------------------------------------------
        # 2. Handle DIRECT response (no agent needed)
        # ------------------------------------------------------------------
        if decision.decision == "direct":
            response_text = decision.direct_response or ""
            logger.info(
                f"Direct response: {len(response_text)} chars, "
                f"preview={response_text[:80]!r}"
            )
            yield Event(
                type=EventType.ORCHESTRATOR_DONE,
                data={
                    "response_text": response_text,
                    "decision": "direct",
                    "elapsed_ms": _elapsed_ms(start_time),
                },
            )
            return

        # ------------------------------------------------------------------
        # 3. EXECUTE plan
        # ------------------------------------------------------------------
        if not decision.plan or not decision.plan.steps:
            yield error_event(
                message=f"Router returned a '{decision.decision}' decision but no plan.",
            )
            yield Event(
                type=EventType.ORCHESTRATOR_DONE,
                data={
                    "response_text": (
                        "I encountered an internal error while routing "
                        "your request. Please try again."
                    ),
                    "decision": "error",
                    "elapsed_ms": _elapsed_ms(start_time),
                },
            )
            return

        # When auto-submit is enabled, prune any pipeline steps that
        # are purely "submit the planned workflow" — those would call
        # agent_chat which re-runs the full planning pipeline and fails.
        # Auto-submit (§3b below) already handles submission directly.
        auto_submit = getattr(registry._config, "auto_submit", False)
        if auto_submit and len(decision.plan.steps) > 1:
            _SUBMIT_KEYWORDS = {"submit", "execute", "run the"}
            kept: list[Step] = []
            for step in decision.plan.steps:
                task_lower = step.task.lower()
                if (
                    any(kw in task_lower for kw in _SUBMIT_KEYWORDS)
                    and "assemble" not in task_lower
                    and "annotate" not in task_lower
                    and "build" not in task_lower
                    and "plan" not in task_lower
                    and "analyze" not in task_lower
                ):
                    logger.info(
                        "Pruning redundant submit step from pipeline "
                        "(auto-submit is enabled): %r",
                        step.task,
                    )
                else:
                    kept.append(step)
            if kept and len(kept) < len(decision.plan.steps):
                # Build mapping from old step indices to new indices
                old_to_new: dict[int, int] = {}
                new_idx = 0
                for old_idx, step in enumerate(decision.plan.steps):
                    if step in kept:
                        old_to_new[old_idx] = new_idx
                        new_idx += 1
                for s in kept:
                    s.depends_on = [
                        old_to_new[d] for d in s.depends_on if d in old_to_new
                    ]
                decision.plan.steps = kept

        # Collect agent results from execution events
        agent_results: list[dict[str, Any]] = []
        agents_used: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        async for event in execute_plan(decision.plan, registry, request):
            yield event

            # Capture results from agent_result events
            if event.type == EventType.AGENT_RESULT:
                result_for_ui = event.data.get("result_for_ui", {})
                agent_results.append(result_for_ui)
                agent_name = event.data.get("agent", event.agent_name or "unknown")
                if agent_name not in agents_used:
                    agents_used.append(agent_name)

            elif event.type == EventType.AGENT_TOOL_CALL:
                tool_calls.append(event.data)

            elif event.type == EventType.ORCHESTRATOR_ERROR:
                # Agent execution failed — still try to synthesize
                agent_results.append(
                    {
                        "agent": event.agent_name or "unknown",
                        "answer": event.data.get("error", "Unknown error"),
                        "status": "error",
                    }
                )

        # ------------------------------------------------------------------
        # 3a. NEEDS_INPUT — emit a dedicated event so the gateway can
        #     forward the agent's clarification question to the frontend.
        # ------------------------------------------------------------------
        for ar in agent_results:
            if ar.get("status") == "needs_input" and not ar.get("clarification_questions"):
                # Skip NEEDS_INPUT for planning-agent clarification results —
                # those are handled by the dedicated ASK_QUESTIONS path in 3b
                # which carries the structured questions list.
                question = ar.get("question") or ar.get("answer", "")
                if question:
                    yield Event(
                        type=EventType.NEEDS_INPUT,
                        agent_name=ar.get("agent"),
                        data={
                            "agent": ar.get("agent"),
                            "question": question,
                            "message": question,
                            "options": ar.get("options"),
                        },
                    )

        # ------------------------------------------------------------------
        # 3b. PLANNING AGENT — special status handling
        #
        # The planning agent returns custom statuses that need dedicated
        # handling:
        #   - clarification_questions -> ASK_QUESTIONS event, stop
        #   - needs_approval + plan   -> PLAN_CREATED event, stop
        #   - step_ready              -> execute the step via normal
        #                                executor, emit step events,
        #                                then fall through to synthesis
        # ------------------------------------------------------------------
        _planning_handled = False
        for ar in agent_results:
            # --- CLARIFICATION QUESTIONS ---
            if ar.get("clarification_questions"):
                yield Event(
                    type=EventType.ASK_QUESTIONS,
                    agent_name="planning",
                    data={
                        "agent": "planning",
                        "questions": ar["clarification_questions"],
                    },
                )
                yield Event(
                    type=EventType.ORCHESTRATOR_DONE,
                    data={
                        "response_text": ar.get("answer", ""),
                        "decision": decision.decision,
                        "agents_used": agents_used,
                        "elapsed_ms": _elapsed_ms(start_time),
                    },
                )
                return  # Skip synthesis — nothing to synthesize

            # --- PLAN CREATED ---
            if ar.get("plan") and ar.get("status") == "needs_approval":
                yield Event(
                    type=EventType.PLAN_CREATED,
                    agent_name="planning",
                    data={
                        "agent": "planning",
                        "plan": ar["plan"],
                    },
                )
                yield Event(
                    type=EventType.ORCHESTRATOR_DONE,
                    data={
                        "response_text": ar.get("answer", ""),
                        "decision": decision.decision,
                        "agents_used": agents_used,
                        "result_for_ui": {"plan": ar["plan"]},
                        "elapsed_ms": _elapsed_ms(start_time),
                    },
                )
                return  # Skip synthesis — plan card is the response

            # --- STEP READY ---
            if ar.get("step_execution") and ar.get("status") == "step_ready":
                step_exec = ar["step_execution"]

                yield Event(
                    type=EventType.PLAN_STEP_STARTED,
                    agent_name=step_exec.get("agent"),
                    data={
                        "plan_id": step_exec.get("plan_id"),
                        "step_id": step_exec.get("step_id"),
                        "step_index": step_exec.get("step_index"),
                        "agent": step_exec["agent"],
                    },
                )

                # Build single-step plan for the normal executor
                single_step_plan = Plan(
                    reasoning=f"Plan step: {step_exec['task'][:100]}",
                    steps=[
                        Step(
                            agent_key=step_exec["agent"],
                            task=step_exec["task"],
                        )
                    ],
                )

                # Strip workflow_context before delegating to the
                # target agent.  workflow_context carries the full plan
                # (all steps, original query, plan description) which is
                # meant for the planning agent only.  Forwarding it to
                # the delegated agent leaks the full plan into its
                # system prompt, causing it to "helpfully" execute work
                # from future steps instead of just its own step.
                step_request = request.model_copy(
                    update={"workflow_context": None}
                )

                step_agent_results: list[dict[str, Any]] = []
                async for event in execute_plan(single_step_plan, registry, step_request):
                    yield event  # Forward all agent events
                    if event.type == EventType.AGENT_RESULT:
                        step_agent_results.append(event.data.get("result_for_ui", {}))

                # Emit plan step result event
                step_result = step_agent_results[0] if step_agent_results else {}
                step_status = step_result.get("status", "completed")

                # --- WORKFLOW SUBMITTED ---
                # If the step submitted a long-running workflow to GoWe,
                # mark the step as completed (submission succeeded) but
                # pause the plan so the frontend can wait for the workflow
                # to finish before advancing to the next step.
                if step_result.get("submission_id") or step_result.get(
                    "submission_ids"
                ) or step_result.get("auto_submitted"):
                    # Step completed (the submission itself succeeded)
                    yield Event(
                        type=EventType.PLAN_STEP_COMPLETED,
                        agent_name=step_exec.get("agent"),
                        data={
                            "plan_id": step_exec.get("plan_id"),
                            "step_id": step_exec.get("step_id"),
                            "step_index": step_exec.get("step_index"),
                            "result_summary": step_result.get("answer", "")[:500],
                            "agent_result": step_result,
                        },
                    )
                    # Notify that a workflow was submitted and the plan
                    # should pause until the workflow reaches a terminal
                    # state (COMPLETED / FAILED / CANCELLED).
                    yield Event(
                        type=EventType.PLAN_WORKFLOW_SUBMITTED,
                        agent_name=step_exec.get("agent"),
                        data={
                            "plan_id": step_exec.get("plan_id"),
                            "step_id": step_exec.get("step_id"),
                            "step_index": step_exec.get("step_index"),
                            "workflow_id": step_result.get("workflow_id", ""),
                            "submission_id": step_result.get("submission_id", ""),
                            "submission_ids": step_result.get("submission_ids", []),
                            "result_summary": step_result.get("answer", "")[:500],
                        },
                    )
                    # Return early — skip synthesis and auto-advance.
                    # The gateway will register a workflow watch and the
                    # frontend will show a "waiting" state on the PlanCard.
                    yield Event(
                        type=EventType.ORCHESTRATOR_DONE,
                        data={
                            "response_text": step_result.get("answer", ""),
                            "decision": decision.decision,
                            "agents_used": agents_used,
                            "elapsed_ms": _elapsed_ms(start_time),
                        },
                    )
                    return

                if step_status in ("completed", "max_iterations"):
                    step_completed_data: dict[str, Any] = {
                        "plan_id": step_exec.get("plan_id"),
                        "step_id": step_exec.get("step_id"),
                        "step_index": step_exec.get("step_index"),
                        "result_summary": step_result.get("answer", "")[:500],
                        "agent_result": step_result,
                    }
                    if step_result.get("structured_data"):
                        step_completed_data["structured_data"] = step_result[
                            "structured_data"
                        ]
                    yield Event(
                        type=EventType.PLAN_STEP_COMPLETED,
                        agent_name=step_exec.get("agent"),
                        data=step_completed_data,
                    )

                elif step_status == "needs_input":
                    # The delegated agent called ask_clarification — it
                    # needs user input before it can finish this step.
                    # Emit a dedicated event so the frontend pauses the
                    # plan and shows the question.  The answer text
                    # (which describes the question) will flow through
                    # synthesis to become a visible chat message.
                    question_text = (
                        step_result.get("question")
                        or step_result.get("answer", "")
                    )
                    yield Event(
                        type=EventType.PLAN_STEP_NEEDS_INPUT,
                        agent_name=step_exec.get("agent"),
                        data={
                            "plan_id": step_exec.get("plan_id"),
                            "step_id": step_exec.get("step_id"),
                            "step_index": step_exec.get("step_index"),
                            "question": question_text,
                            "agent": step_exec.get("agent"),
                        },
                    )

                else:
                    yield Event(
                        type=EventType.PLAN_STEP_FAILED,
                        agent_name=step_exec.get("agent"),
                        data={
                            "plan_id": step_exec.get("plan_id"),
                            "step_id": step_exec.get("step_id"),
                            "step_index": step_exec.get("step_index"),
                            "error": step_result.get("answer", "Step failed"),
                        },
                    )

                # Replace agent_results so the synthesizer works on the
                # step result. The step result becomes the chat message.
                agent_results = [step_result]
                _planning_handled = True
                break  # Only handle one step_ready per request

            # --- DIRECT STEP COMPLETED ---
            # When a 'direct' step is executed, the planning agent
            # returns status="completed" with step_execution data.
            # No synthesis needed — the step result is emitted directly.
            if (
                ar.get("step_execution")
                and ar.get("status") == "completed"
                and ar["step_execution"].get("agent") == "direct"
            ):
                step_exec = ar["step_execution"]
                direct_answer = step_exec.get(
                    "direct_answer", ar.get("answer", "")
                )

                yield Event(
                    type=EventType.PLAN_STEP_COMPLETED,
                    agent_name="planning",
                    data={
                        "plan_id": step_exec.get("plan_id"),
                        "step_id": step_exec.get("step_id"),
                        "step_index": step_exec.get("step_index"),
                        "result_summary": direct_answer[:500],
                        "agent_result": {
                            "agent": "planning",
                            "answer": direct_answer,
                            "status": "completed",
                        },
                    },
                )
                # Return early — skip synthesis. The PlanCard shows
                # the step result directly; no chat message needed.
                yield Event(
                    type=EventType.ORCHESTRATOR_DONE,
                    data={
                        "response_text": direct_answer,
                        "decision": decision.decision,
                        "agents_used": agents_used,
                        "elapsed_ms": _elapsed_ms(start_time),
                    },
                )
                return

        # ------------------------------------------------------------------
        # 3c. AUTO-SUBMIT (when ORCH_AUTO_SUBMIT is enabled)
        # ------------------------------------------------------------------
        auto_submit = getattr(registry._config, "auto_submit", False)
        if auto_submit and agent_results:
            for ar in agent_results:
                wf_id = ar.get("workflow_id")
                persisted = ar.get("persisted", False)
                already_submitted = ar.get("auto_submitted", False) or ar.get(
                    "submission_id"
                )
                if wf_id and persisted and not already_submitted:
                    logger.info("Auto-submit enabled: submitting workflow %s", wf_id)
                    try:
                        # Find the service agent and call submit_workflow
                        service_agent = registry.get("service")
                        if (
                            service_agent
                            and "submit_workflow" in service_agent.tool_names
                        ):
                            submit_result = await service_agent.call_tool(
                                "submit_workflow",
                                {
                                    "workflow_id": wf_id,
                                    "token": request.auth_token or "",
                                },
                            )
                            # call_tool always returns a dict
                            submit_data = submit_result

                            if submit_data.get("error"):
                                logger.warning(
                                    "Auto-submit failed for %s: %s",
                                    wf_id,
                                    submit_data["error"],
                                )
                            else:
                                ar["auto_submitted"] = True
                                ar["submission_status"] = submit_data.get(
                                    "status", "pending"
                                )
                                # Capture GoWe submission_id if returned
                                if submit_data.get("submission_id"):
                                    ar["submission_id"] = submit_data["submission_id"]
                                logger.info(
                                    "Auto-submitted workflow %s: status=%s, "
                                    "submission_id=%s",
                                    wf_id,
                                    submit_data.get("status"),
                                    submit_data.get("submission_id"),
                                )
                        else:
                            logger.warning(
                                "Auto-submit: service agent not found or "
                                "missing submit_workflow tool"
                            )
                    except Exception as e:
                        logger.warning(
                            "Auto-submit failed for workflow %s: %s", wf_id, e
                        )

        # ------------------------------------------------------------------
        # 4. SYNTHESIZE response
        # ------------------------------------------------------------------
        response_text = ""
        result_for_ui: dict[str, Any] = {}

        async for event in synthesize(request, agent_results, llm):
            yield event

            if event.type == EventType.SYNTHESIS_DONE:
                response_text = event.data.get("response_text", "")

        # Merge result_for_ui from all agent results
        if len(agent_results) == 1:
            result_for_ui = agent_results[0]
        elif agent_results:
            result_for_ui = {"agent_results": agent_results}

        # ------------------------------------------------------------------
        # 5. DONE
        # ------------------------------------------------------------------
        logger.info(
            f"Orchestration complete: decision={decision.decision}, "
            f"response_text={len(response_text)} chars, "
            f"preview={response_text[:80]!r}, "
            f"agents_used={agents_used}"
        )
        yield Event(
            type=EventType.ORCHESTRATOR_DONE,
            data={
                "response_text": response_text,
                "decision": decision.decision,
                "agents_used": agents_used,
                "tool_calls": tool_calls,
                "result_for_ui": result_for_ui,
                "elapsed_ms": _elapsed_ms(start_time),
            },
        )

    except Exception as e:
        logger.error(f"Orchestration failed: {e}", exc_info=True)
        yield error_event(
            message=f"Orchestration error: {e}",
            details={"elapsed_ms": _elapsed_ms(start_time)},
        )
        yield Event(
            type=EventType.ORCHESTRATOR_DONE,
            data={
                "response_text": (
                    "I encountered an error while processing your request. "
                    "Please try again."
                ),
                "decision": "error",
                "elapsed_ms": _elapsed_ms(start_time),
            },
        )


async def orchestrate_to_response(
    request: OrchestratorRequest,
    registry: AgentRegistry,
    llm: LLMClient,
    routing_llm: LLMClient | None = None,
) -> OrchestratorResponse:
    """Convenience wrapper: run orchestrate() and collect into an OrchestratorResponse.

    Useful for non-streaming callers that just want the final result.
    """
    response_text = ""
    agents_used: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    result_for_ui: dict[str, Any] = {}
    execution_trace: list[dict[str, Any]] = []
    status = "completed"

    async for event in orchestrate(request, registry, llm, routing_llm=routing_llm):
        execution_trace.append(
            {
                "type": event.type.value,
                "data": event.data,
                "agent_name": event.agent_name,
                "timestamp": event.timestamp,
            }
        )

        if event.type == EventType.ORCHESTRATOR_DONE:
            response_text = event.data.get("response_text", "")
            agents_used = event.data.get("agents_used", [])
            tool_calls = event.data.get("tool_calls", [])
            result_for_ui = event.data.get("result_for_ui", {})
            if event.data.get("decision") == "error":
                status = "error"

        elif event.type == EventType.ORCHESTRATOR_ERROR:
            status = "error"

    return OrchestratorResponse(
        response_text=response_text,
        agent_used=agents_used[0] if agents_used else None,
        agents_used=agents_used,
        tool_calls=tool_calls,
        result_for_ui=result_for_ui,
        execution_trace=execution_trace,
        status=status,
    )


def _elapsed_ms(start: float) -> float:
    """Calculate elapsed time in milliseconds."""
    return round((time.monotonic() - start) * 1000, 1)
