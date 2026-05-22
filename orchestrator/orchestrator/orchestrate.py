"""Core orchestration loop.

The main orchestrate() async generator wires together:
  route -> execute -> synthesize

It yields Event objects throughout, making the entire pipeline
streamable to the gateway.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from orchestrator.events.events import Event, EventType, error_event
from orchestrator.executor.executor import execute_plan
from orchestrator.llm.client import LLMClient
from orchestrator.models import OrchestratorRequest, OrchestratorResponse
from orchestrator.registry.agent_registry import AgentRegistry
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
            "reasoning": (
                decision.plan.reasoning if decision.plan else None
            ),
        }
        if decision.plan and decision.plan.steps:
            if decision.decision == "pipeline":
                routing_event_data["steps"] = [
                    {"agent_key": s.agent_key, "task": s.task, "depends_on": s.depends_on}
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
                agent_results.append({
                    "agent": event.agent_name or "unknown",
                    "answer": event.data.get("error", "Unknown error"),
                    "status": "error",
                })

        # ------------------------------------------------------------------
        # 3b. AUTO-SUBMIT (when ORCH_AUTO_SUBMIT is enabled)
        # ------------------------------------------------------------------
        auto_submit = getattr(registry._config, "auto_submit", False)
        if auto_submit and agent_results:
            for ar in agent_results:
                wf_id = ar.get("workflow_id")
                persisted = ar.get("persisted", False)
                if wf_id and persisted:
                    logger.info(
                        "Auto-submit enabled: submitting workflow %s", wf_id
                    )
                    try:
                        # Find the service2 agent and call submit_workflow
                        service2 = registry.get("service2")
                        if service2 and "submit_workflow" in service2.tool_names:
                            submit_result = await service2.call_tool(
                                "submit_workflow",
                                {
                                    "workflow_id": wf_id,
                                    "token": request.auth_token or "",
                                },
                            )
                            # Parse the MCP result
                            submit_data = {}
                            if hasattr(submit_result, "content"):
                                for block in submit_result.content:
                                    if hasattr(block, "text"):
                                        try:
                                            submit_data = json.loads(block.text)
                                        except (json.JSONDecodeError, TypeError):
                                            pass
                                        break

                            if submit_data.get("error"):
                                logger.warning(
                                    "Auto-submit failed for %s: %s",
                                    wf_id, submit_data["error"],
                                )
                            else:
                                ar["auto_submitted"] = True
                                ar["submission_status"] = submit_data.get(
                                    "status", "pending"
                                )
                                logger.info(
                                    "Auto-submitted workflow %s: status=%s",
                                    wf_id, submit_data.get("status"),
                                )
                        else:
                            logger.warning(
                                "Auto-submit: service2 agent not found or "
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
        execution_trace.append({
            "type": event.type.value,
            "data": event.data,
            "agent_name": event.agent_name,
            "timestamp": event.timestamp,
        })

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
