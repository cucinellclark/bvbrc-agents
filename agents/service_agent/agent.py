"""
Core agent orchestrator for the BV-BRC Service Agent v2.

Intent-dispatched architecture:
  1. Classify intent (lightweight LLM call or main model, configurable)
  2. Dispatch to the appropriate handler:
     - plan   -> 3-phase pipeline (Decompose -> Build -> Compose -> Persist)
     - submit -> direct workflow engine call
     - status -> direct workflow engine call
     - cancel -> direct workflow engine call
     - modify -> reserved for future use (falls through to plan)
     - unknown -> falls through to plan (LLM-powered fallback)

Single entry point: run_agent().
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Shared utilities -- deduplicated across all agents
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))
from agent_utils import emit_progress

from service_agent.classifier import classify_intent
from service_agent.handlers import handle_submit, handle_status, handle_cancel
from service_agent.models import AgentConfig, AgentResult, AgentState
from service_agent.phases.decompose import decompose
from service_agent.phases.build import build_step
from service_agent.phases.compose import compose_manifest

logger = logging.getLogger(__name__)

ProgressCallback = Any  # async (progress: float, total: float|None, message: str) -> None


async def run_agent(
    query: str,
    config: AgentConfig | None = None,
    context: dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AgentResult:
    """Service agent entry point.  Classifies intent, then dispatches.

    Returns AgentResult with status:
    - "completed": Workflow manifest ready, or lifecycle operation succeeded
    - "needs_input": Agent needs user input to continue (question in result)
    - "error": Unrecoverable error

    Args:
        query: Natural language service request.
        config: Agent configuration. Uses defaults if not provided.
        context: Optional additional context (e.g., prior conversation state).
        progress_callback: Optional async callback for progress updates.

    Returns:
        AgentResult with manifest, question, operation_message, or error.
    """
    cfg = config or AgentConfig()
    ctx = context or {}
    state = AgentState(query=query, context=ctx)

    # ------------------------------------------------------------------
    # Step 1: Classify intent
    # ------------------------------------------------------------------
    await emit_progress(progress_callback, 0, 1, "Analyzing request...")
    intent = await classify_intent(query, ctx, cfg)
    state.classified_intent = intent

    logger.info(
        "Intent: action=%s, workflow_id=%s, confidence=%.2f, "
        "submit_after_plan=%s",
        intent.action, intent.workflow_id, intent.confidence,
        intent.submit_after_plan,
    )

    # ------------------------------------------------------------------
    # Step 2: Dispatch based on intent
    # ------------------------------------------------------------------

    # Submit: direct engine call, no LLM
    if intent.action == "submit" and intent.workflow_id:
        return await handle_submit(
            intent.workflow_id, cfg, state, progress_callback,
        )

    # Status: direct engine call, no LLM
    if intent.action == "status" and intent.workflow_id:
        return await handle_status(
            intent.workflow_id, cfg, state, progress_callback,
        )

    # Cancel: direct engine call, no LLM
    if intent.action == "cancel" and intent.workflow_id:
        return await handle_cancel(
            intent.workflow_id, cfg, state, progress_callback,
        )

    # Unknown with no workflow_id: probably ambiguous, ask for clarification
    if intent.action == "unknown":
        state.status = "needs_input"
        state.question = (
            "I'm not sure which workflow you're referring to. "
            "Could you clarify which workflow you'd like to "
            "submit, check, or cancel?"
        )
        return state.to_result()

    # Submit/status/cancel without workflow_id: classifier couldn't resolve
    if intent.action in ("submit", "status", "cancel") and not intent.workflow_id:
        state.status = "needs_input"
        state.question = (
            f"I understand you want to {intent.action} a workflow, but I "
            f"couldn't determine which one. Could you specify the workflow "
            f"ID or describe which workflow you mean?"
        )
        return state.to_result()

    # Plan (or modify/fallback): run the full 3-phase pipeline
    return await _run_planning_pipeline(query, cfg, state, progress_callback)


# ======================================================================
# 3-Phase Planning Pipeline
# ======================================================================

async def _run_planning_pipeline(
    query: str,
    config: AgentConfig,
    state: AgentState,
    progress_callback: ProgressCallback | None = None,
) -> AgentResult:
    """Three-phase workflow construction: Decompose -> Build -> Compose.

    This is the existing planning logic, extracted from the former
    run_agent() to keep the dispatcher clean.
    """

    # ------------------------------------------------------------------
    # Phase 1: Decompose (if no plan yet)
    # ------------------------------------------------------------------
    if not state.workflow_plan:
        state.current_phase = "decompose"
        await emit_progress(progress_callback, 0, 3, "Phase 1: Analyzing request and identifying services...")
        state = await decompose(query, config, state, progress_callback=progress_callback)

        if state.status == "needs_input":
            return state.to_result()

        if state.status == "error":
            return state.to_result()

        if state.status == "completed":
            # submit_workflow short-circuit: decompose handled a submit
            # request and set status to "completed" without producing a
            # workflow plan.  Return immediately — no build/compose needed.
            # NOTE: This path is legacy -- the classifier should catch
            # submit requests before we reach here.  Kept as a safety net.
            return state.to_result()

        if not state.workflow_plan:
            state.status = "error"
            state.error_message = "Phase 1 completed without producing a plan."
            return state.to_result()

    # ------------------------------------------------------------------
    # Phase 2: Build Steps (incrementally, batch independent steps)
    # ------------------------------------------------------------------
    state.current_phase = "build"
    _total_steps = len(state.workflow_plan.steps) if state.workflow_plan else 0
    await emit_progress(progress_callback, 1, 3, f"Phase 2: Building {_total_steps} service step(s)...")

    for batch in state.next_buildable_batches():
        if len(batch) == 1:
            await emit_progress(progress_callback, 1, 3, f"Building step '{batch[0]}'...")
            # Sequential build for single-step batches
            state = await build_step(batch[0], config, state, progress_callback=progress_callback)

            if state.status == "needs_input":
                return state.to_result()

            if state.status == "error":
                return state.to_result()
        else:
            # Parallel build for independent steps in the same batch
            await emit_progress(progress_callback, 1, 3, f"Building {len(batch)} steps in parallel: {', '.join(batch)}...")
            results = await asyncio.gather(
                *[_build_step_isolated(step_id, config, state, progress_callback=progress_callback) for step_id in batch],
                return_exceptions=True,
            )

            for step_id, result in zip(batch, results):
                if isinstance(result, Exception):
                    state.status = "error"
                    state.error_message = (
                        f"Parallel build failed for step '{step_id}': "
                        f"{type(result).__name__}: {str(result)}"
                    )
                    return state.to_result()

                if result.status == "needs_input":
                    state.status = "needs_input"
                    state.question = result.question
                    return state.to_result()

                if result.status == "error":
                    state.status = "error"
                    state.error_message = result.error_message
                    return state.to_result()

                # Merge completed step and tool executions back
                if step_id in result.completed_steps:
                    state.completed_steps[step_id] = result.completed_steps[step_id]
                state.tool_executions.extend(result.tool_executions)

    # ------------------------------------------------------------------
    # Phase 3: Compose (programmatic -- no LLM)
    # ------------------------------------------------------------------
    state.current_phase = "compose"
    await emit_progress(progress_callback, 2, 3, "Phase 3: Composing workflow manifest...")

    manifest = compose_manifest(state, config)

    if isinstance(manifest, dict) and "error" in manifest:
        state.status = "error"
        state.error_message = manifest["error"]
        return state.to_result()

    state.manifest = manifest
    state.status = "completed"
    state.current_phase = "done"

    await emit_progress(progress_callback, 2, 3, "Workflow manifest composed successfully.")

    # ------------------------------------------------------------------
    # Persist manifest to the workflow engine to get a real workflow_id
    # ------------------------------------------------------------------
    await emit_progress(progress_callback, 2, 3, "Persisting to workflow engine...")
    try:
        from common.workflow_engine_client import WorkflowEngineClient

        client = WorkflowEngineClient(base_url=config.workflow_engine_url)
        engine_payload = manifest
        if isinstance(manifest, dict) and isinstance(manifest.get("manifest"), dict):
            engine_payload = manifest["manifest"]
        _session_id = state.context.get("session_id")
        engine_result = await client.plan_workflow(
            engine_payload, config.bvbrc_auth_token or "",
            session_id=_session_id,
        )
        state.workflow_id = engine_result.get("workflow_id")
        state.persisted = True
        if state.workflow_id and isinstance(state.manifest, dict):
            state.manifest["workflow_id"] = state.workflow_id
        logger.info(
            "Manifest persisted to workflow engine: workflow_id=%s",
            state.workflow_id,
        )
    except Exception as e:
        state.persisted = False
        if isinstance(state.manifest, dict) and "workflow_id" in state.manifest:
            state.manifest.pop("workflow_id", None)
        logger.warning(
            "Failed to persist manifest to workflow engine: %s: %s",
            type(e).__name__, e,
        )

    # ------------------------------------------------------------------
    # Auto-submit (based on user preference in context)
    # ------------------------------------------------------------------
    preference = (
        state.context.get("auto_submit_preference")
        or getattr(config, "auto_submit_preference", None)
        or "always_review"
    )
    if (
        preference != "always_review"
        and state.workflow_id
        and state.persisted
    ):
        complexity = _assess_complexity(state)
        should_submit = (
            preference == "auto_all"
            or (preference == "auto_simple" and complexity == "simple")
        )
        if should_submit:
            logger.info(
                "Auto-submitting workflow %s (preference=%s, complexity=%s)",
                state.workflow_id, preference, complexity,
            )
            await handle_submit(
                state.workflow_id, config, state, progress_callback,
            )
            # The submit handler mutated state; mark as auto-submitted
            state.auto_submitted = True
            return state.to_result()

    # ------------------------------------------------------------------
    # Auto-submit (based on classifier detecting "and submit" intent)
    # ------------------------------------------------------------------
    if (
        state.classified_intent
        and state.classified_intent.submit_after_plan
        and state.workflow_id
        and state.persisted
        and not state.auto_submitted
    ):
        logger.info(
            "Auto-submitting workflow %s (submit_after_plan=True from classifier)",
            state.workflow_id,
        )
        await handle_submit(
            state.workflow_id, config, state, progress_callback,
        )
        state.auto_submitted = True
        return state.to_result()

    await emit_progress(progress_callback, 3, 3, "Workflow planning complete.")
    return state.to_result()


def _assess_complexity(state: AgentState) -> str:
    """Assess workflow complexity for auto-submit decisions.

    Returns "simple" for single-step workflows with no dependencies,
    "complex" for everything else.
    """
    if not state.workflow_plan:
        return "simple"
    steps = state.workflow_plan.steps
    if len(steps) == 1 and not steps[0].depends_on:
        return "simple"
    return "complex"


async def _build_step_isolated(
    step_id: str,
    config: AgentConfig,
    parent_state: AgentState,
    progress_callback: ProgressCallback | None = None,
) -> AgentState:
    """Build a step with an isolated state copy for parallel execution.

    Creates a minimal state copy with the plan and completed steps from
    the parent, runs build_step, and returns the updated state.
    """
    isolated_state = AgentState(
        query=parent_state.query,
        context=parent_state.context,
        current_phase="build",
        workflow_plan=parent_state.workflow_plan,
        completed_steps=dict(parent_state.completed_steps),
        start_time=parent_state.start_time,
    )

    return await build_step(step_id, config, isolated_state, progress_callback=progress_callback)
