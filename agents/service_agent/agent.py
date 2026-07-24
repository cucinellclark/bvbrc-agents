"""
Core agent orchestrator for the BV-BRC Service Agent.

Intent-dispatched architecture:
  1. Classify intent (lightweight LLM call, configurable)
  2. Dispatch to the appropriate handler:
     - plan   -> GoWe-first workflow: discover, select, populate, submit
     - submit -> direct GoWe call
     - status -> direct GoWe call
     - cancel -> direct GoWe call
     - modify -> reserved (falls through to plan)
     - unknown -> falls through to plan (LLM-powered fallback)

Single entry point: run_agent().
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

# Shared utilities -- deduplicated across all agents
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))
# Also add repo root so `shared` package imports work (shared.tools, shared.prompts)
if str(Path(__file__).resolve().parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from agent_utils import emit_progress

from service_agent.classifier import classify_intent
from service_agent.handlers import handle_submit, handle_status, handle_cancel
from service_agent.models import AgentConfig, AgentResult, AgentState
from service_agent.phases.populate import populate_and_submit

logger = logging.getLogger(__name__)

ProgressCallback = (
    Any  # async (progress: float, total: float|None, message: str) -> None
)


async def run_agent(
    query: str,
    config: AgentConfig | None = None,
    context: dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AgentResult:
    """Service agent entry point.  Classifies intent, then dispatches.

    Returns AgentResult with status:
    - "completed": Workflow submitted, or lifecycle operation succeeded
    - "needs_input": Agent needs user input to continue (question in result)
    - "error": Unrecoverable error

    Args:
        query: Natural language service request.
        config: Agent configuration. Uses defaults if not provided.
        context: Optional additional context (e.g., prior conversation state).
        progress_callback: Optional async callback for progress updates.

    Returns:
        AgentResult with operation result or error.
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
        "Intent: action=%s, workflow_id=%s, confidence=%.2f, submit_after_plan=%s",
        intent.action,
        intent.workflow_id,
        intent.confidence,
        intent.submit_after_plan,
    )

    # ------------------------------------------------------------------
    # Step 2: Dispatch based on intent
    # ------------------------------------------------------------------

    # Submit: direct engine call, no LLM
    if intent.action == "submit" and intent.workflow_id:
        return await handle_submit(
            intent.workflow_id,
            cfg,
            state,
            progress_callback,
        )

    # Status: direct engine call, no LLM
    if intent.action == "status" and intent.workflow_id:
        return await handle_status(
            intent.workflow_id,
            cfg,
            state,
            progress_callback,
        )

    # Cancel: direct engine call, no LLM
    if intent.action == "cancel" and intent.workflow_id:
        return await handle_cancel(
            intent.workflow_id,
            cfg,
            state,
            progress_callback,
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

    # Plan (or modify/fallback): GoWe-first workflow populate + submit.
    # The LLM discovers workflows from GoWe, selects one, populates
    # its inputs, and submits the job in a single loop.
    return await _run_gowe_flow(query, cfg, state, progress_callback)


# ======================================================================
# GoWe-first workflow flow (primary)
# ======================================================================


async def _run_gowe_flow(
    query: str,
    config: AgentConfig,
    state: AgentState,
    progress_callback: ProgressCallback | None = None,
) -> AgentResult:
    """GoWe-first workflow: discover, select, populate, submit.

    The LLM queries GoWe for available workflows, picks one, gets its
    input schema, populates the inputs using workspace/data tools, and
    submits the job.  All in a single LLM loop.
    """
    await emit_progress(
        progress_callback,
        0,
        1,
        "Discovering workflows and preparing job...",
    )

    state = await populate_and_submit(
        query,
        config,
        state,
        progress_callback=progress_callback,
    )

    if state.status in ("needs_input", "error"):
        return state.to_result()

    await emit_progress(progress_callback, 1, 1, "Done.")
    return state.to_result()
