"""
Core agent orchestrator for the BV-BRC Service Agent v2.

Three-phase workflow construction:
  Phase 1: Decompose -- user request -> WorkflowPlan (abstract DAG)
  Phase 2: Build     -- WorkflowPlan -> ValidatedStep[] (incremental)
  Phase 3: Compose   -- ValidatedStep[] -> manifest (programmatic)

Single entry point: run_agent(). No plan_only mode -- the entire agent's
purpose is workflow planning. It never submits or executes anything.
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
    """
    Three-phase workflow construction. Single entry point.

    Returns AgentResult with status:
    - "completed": Full workflow manifest ready
    - "needs_input": Agent needs user input to continue (question in result)
    - "error": Unrecoverable error

    Args:
        query: Natural language service request.
        config: Agent configuration. Uses defaults if not provided.
        context: Optional additional context (e.g., prior conversation state).
        progress_callback: Optional async callback for progress updates.
            Signature: async (progress, total, message) -> None.

    Returns:
        AgentResult with manifest, question, or error.
    """
    cfg = config or AgentConfig()
    state = AgentState(query=query, context=context or {})

    # ------------------------------------------------------------------
    # Phase 1: Decompose (if no plan yet)
    # ------------------------------------------------------------------
    if not state.workflow_plan:
        state.current_phase = "decompose"
        await emit_progress(progress_callback, 0, 3, "Phase 1: Analyzing request and identifying services...")
        state = await decompose(query, cfg, state, progress_callback=progress_callback)

        if state.status == "needs_input":
            return state.to_result()

        if state.status == "error":
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
            state = await build_step(batch[0], cfg, state, progress_callback=progress_callback)

            if state.status == "needs_input":
                return state.to_result()

            if state.status == "error":
                return state.to_result()
        else:
            # Parallel build for independent steps in the same batch
            # Create separate state snapshots for each parallel build,
            # then merge results back
            await emit_progress(progress_callback, 1, 3, f"Building {len(batch)} steps in parallel: {', '.join(batch)}...")
            results = await asyncio.gather(
                *[_build_step_isolated(step_id, cfg, state, progress_callback=progress_callback) for step_id in batch],
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

                # result is an AgentState from the isolated build
                if result.status == "needs_input":
                    # Surface the first question encountered
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

    manifest = compose_manifest(state, cfg)

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

        client = WorkflowEngineClient(base_url=cfg.workflow_engine_url)
        # The compose result is a wrapper dict with metadata; the engine
        # expects the canonical workflow definition which lives under the
        # "manifest" key.  Fall back to the full dict if the key is absent
        # (e.g. if compose_manifest changes its output format).
        engine_payload = manifest
        if isinstance(manifest, dict) and isinstance(manifest.get("manifest"), dict):
            engine_payload = manifest["manifest"]
        engine_result = await client.plan_workflow(
            engine_payload, cfg.bvbrc_auth_token or ""
        )
        state.workflow_id = engine_result.get("workflow_id")
        state.persisted = True
        # Update the manifest dict so it carries the authoritative engine
        # ID rather than the local compose-time placeholder.
        if state.workflow_id and isinstance(state.manifest, dict):
            state.manifest["workflow_id"] = state.workflow_id
        logger.info(
            "Manifest persisted to workflow engine: workflow_id=%s",
            state.workflow_id,
        )
    except Exception as e:
        state.persisted = False
        # Clear the placeholder ID from the manifest so it cannot be
        # mistaken for a real engine-registered workflow.
        if isinstance(state.manifest, dict) and "workflow_id" in state.manifest:
            state.manifest.pop("workflow_id", None)
        logger.warning(
            "Failed to persist manifest to workflow engine: %s: %s",
            type(e).__name__, e,
        )

    await emit_progress(progress_callback, 3, 3, "Workflow planning complete.")
    return state.to_result()


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
    # Create a shallow copy of state for this parallel branch
    isolated_state = AgentState(
        query=parent_state.query,
        context=parent_state.context,
        current_phase="build",
        workflow_plan=parent_state.workflow_plan,
        completed_steps=dict(parent_state.completed_steps),
        start_time=parent_state.start_time,
    )

    return await build_step(step_id, config, isolated_state, progress_callback=progress_callback)
