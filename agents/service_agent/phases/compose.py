"""Phase 3: Compose -- assemble validated steps into a CWL workflow document.

This phase is entirely programmatic -- no LLM involvement. It generates
a CWL v1.2 $graph packed document and submission inputs using the M1 CWL
generator library, suitable for registration with GoWe.

Previously, this phase resolved output_of:X:Y references manually and
called the MCP server's compose_workflow_manifest(). Now the CWL generator
handles output reference resolution internally using CWL's step_id/output_id
syntax.
"""

from __future__ import annotations

from typing import Any

from service_agent.cwl.generator import generate_cwl_workflow, generate_submission_inputs
from service_agent.models import AgentConfig, AgentState


# ---------------------------------------------------------------------------
# Manifest composition (now CWL-based)
# ---------------------------------------------------------------------------

def compose_manifest(
    state: AgentState,
    config: AgentConfig | None = None,
) -> dict[str, Any]:
    """
    Phase 3: Compose all validated steps into a CWL workflow document.

    This is a deterministic function -- no LLM. It:
    1. Calls generate_cwl_workflow() to produce a CWL v1.2 $graph document
    2. Calls generate_submission_inputs() to produce CWL-typed inputs
    3. Returns a dict with both, plus metadata

    Args:
        state: Agent state with completed_steps and workflow_plan.
        config: Agent configuration (for extracting user_id from auth token).

    Returns:
        A dict with keys:
        - cwl_document: CWL v1.2 $graph document (dict)
        - submission_inputs: CWL-typed input values (dict)
        - workflow_name: Name of the workflow
        - step_count: Number of steps in the workflow
    """
    if not state.workflow_plan:
        return {"error": "No workflow plan available for composition."}

    if not state.completed_steps:
        return {"error": "No completed steps available for composition."}

    cfg = config or AgentConfig()

    # Verify all planned steps are completed
    for step_id in state.workflow_plan.topological_order:
        if step_id not in state.completed_steps:
            return {
                "error": f"Step '{step_id}' not completed. Cannot compose manifest.",
            }

    # Extract user_id from auth token
    user_id = "anonymous"
    if cfg.bvbrc_auth_token:
        try:
            for part in cfg.bvbrc_auth_token.split("|"):
                if part.startswith("un="):
                    user_id = part[3:]
                    break
        except Exception:
            pass

    try:
        # Generate CWL workflow document
        cwl_doc = generate_cwl_workflow(
            state.workflow_plan,
            state.completed_steps,
            user_id=user_id,
        )

        # Generate submission inputs
        submission_inputs = generate_submission_inputs(
            state.completed_steps,
            user_id=user_id,
        )

        return {
            "cwl_document": cwl_doc,
            "submission_inputs": submission_inputs,
            "workflow_name": state.workflow_plan.workflow_name,
            "step_count": len(state.workflow_plan.steps),
        }

    except Exception as e:
        return {
            "error": f"CWL workflow generation failed: {type(e).__name__}: {str(e)}",
        }
