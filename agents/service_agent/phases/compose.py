"""Phase 3: Compose -- assemble validated steps into a workflow manifest.

This phase is entirely programmatic -- no LLM involvement. It resolves
abstract references (output_of:X:Y) to concrete template paths and
calls the MCP server's compose_workflow_manifest function.
"""

from __future__ import annotations

import re
from typing import Any

from service_agent.models import AgentConfig, AgentState, ValidatedStep, WorkflowPlan


# ---------------------------------------------------------------------------
# Template resolution
# ---------------------------------------------------------------------------

_OUTPUT_OF_PATTERN = re.compile(r"^output_of:(\w+):(\w+)$")


def _resolve_output_references(
    params: dict[str, Any],
    completed_steps: dict[str, ValidatedStep],
) -> dict[str, Any]:
    """Resolve output_of:step:key references in parameters.

    For each parameter value matching 'output_of:<step_id>:<output_key>',
    replace it with the template variable '${steps.<step_id>.outputs.<key>}'.

    If the upstream step has a concrete output path in its output_patterns,
    use that path directly. Otherwise, use the template variable format.
    """
    resolved = {}
    for key, value in params.items():
        if isinstance(value, str):
            match = _OUTPUT_OF_PATTERN.match(value)
            if match:
                step_id = match.group(1)
                output_key = match.group(2)

                # Try to get the concrete output path from the upstream step
                upstream = completed_steps.get(step_id)
                if upstream and output_key in upstream.output_patterns:
                    resolved[key] = upstream.output_patterns[output_key]
                else:
                    # Fallback to template variable
                    resolved[key] = (
                        f"${{steps.{step_id}.outputs.{output_key}}}"
                    )
            else:
                resolved[key] = value
        elif isinstance(value, list):
            # Resolve within lists too
            resolved_list = []
            for item in value:
                if isinstance(item, str):
                    match = _OUTPUT_OF_PATTERN.match(item)
                    if match:
                        step_id = match.group(1)
                        output_key = match.group(2)
                        upstream = completed_steps.get(step_id)
                        if upstream and output_key in upstream.output_patterns:
                            resolved_list.append(
                                upstream.output_patterns[output_key]
                            )
                        else:
                            resolved_list.append(
                                f"${{steps.{step_id}.outputs.{output_key}}}"
                            )
                    else:
                        resolved_list.append(item)
                else:
                    resolved_list.append(item)
            resolved[key] = resolved_list
        else:
            resolved[key] = value
    return resolved


# ---------------------------------------------------------------------------
# Manifest composition
# ---------------------------------------------------------------------------

def compose_manifest(
    state: AgentState,
    config: AgentConfig | None = None,
) -> dict[str, Any]:
    """
    Phase 3: Compose all validated steps into a workflow manifest.

    This is a deterministic function -- no LLM. It:
    1. Resolves output_of:X:Y references to concrete paths
    2. Calls compose_workflow_manifest() from the MCP server
    3. Returns the final manifest

    Args:
        state: Agent state with completed_steps and workflow_plan.
        config: Agent configuration (for MCP server path).

    Returns:
        The workflow manifest dict.
    """
    if not state.workflow_plan:
        return {"error": "No workflow plan available for composition."}

    if not state.completed_steps:
        return {"error": "No completed steps available for composition."}

    cfg = config or AgentConfig()

    # Build steps for compose_workflow_manifest
    steps_for_composition: list[dict[str, Any]] = []

    for step_id in state.workflow_plan.topological_order:
        if step_id not in state.completed_steps:
            return {
                "error": f"Step '{step_id}' not completed. Cannot compose manifest.",
            }

        validated = state.completed_steps[step_id]

        # Resolve output_of references in params
        resolved_params = _resolve_output_references(
            validated.params,
            state.completed_steps,
        )

        steps_for_composition.append({
            "step_name": step_id,
            "service_name": validated.service_name,
            "params": resolved_params,
            "depends_on": validated.depends_on if validated.depends_on else [],
        })

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

    # Call compose_workflow_manifest from MCP server
    try:
        from service_agent.tools._mcp_imports import (
            get_workflow_composition_functions,
        )
        wf_fn = get_workflow_composition_functions(cfg)
        manifest = wf_fn.compose_workflow_manifest(
            steps=steps_for_composition,
            user_id=user_id,
            workflow_name=state.workflow_plan.workflow_name,
        )
        return manifest

    except Exception as e:
        return {
            "error": f"Workflow composition failed: {type(e).__name__}: {str(e)}",
        }
