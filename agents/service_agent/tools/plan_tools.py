"""
Phase 1 tool: create_workflow_plan.

Captures the LLM's decomposition of a user request into a structured
WorkflowPlan (abstract DAG of service steps with dependencies).
This is a local tool -- it does not call any external API. It validates
the plan structure and computes topological order and subgraph info.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from service_agent.models import AgentConfig, StepPlan, WorkflowPlan


# ---------------------------------------------------------------------------
# Valid service names (kept in sync with service_mapping.json)
# ---------------------------------------------------------------------------

VALID_SERVICE_NAMES = {
    "bacterial_genome_tree",
    "blast",
    "comparative_systems",
    "comprehensive_genome_analysis",
    "core_genome_mlst",
    "date",
    "docking",
    "expression_import",
    "fastqutils",
    "gene_tree",
    "genome_alignment",
    "genome_annotation",
    "genome_assembly",
    "influenza_ha_subtype_conversion",
    "metacats",
    "metagenomic_binning",
    "metagenomic_read_mapping",
    "msa_snp_analysis",
    "primer_design",
    "proteome_comparison",
    "rnaseq",
    "sars_genome_analysis",
    "sars_wastewater_analysis",
    "sequence_submission",
    "similar_genome_finder",
    "subspecies_classification",
    "taxonomic_classification",
    "tnseq",
    "variation",
    "viral_assembly",
    "whole_genome_snp",
}


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------

async def create_workflow_plan(
    workflow_name: str,
    description: str,
    steps: List[Dict[str, Any]],
    config: Optional[AgentConfig] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Validate and create a structured WorkflowPlan from the LLM's decomposition.

    This is a local validation tool -- no external API calls.

    Args:
        workflow_name: Short name for the workflow.
        description: What the workflow accomplishes.
        steps: List of step dicts, each with:
            - step_id: Unique identifier (snake_case)
            - service_name: BV-BRC service name
            - intent: What this step does
            - depends_on: List of step_ids this step depends on (optional)
            - input_sources: Dict of param_name -> source description (optional)

    Returns:
        Dict with:
          - status: "valid" or "invalid"
          - plan: The validated WorkflowPlan dict (if valid)
          - errors: List of validation errors (if invalid)
          - topological_order: Build order (if valid)
          - independent_subgraphs: Disconnected components (if valid)
    """
    errors: list[str] = []

    # Parse steps into StepPlan objects
    step_plans: list[StepPlan] = []
    step_ids: set[str] = set()

    for i, step_dict in enumerate(steps):
        # Validate required fields
        if "step_id" not in step_dict:
            errors.append(f"Step {i}: missing required field 'step_id'")
            continue
        if "service_name" not in step_dict:
            errors.append(
                f"Step {i} ({step_dict.get('step_id', '?')}): "
                f"missing required field 'service_name'"
            )
            continue
        if "intent" not in step_dict:
            errors.append(
                f"Step {i} ({step_dict['step_id']}): "
                f"missing required field 'intent'"
            )
            continue

        step_id = step_dict["step_id"]

        # Check uniqueness
        if step_id in step_ids:
            errors.append(f"Duplicate step_id: '{step_id}'")
            continue
        step_ids.add(step_id)

        # Validate service name
        service_name = step_dict["service_name"]
        if service_name not in VALID_SERVICE_NAMES:
            # Give a more targeted hint for common mistakes
            tool_names = {"get_sra_metadata", "list_services", "get_service_schema",
                          "plan_service", "workspace_browse", "search_data",
                          "get_genome_group", "get_feature_group", "read_file_info",
                          "create_workflow_plan", "compose_workflow"}
            if service_name in tool_names:
                errors.append(
                    f"Step '{step_id}': '{service_name}' is a TOOL, not a "
                    f"BV-BRC service. Tools (like get_sra_metadata) are called "
                    f"during planning -- they must NOT appear as workflow steps. "
                    f"Remove this step from the plan. The tool's results are "
                    f"already available from your earlier tool calls."
                )
            else:
                errors.append(
                    f"Step '{step_id}': unknown service_name '{service_name}'. "
                    f"Valid services: {', '.join(sorted(VALID_SERVICE_NAMES))}"
                )

        # Normalize input_sources: strict schema sends an array of
        # {param_name, source} objects; legacy callers may send a dict.
        raw_sources = step_dict.get("input_sources", [])
        coerced_sources: dict[str, Any] = {}
        if isinstance(raw_sources, list):
            for entry in raw_sources:
                if isinstance(entry, dict) and "param_name" in entry:
                    coerced_sources[entry["param_name"]] = str(
                        entry.get("source", "user_provided")
                    )
        elif isinstance(raw_sources, dict):
            for k, v in raw_sources.items():
                if isinstance(v, list):
                    coerced_sources[k] = "user_provided"
                elif isinstance(v, str):
                    coerced_sources[k] = v
                else:
                    coerced_sources[k] = str(v)

        try:
            step_plans.append(StepPlan(
                step_id=step_id,
                service_name=service_name,
                intent=step_dict.get("intent", ""),
                depends_on=step_dict.get("depends_on", []),
                input_sources=coerced_sources,
            ))
        except Exception as e:
            errors.append(
                f"Step '{step_id}': failed to create step plan: {str(e)}"
            )

    # Validate dependency references
    for step in step_plans:
        for dep in step.depends_on:
            if dep not in step_ids:
                errors.append(
                    f"Step '{step.step_id}': depends_on '{dep}' "
                    f"which is not a valid step_id"
                )

    if errors:
        return {
            "status": "invalid",
            "errors": errors,
            "hint": (
                "Fix the errors above and call create_workflow_plan again. "
                "Ensure all step_ids are unique, all service_names are valid, "
                "and all depends_on references point to existing step_ids."
            ),
        }

    # Build the plan
    plan = WorkflowPlan(
        workflow_name=workflow_name,
        description=description,
        steps=step_plans,
    )

    # Compute topological order (validates no cycles)
    try:
        topo = plan.compute_topological_order()
    except ValueError as e:
        return {
            "status": "invalid",
            "errors": [str(e)],
            "hint": "The workflow plan contains a cycle. Remove circular dependencies.",
        }

    # Compute independent subgraphs
    subgraphs = plan.compute_independent_subgraphs()

    return {
        "status": "valid",
        "plan": plan.model_dump(),
        "topological_order": topo,
        "independent_subgraphs": subgraphs,
        "step_count": len(step_plans),
        "message": (
            f"Workflow plan '{workflow_name}' validated successfully with "
            f"{len(step_plans)} step(s). "
            f"Build order: {' -> '.join(topo)}. "
            f"Independent subgraphs: {len(subgraphs)}."
        ),
    }
