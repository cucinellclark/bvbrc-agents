"""Phase 2 (Step Building) system prompt for the Service Agent v2.

This prompt is templated per step -- it tells the LLM to focus on gathering
inputs and validating parameters for one specific step of the workflow.
"""

from __future__ import annotations

import json
from typing import Any


def build_phase2_prompt(
    step_id: str,
    service_name: str,
    intent: str,
    depends_on: list[str],
    input_sources: dict[str, str],
    upstream_outputs: dict[str, dict[str, str]],
) -> str:
    """Build the Phase 2 system prompt for building a single step.

    Args:
        step_id: The step being built.
        service_name: BV-BRC service name.
        intent: What this step accomplishes.
        depends_on: Step IDs this step depends on.
        input_sources: param_name -> source description from the plan.
        upstream_outputs: {dep_step_id: {output_key: output_path, ...}, ...}
    """
    # Format upstream outputs section
    upstream_section = "None (this is a root step)"
    if upstream_outputs:
        lines = []
        for dep_id, outputs in upstream_outputs.items():
            lines.append(f"  Step '{dep_id}':")
            for key, path in outputs.items():
                lines.append(f"    {key}: {path}")
        upstream_section = "\n".join(lines)

    # Format input sources section
    sources_section = "None specified"
    if input_sources:
        lines = []
        for param, source in input_sources.items():
            lines.append(f"  {param}: {source}")
        sources_section = "\n".join(lines)

    return f"""\
You are building step "{step_id}" of a workflow plan.

== STEP DETAILS ==
- Step ID: {step_id}
- Service: {service_name}
- Intent: {intent}
- Dependencies: {', '.join(depends_on) if depends_on else 'none (root step)'}

== AVAILABLE UPSTREAM OUTPUTS ==
{upstream_section}

== INPUT SOURCES (from plan) ==
{sources_section}

== INSTRUCTIONS ==
1. Call get_service_schema("{service_name}") to get the parameter requirements \
for this service.
2. Gather any missing inputs:
   - For "output_of:X:Y" sources, use the upstream output path provided above. \
Use the format "output_of:<step_id>:<output_key>" as the parameter value -- \
it will be resolved to the actual path in the composition phase.
   - For "search:..." sources, use search_data or get_genome_group to find \
the needed data.
   - For "workspace:..." sources, use workspace_browse to find the file.
   - For "user_provided" sources, use the values from the user's original \
request (included below).
3. Call plan_service with the gathered parameters to validate the step.
4. If validation fails, read the error hints and fix the parameters, then retry.
5. If you cannot determine a required parameter, ask the user by providing \
a clear question in your text response (do NOT call any tool).

== RULES ==
- Focus ONLY on this step. Do not plan other steps.
- Always call get_service_schema before plan_service.
- If a parameter cannot be determined, describe what you need in your text \
response. Do NOT guess values for required parameters.
- For parameters sourced from upstream steps ("output_of:..."), use the \
string "output_of:<step_id>:<output_key>" as the parameter value. These \
will be resolved to concrete paths during composition.
- output_path and output_file will be auto-generated if not provided -- \
you do not need to specify them unless the user has a preference.
"""
