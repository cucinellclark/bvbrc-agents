"""
Workflow composition functions for the Service Agent.

Composes multiple planned service steps into a single workflow manifest
with dependency chains. This is a deterministic alternative to the
LLM-based workflow generation in workflow_functions.py.
"""

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from functions.service_validation_functions import (
    _load_config_file,
    _default_output,
    _output_patterns,
)


def compose_workflow_manifest(
    steps: List[Dict[str, Any]],
    user_id: str,
    workflow_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compose multiple planned service steps into a single workflow manifest.

    Each step should have:
        - step_name: str - unique name for this step
        - service_name: str - friendly service name
        - params: dict - validated service parameters
        - depends_on: list[str] - step names this step depends on (optional)

    Args:
        steps: List of planned steps.
        user_id: BV-BRC user ID for workspace paths.
        workflow_name: Optional workflow name (auto-generated if omitted).

    Returns:
        Dict with workflow_id, status, manifest, step_count, etc.
    """
    if not steps:
        return {
            "error": "No steps provided for workflow composition.",
            "status": "validation_failed",
        }

    # Load service mapping
    try:
        mapping_data = _load_config_file('service_mapping.json')
        service_mapping = mapping_data.get('friendly_to_api', mapping_data)
    except Exception:
        service_mapping = {}

    # Validate step names are unique
    step_names = [s.get("step_name", f"step_{i}") for i, s in enumerate(steps)]
    if len(set(step_names)) != len(step_names):
        return {
            "error": "Step names must be unique.",
            "status": "validation_failed",
            "step_names": step_names,
        }

    # Build step name set for dependency validation
    step_name_set = set(step_names)

    # Validate dependencies (DAG check)
    errors = []
    for step in steps:
        step_name = step.get("step_name", "unknown")
        depends_on = step.get("depends_on", [])
        for dep in depends_on:
            if dep not in step_name_set:
                errors.append(
                    f"Step '{step_name}' depends on '{dep}' which does not exist."
                )
            if dep == step_name:
                errors.append(
                    f"Step '{step_name}' cannot depend on itself."
                )

    # Simple cycle detection using topological sort
    if not errors:
        visited = set()
        in_progress = set()

        def has_cycle(name: str, dep_map: Dict[str, List[str]]) -> bool:
            if name in in_progress:
                return True
            if name in visited:
                return False
            in_progress.add(name)
            for dep in dep_map.get(name, []):
                if has_cycle(dep, dep_map):
                    return True
            in_progress.discard(name)
            visited.add(name)
            return False

        dep_map = {
            s.get("step_name", f"step_{i}"): s.get("depends_on", [])
            for i, s in enumerate(steps)
        }
        for sn in step_names:
            if has_cycle(sn, dep_map):
                errors.append("Dependency cycle detected in workflow steps.")
                break

    if errors:
        return {
            "error": "; ".join(errors),
            "status": "validation_failed",
        }

    # Generate workflow name if not provided
    if not workflow_name:
        svc_names = [s.get("service_name", "svc") for s in steps]
        workflow_name = "-".join(svc_names[:3]) + f"-{time.strftime('%Y%m%d-%H%M%S')}"

    # Build the manifest
    manifest_steps = []
    all_outputs = []

    for i, step in enumerate(steps):
        step_name = step.get("step_name", f"step_{i}")
        service_name = step.get("service_name", "unknown")
        params = step.get("params", {})
        depends_on = step.get("depends_on", [])

        api_name = service_mapping.get(service_name, service_name)

        # Ensure output_path and output_file are set
        output_path = params.get("output_path")
        output_file = params.get("output_file")
        output_path, output_file = _default_output(
            user_id, api_name, output_path, output_file
        )
        params["output_path"] = output_path
        params["output_file"] = output_file

        # Get output patterns for this service
        outputs = _output_patterns(api_name)

        manifest_step = {
            "step_name": step_name,
            "app": api_name,
            "params": params,
            "outputs": outputs,
            "depends_on": depends_on,
        }
        manifest_steps.append(manifest_step)
        all_outputs.extend(outputs.values())

    manifest = {
        "workflow_name": workflow_name,
        "version": "1.0",
        "base_context": {
            "base_url": "https://www.bv-brc.org",
            "workspace_output_folder": f"/{user_id}/home",
        },
        "steps": manifest_steps,
        "workflow_outputs": all_outputs,
    }

    # Assign a local workflow ID
    workflow_id = f"wf_planned_{uuid.uuid4().hex[:12]}"

    return {
        "workflow_id": workflow_id,
        "status": "planned",
        "workflow_name": workflow_name,
        "step_count": len(steps),
        "manifest": manifest,
        "steps_summary": [
            {
                "step_name": s.get("step_name"),
                "service_name": s.get("service_name"),
                "api_name": service_mapping.get(s.get("service_name", ""), s.get("service_name", "")),
                "depends_on": s.get("depends_on", []),
            }
            for s in steps
        ],
    }
