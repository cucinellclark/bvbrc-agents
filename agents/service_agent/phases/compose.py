"""Phase 3: Compose -- resolve a workflow and prepare submission inputs.

This phase is entirely programmatic -- no LLM involvement.

Two distinct composition strategies:

1. **Pre-registered workflows** (``compose_from_registry``):
   Look up an existing workflow in GoWe by name.  Works for both
   single-step workflows (name derived from the service's API name,
   e.g., "GenomeAnnotation") and multi-step pipelines (name derived
   from the combination of services, e.g., "GenomeAssembly to
   GenomeAnnotation").  Maps validated params to the workflow's
   input schema.

2. **Generated CWL workflows** (``compose_from_cwl``):
   Generate a CWL v1.2 $graph document from scratch using the CWL
   generator library, then register it with GoWe.  This path is for
   multi-step pipelines or apps without a pre-registered workflow.

These are independent code paths -- one does NOT fall back to the other.
The caller (agent.py) decides which to use.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from service_agent.models import AgentConfig, AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GoWe workflow name resolution
# ---------------------------------------------------------------------------

# BV-BRC API names sometimes differ from how workflows are registered in
# GoWe (e.g., "GenomeAssembly2" in service_mapping.json, but registered
# as "GenomeAssembly" in GoWe).  This table provides explicit overrides.
# Falls back to stripping trailing digits if no override is found.
_API_NAME_TO_GOWE_NAME: dict[str, str] = {
    "GenomeAssembly2": "GenomeAssembly",
    "ComprehensiveGenomeAnalysis": "ComprehensiveGenomeAnalysis",
}


def _normalize_api_name(api_name: str) -> str:
    """Normalize a BV-BRC API name to its GoWe workflow name.

    Applies explicit overrides first, then strips trailing digits.
    E.g., "GenomeAssembly2" -> "GenomeAssembly"
    """
    override = _API_NAME_TO_GOWE_NAME.get(api_name)
    if override:
        return override
    stripped = re.sub(r"\d+$", "", api_name)
    return stripped or api_name


def _gowe_workflow_name_candidates(api_name: str) -> list[str]:
    """Generate candidate GoWe workflow names from an API name.

    Returns a list of names to try, in priority order:
    1. Exact api_name (may already match)
    2. Explicit override from _API_NAME_TO_GOWE_NAME
    3. api_name with trailing digits stripped (e.g., GenomeAssembly2 -> GenomeAssembly)

    Duplicates are removed while preserving order.
    """
    candidates: list[str] = [api_name]

    # Check explicit override
    override = _API_NAME_TO_GOWE_NAME.get(api_name)
    if override and override not in candidates:
        candidates.append(override)

    # Strip trailing digits (common BV-BRC convention: "FooBar2" -> "FooBar")
    stripped = re.sub(r"\d+$", "", api_name)
    if stripped and stripped != api_name and stripped not in candidates:
        candidates.append(stripped)

    return candidates


def _pipeline_name_candidates(api_names: list[str]) -> list[str]:
    """Generate candidate GoWe workflow names for a multi-step pipeline.

    Given an ordered list of API names (from topological order), generates
    name patterns that GoWe workflows might be registered under.

    For ["GenomeAssembly2", "GenomeAnnotation"] this produces:
      - "GenomeAssembly to GenomeAnnotation"
      - "GenomeAssembly2 to GenomeAnnotation"
      - "genome-assembly-annotation"  (kebab-case)
      - "GenomeAssemblyAnnotation"    (concatenated)
      - "AssemblyAnnotation"          (short form)

    Duplicates are removed while preserving order.
    """
    # Normalize each name (strip trailing digits, apply overrides)
    normalized = [_normalize_api_name(n) for n in api_names]

    candidates: list[str] = []

    def _add(name: str) -> None:
        if name and name not in candidates:
            candidates.append(name)

    # "Name1 to Name2" (most explicit)
    _add(" to ".join(normalized))

    # Same with raw api names
    _add(" to ".join(api_names))

    # kebab-case: "genome-assembly-annotation"
    kebab_parts = []
    for n in normalized:
        # CamelCase to kebab: "GenomeAssembly" -> "genome-assembly"
        kebab = re.sub(r"(?<=[a-z])(?=[A-Z])", "-", n).lower()
        kebab_parts.append(kebab)
    _add("-".join(kebab_parts))

    # Concatenated: "GenomeAssemblyGenomeAnnotation"
    _add("".join(normalized))

    # Short concatenated (drop repeated prefix "Genome"):
    # "GenomeAssembly" + "GenomeAnnotation" -> "GenomeAssemblyAnnotation"
    if len(normalized) == 2:
        # Find common prefix
        a, b = normalized
        # Try dropping the prefix of the second name if it matches the first's prefix
        for prefix_len in range(min(len(a), len(b)), 0, -1):
            if a[:prefix_len] == b[:prefix_len] and b[prefix_len:]:
                _add(a + b[prefix_len:])
                break

    return candidates


# ===================================================================
# Path 1: Pre-registered workflow lookup
# ===================================================================


async def compose_from_registry(
    state: AgentState,
    config: AgentConfig | None = None,
) -> dict[str, Any]:
    """
    Look up a pre-registered GoWe workflow and map validated params.

    For single-step workflows, the API name (e.g., "GenomeAnnotation")
    maps directly to a GoWe workflow name.  The workflow's input schema
    defines how values should be typed (File, Directory, string, etc.).

    Args:
        state: Agent state with completed_steps and workflow_plan.
        config: Agent configuration.

    Returns:
        Dict with keys:
        - workflow_id:        GoWe workflow ID
        - submission_inputs:  Input values ready for GoWe submission
        - workflow_name:      Name of the workflow
        - step_count:         Number of steps
        - source:             "pre_registered"
        - error:              Error message (only on failure)
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
                "error": f"Step '{step_id}' not completed. Cannot compose.",
            }

    if len(state.completed_steps) != 1:
        return {
            "error": (
                "Pre-registered workflow lookup only supports single-step "
                f"workflows. Got {len(state.completed_steps)} steps. "
                "Use compose_from_cwl for multi-step pipelines."
            ),
        }

    step = next(iter(state.completed_steps.values()))

    try:
        _ensure_mcp_path(cfg)
        from common.gowe_client import GoWeClient

        client = GoWeClient(base_url=cfg.gowe_url)
        auth = cfg.bvbrc_auth_token or ""

        # Find workflow by name.  Try the exact API name first, then
        # common variations (e.g., GenomeAssembly2 -> GenomeAssembly).
        wf = None
        candidates = _gowe_workflow_name_candidates(step.api_name)
        for name_candidate in candidates:
            wf = await client.find_workflow_by_name(
                name_candidate,
                auth_token=auth,
            )
            if wf is not None:
                if name_candidate != step.api_name:
                    logger.info(
                        "Matched GoWe workflow '%s' via alias '%s' (API name was '%s')",
                        wf.get("name"),
                        name_candidate,
                        step.api_name,
                    )
                break

        if wf is None:
            tried = ", ".join(f"'{c}'" for c in candidates)
            return {
                "error": (
                    f"No pre-registered workflow found in GoWe for "
                    f"'{step.api_name}' (tried: {tried}). Register the "
                    f"workflow in GoWe first, or use compose_from_cwl "
                    f"to generate one."
                ),
            }

        workflow_id = wf["id"]

        # Get the workflow's input schema for type-aware mapping
        wf_detail = await client.get_workflow(workflow_id, auth_token=auth)
        wf_inputs = wf_detail.get("inputs", [])

        # Build input type and required maps from workflow schema
        input_types = {}
        input_required = {}
        input_defaults = {}
        for inp in wf_inputs:
            inp_id = inp.get("id", "")
            inp_type = inp.get("type", "string")
            input_types[inp_id] = inp_type
            # A param is required if its type doesn't end with '?' and
            # the schema says required=True (or omits required, defaulting
            # to True for non-optional types)
            is_optional = inp_type.endswith("?")
            input_required[inp_id] = inp.get("required", not is_optional)
            if "default" in inp:
                input_defaults[inp_id] = inp["default"]

        # Map step params to GoWe submission inputs
        submission_inputs = _map_params_to_inputs(
            step.params,
            input_types,
            input_required,
            input_defaults,
        )

        # Check for missing required inputs
        missing = []
        for inp_id, required in input_required.items():
            if required and inp_id not in submission_inputs:
                missing.append(inp_id)
        if missing:
            return {
                "error": (
                    f"Missing required inputs for workflow "
                    f"'{step.api_name}': {', '.join(missing)}. "
                    f"Please provide values for these parameters."
                ),
            }

        logger.info(
            "Using pre-registered workflow '%s' (%s) with %d inputs",
            step.api_name,
            workflow_id,
            len(submission_inputs),
        )

        return {
            "workflow_id": workflow_id,
            "submission_inputs": submission_inputs,
            "workflow_name": (state.workflow_plan.workflow_name or step.api_name),
            "step_count": 1,
            "source": "pre_registered",
        }

    except Exception as e:
        return {
            "error": (
                f"Failed to look up pre-registered workflow "
                f"'{step.api_name}': {type(e).__name__}: {e}"
            ),
        }


def _map_params_to_inputs(
    params: dict[str, Any],
    input_types: dict[str, str],
    input_required: dict[str, bool] | None = None,
    input_defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Map validated step params to GoWe submission input format.

    Uses the workflow's input type definitions to wrap values correctly:
    - File-typed inputs with workspace paths -> CWL File objects
    - Directory-typed inputs with workspace paths -> CWL Directory objects
    - Record array types (e.g., paired_end_libs) -> pass through with
      workspace paths wrapped in nested fields
    - Scalar values pass through as-is

    Only includes params that exist in the workflow's input schema.
    Skips None values.  For required params missing from the agent's
    output, fills in the workflow schema default if one exists.
    """
    from service_agent.cwl.types import is_workspace_path, wrap_workspace_path

    input_required = input_required or {}
    input_defaults = input_defaults or {}
    inputs: dict[str, Any] = {}

    for param_name, value in params.items():
        cwl_type = input_types.get(param_name)

        if cwl_type is None:
            logger.debug(
                "Skipping param '%s' -- not in workflow input schema",
                param_name,
            )
            continue

        # Skip None values -- they're unset params
        if value is None:
            continue

        # Strip spurious URI scheme prefixes the LLM sometimes adds
        # (e.g., "workspace:/user@bvbrc/..." -> "/user@bvbrc/...")
        # Strip "ws://" (5 chars) so the path's leading "/" is preserved.
        if isinstance(value, str):
            if value.startswith("ws://"):
                value = value[len("ws://") :]
            elif value.startswith("workspace:"):
                value = value[len("workspace:") :]

        # Strip optional marker for type matching
        base_type = cwl_type.rstrip("?").strip()

        if base_type == "File" and isinstance(value, str):
            if is_workspace_path(value):
                inputs[param_name] = wrap_workspace_path(
                    value,
                    as_type="File",
                )
            else:
                inputs[param_name] = value
        elif base_type == "Directory" and isinstance(value, str):
            if is_workspace_path(value):
                inputs[param_name] = wrap_workspace_path(
                    value,
                    as_type="Directory",
                )
            else:
                inputs[param_name] = value
        elif isinstance(value, dict) and "class" in value:
            # Already a CWL-typed object
            inputs[param_name] = value
        elif _is_record_array_type(base_type):
            # Complex record array types (e.g., "record:paired_end_lib[]").
            # These are arrays of record/struct objects (like paired_end_libs).
            # Fields inside records (read1, read2, etc.) must remain as plain
            # workspace path strings -- the BV-BRC app service resolves them
            # internally.  Only strip spurious URI prefixes the LLM may add.
            inputs[param_name] = _clean_record_values(value)
        else:
            # Scalar value -- pass through
            inputs[param_name] = value

    # Fill in defaults for required params the agent didn't provide
    for param_name, required in input_required.items():
        if param_name not in inputs and required:
            if param_name in input_defaults:
                inputs[param_name] = input_defaults[param_name]
                logger.debug(
                    "Using schema default for required param '%s': %s",
                    param_name,
                    input_defaults[param_name],
                )

    return inputs


def _is_record_array_type(cwl_type: str) -> bool:
    """Check if a CWL type string represents a record array.

    GoWe reports complex types like: "record:paired_end_lib[]"
    """
    return cwl_type.startswith("record:") or "record" in cwl_type.lower()


def _clean_record_values(value: Any) -> Any:
    """Clean string values inside record structures.

    For record array types like paired_end_libs, the BV-BRC app service
    expects fields (read1, read2, etc.) as **plain workspace path strings**,
    NOT as CWL File objects.  GoWe stages the parent record as a whole and
    the app resolves paths internally.

    This function only strips spurious URI prefixes the LLM sometimes adds
    (e.g., "workspace:/user@bvbrc/..." -> "/user@bvbrc/...") and unwraps
    any accidental CWL File objects back to plain paths.
    """
    if isinstance(value, list):
        return [_clean_record_values(item) for item in value]

    if isinstance(value, dict):
        # If the LLM or an earlier step accidentally wrapped a path as a
        # CWL File/Directory object, unwrap it back to a plain path string.
        if "class" in value and "location" in value:
            location = value["location"]
            # Strip URI scheme to get the bare workspace path.
            # ws:// + /user@bvbrc/... = ws:///user@bvbrc/...
            # Strip "ws://" (5 chars) to preserve the path's leading "/".
            if location.startswith("ws://"):
                return location[len("ws://") :]
            if location.startswith("file://"):
                return location[len("file://") :]
            return location

        result = {}
        for k, v in value.items():
            if isinstance(v, str):
                # Strip spurious URI prefixes the LLM sometimes adds.
                # "ws://" + "/path" = "ws:///path" -> strip "ws://" to get "/path"
                if v.startswith("ws://"):
                    v = v[len("ws://") :]
                elif v.startswith("workspace:"):
                    v = v[len("workspace:") :]
                result[k] = v
            else:
                result[k] = _clean_record_values(v)
        return result

    return value


# ===================================================================
# Path 2: CWL generation
# ===================================================================


def compose_from_cwl(
    state: AgentState,
    config: AgentConfig | None = None,
) -> dict[str, Any]:
    """
    Generate a CWL v1.2 $graph workflow document from validated steps.

    Produces a self-contained CWL document suitable for registration
    with GoWe.  Supports single-step and multi-step DAG workflows.

    This path is for:
    - Multi-step pipelines with inter-step data flow
    - Apps that don't have a pre-registered workflow in GoWe
    - Custom/ad-hoc workflow compositions

    The generated CWL must be registered with GoWe separately before
    submission (the caller is responsible for that).

    Args:
        state: Agent state with completed_steps and workflow_plan.
        config: Agent configuration.

    Returns:
        Dict with keys:
        - cwl_document:       CWL v1.2 $graph document (dict)
        - submission_inputs:  CWL-typed input values (dict)
        - workflow_name:      Name of the workflow
        - step_count:         Number of steps
        - source:             "generated"
        - error:              Error message (only on failure)
    """
    from service_agent.cwl.generator import (
        generate_cwl_workflow,
        generate_submission_inputs,
    )

    if not state.workflow_plan:
        return {"error": "No workflow plan available for composition."}
    if not state.completed_steps:
        return {"error": "No completed steps available for composition."}

    cfg = config or AgentConfig()

    # Verify all planned steps are completed
    for step_id in state.workflow_plan.topological_order:
        if step_id not in state.completed_steps:
            return {
                "error": f"Step '{step_id}' not completed. Cannot compose.",
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
        cwl_doc = generate_cwl_workflow(
            state.workflow_plan,
            state.completed_steps,
            user_id=user_id,
        )

        submission_inputs = generate_submission_inputs(
            state.completed_steps,
            user_id=user_id,
        )

        return {
            "cwl_document": cwl_doc,
            "submission_inputs": submission_inputs,
            "workflow_name": state.workflow_plan.workflow_name,
            "step_count": len(state.workflow_plan.steps),
            "source": "generated",
        }

    except Exception as e:
        return {
            "error": (f"CWL workflow generation failed: {type(e).__name__}: {e}"),
        }


# ===================================================================
# Legacy entry point (for backward compatibility during transition)
# ===================================================================


async def compose_manifest(
    state: AgentState,
    config: AgentConfig | None = None,
) -> dict[str, Any]:
    """
    Compose a workflow manifest -- dispatches to the appropriate path.

    Currently defaults to the pre-registered path for single-step
    workflows.  Multi-step workflows use the CWL generation path.

    This function exists for backward compatibility. New code should
    call compose_from_registry or compose_from_cwl directly.
    """
    if len(state.completed_steps) == 1:
        return await compose_from_registry(state, config)
    else:
        return compose_from_cwl(state, config)


# ===================================================================
# Helpers
# ===================================================================


def _ensure_mcp_path(config: AgentConfig) -> None:
    """Add the MCP server path to sys.path if needed."""
    import sys

    mcp_path = config.mcp_server_path
    if mcp_path and mcp_path not in sys.path:
        sys.path.insert(0, mcp_path)
