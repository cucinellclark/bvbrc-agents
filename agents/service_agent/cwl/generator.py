"""CWL v1.2 workflow generation for BV-BRC service agent.

Pure-function library that transforms validated service step data (Phase 2
output) into CWL v1.2 $graph packed documents suitable for GoWe ingestion.

No LLM involvement, no API calls -- just data transformation.

The generated CWL follows GoWe's conventions:
- $graph packed format with inline CommandLineTool definitions
- gowe:Execution hints with bvbrc_app_id and executor: bvbrc
- Workflow entry with id: "main" that wires steps together
- CWL step_id/output_id syntax for inter-step data flow

Usage:
    from service_agent.cwl.generator import generate_cwl_workflow

    cwl_doc = generate_cwl_workflow(workflow_plan, completed_steps)
    # cwl_doc is a dict, serialize to YAML for GoWe registration
"""

from __future__ import annotations

import re
from typing import Any

from .types import (
    infer_cwl_input_type,
    infer_cwl_type,
    is_workspace_path,
    python_to_cwl_value,
)

# Avoid hard import of models at module level — the models module does
# sys.path manipulation for shared config, which may not be available in
# all test environments.  We import inside functions when we need type
# info, or accept duck-typed objects.

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CWL_VERSION = "v1.2"
GOWE_NAMESPACE = "https://github.com/wilke/GoWe#"

_OUTPUT_OF_PATTERN = re.compile(r"^output_of:(\w+):(\w+)$")


# ---------------------------------------------------------------------------
# Tool definition generation
# ---------------------------------------------------------------------------


def generate_tool_definition(step: Any) -> dict[str, Any]:
    """Generate a CWL CommandLineTool dict for a single validated step.

    Each tool is an inline definition in the $graph with:
    - gowe:Execution hint (bvbrc_app_id from step.api_name, executor: bvbrc)
    - baseCommand set to [api_name]
    - Inputs derived from step.params (types inferred from values)
    - Outputs from step.output_patterns (or generic BV-BRC pattern)

    Args:
        step: A ValidatedStep instance (or duck-typed object with step_id,
              api_name, params, output_patterns attributes).

    Returns:
        CWL CommandLineTool dict.
    """
    tool_id = _step_id_to_tool_id(step.step_id)

    # Build inputs from params
    inputs = _build_tool_inputs(step.params)

    # Build outputs
    outputs = _build_tool_outputs(step.output_patterns)

    tool: dict[str, Any] = {
        "id": tool_id,
        "class": "CommandLineTool",
        "hints": {
            "gowe:Execution": {
                "bvbrc_app_id": step.api_name,
            },
        },
        "baseCommand": [step.api_name],
        "inputs": inputs,
        "outputs": outputs,
    }

    return tool


def _build_tool_inputs(params: dict[str, Any]) -> dict[str, Any]:
    """Build CWL input definitions from parameter dict.

    Each param becomes an input with its type inferred from the value.
    Default values are set for optional params.
    """
    inputs: dict[str, Any] = {}

    for name, value in params.items():
        cwl_type = infer_cwl_input_type(value, param_name=name)
        input_def: dict[str, Any] = {"type": cwl_type}

        # Add default for optional scalar values (not Files/Directories)
        if cwl_type not in ("File", "Directory") and value is not None:
            if isinstance(value, (str, int, float, bool)):
                input_def["default"] = value

        inputs[name] = input_def

    return inputs


def _build_tool_outputs(
    output_patterns: dict[str, str],
) -> dict[str, Any]:
    """Build CWL output definitions from output_patterns.

    If output_patterns is empty or has only generic patterns, use the
    standard BV-BRC generic output pattern.

    Patterns from service_outputs.json use shell-style interpolation
    (``${params.output_path}``, ``${params.output_file}``).  CWL requires
    ``$(inputs.<id>)`` syntax for parameter references in glob expressions.
    This function translates between the two.
    """
    if not output_patterns:
        # Generic BV-BRC output pattern (matches all files produced by the app)
        return {
            "result": {
                "type": "File[]",
                "outputBinding": {
                    "glob": "$(inputs.output_path)/$(inputs.output_file)*",
                },
            },
        }

    outputs: dict[str, Any] = {}
    for output_name, pattern in output_patterns.items():
        # If pattern looks like a workspace path, extract a glob from basename
        if pattern.startswith("/"):
            # Use basename-based glob
            basename = pattern.rsplit("/", 1)[-1]
            # Create a glob from the basename (use * wildcard for extension matching)
            if "." in basename:
                ext = basename.rsplit(".", 1)[-1]
                glob_pattern = f"*.{ext}"
            else:
                glob_pattern = f"{basename}*"
        else:
            glob_pattern = _translate_pattern_to_cwl(pattern)

        outputs[output_name] = {
            "type": "File",
            "outputBinding": {"glob": glob_pattern},
        }

    return outputs


# Regex matching ${params.<name>} shell-style interpolation tokens
_SHELL_PARAM_RE = re.compile(r"\$\{params\.(\w+)\}")


def _translate_pattern_to_cwl(pattern: str) -> str:
    """Translate shell-style ``${params.xxx}`` refs to CWL ``$(inputs.xxx)``.

    The output_path parameter is a CWL Directory, so its value must be
    All parameters (including ``output_path``) are plain scalars and are
    referenced directly as ``$(inputs.<name>)``.

    Examples::

        ${params.output_path}/.${params.output_file}/${params.output_file}.genome
        ->
        $(inputs.output_path)/.$(inputs.output_file)/$(inputs.output_file).genome
    """

    def _replace(m: re.Match) -> str:
        param_name = m.group(1)
        return f"$(inputs.{param_name})"

    return _SHELL_PARAM_RE.sub(_replace, pattern)


# ---------------------------------------------------------------------------
# Step input wiring
# ---------------------------------------------------------------------------


def wire_step_inputs(
    step: Any,
    all_steps: dict[str, Any],
    workflow_inputs: dict[str, str],
) -> dict[str, Any]:
    """Resolve input wiring for a workflow step.

    Maps each parameter to either:
    - A workflow-level input (by name): param_name -> workflow_input_name
    - An upstream step output: output_of:step_id:key -> step_id/output_id

    Parameters whose values are output_of references get wired to
    upstream step outputs using CWL step_id/output_id syntax.
    All other parameters become workflow-level inputs.

    Args:
        step: A ValidatedStep instance.
        all_steps: Dict of {step_id: ValidatedStep} for all steps.
        workflow_inputs: Dict of {workflow_input_name: cwl_type} for
            workflow-level inputs already defined.

    Returns:
        Dict mapping param_name to CWL source (workflow input name or
        step_id/output_id reference).
    """
    wiring: dict[str, Any] = {}

    for param_name, value in step.params.items():
        source = _resolve_param_source(param_name, value, step, all_steps)
        wiring[param_name] = source

    return wiring


def _resolve_param_source(
    param_name: str,
    value: Any,
    step: Any,
    all_steps: dict[str, Any],
) -> Any:
    """Resolve a single parameter to its CWL source.

    Returns either:
    - A string: workflow input name or step_id/output_id reference
    - A dict: {source: ..., default: ...} for values with defaults
    """
    if isinstance(value, str):
        match = _OUTPUT_OF_PATTERN.match(value)
        if match:
            upstream_step_id = match.group(1)
            output_key = match.group(2)
            upstream_cwl_step_id = _step_id_to_workflow_step_id(upstream_step_id)
            return f"{upstream_cwl_step_id}/{output_key}"

    if isinstance(value, list):
        # Check if any item is an output reference
        sources = []
        for item in value:
            if isinstance(item, str):
                match = _OUTPUT_OF_PATTERN.match(item)
                if match:
                    upstream_step_id = match.group(1)
                    output_key = match.group(2)
                    upstream_cwl_step_id = _step_id_to_workflow_step_id(
                        upstream_step_id
                    )
                    sources.append(f"{upstream_cwl_step_id}/{output_key}")
                else:
                    # Mixed list — just use workflow input
                    return _make_workflow_input_name(step.step_id, param_name)
            else:
                return _make_workflow_input_name(step.step_id, param_name)
        if sources:
            # Multiple upstream refs — rare, return first for now
            return sources[0] if len(sources) == 1 else sources

    # Regular value: map to a workflow-level input
    return _make_workflow_input_name(step.step_id, param_name)


# ---------------------------------------------------------------------------
# Workflow generation
# ---------------------------------------------------------------------------


def generate_cwl_workflow(
    workflow_plan: Any,
    completed_steps: dict[str, Any],
    user_id: str | None = None,
) -> dict[str, Any]:
    """Generate a CWL v1.2 $graph packed workflow document.

    Main entry point. Takes a WorkflowPlan (Phase 1) and completed
    ValidatedSteps (Phase 2) and produces a self-contained CWL document.

    The document contains:
    - One CommandLineTool entry per step in $graph
    - One Workflow entry (id: "main") that wires steps together

    Args:
        workflow_plan: WorkflowPlan from Phase 1 (provides topology,
            workflow_name, description, and step ordering).
        completed_steps: Dict of {step_id: ValidatedStep} from Phase 2.
        user_id: Optional user ID for metadata (unused in CWL, but
            available for logging/provenance).

    Returns:
        CWL v1.2 $graph document as a Python dict. Serialize to YAML
        or JSON for GoWe registration.
    """
    # Ensure topological order is computed
    topo_order = workflow_plan.topological_order
    if not topo_order:
        topo_order = workflow_plan.compute_topological_order()

    graph: list[dict[str, Any]] = []

    # Collect workflow-level inputs and step wiring
    workflow_inputs: dict[str, str] = {}  # name -> cwl_type
    step_definitions: list[dict[str, Any]] = []
    workflow_outputs: dict[str, dict[str, Any]] = {}

    for step_id in topo_order:
        step = completed_steps[step_id]

        # 1. Generate inline tool definition
        tool_def = generate_tool_definition(step)
        graph.append(tool_def)

        # 2. Wire step inputs
        wiring = wire_step_inputs(step, completed_steps, workflow_inputs)

        # 3. Collect workflow-level inputs from non-upstream params
        for param_name, source in wiring.items():
            if isinstance(source, str) and "/" not in source:
                # This is a workflow-level input reference
                cwl_type = infer_cwl_input_type(
                    step.params.get(param_name), param_name=param_name
                )
                if source not in workflow_inputs:
                    workflow_inputs[source] = cwl_type

        # 4. Build workflow step entry
        tool_id = _step_id_to_tool_id(step_id)
        cwl_step_id = _step_id_to_workflow_step_id(step_id)

        # Determine outputs for this step
        step_out = list(tool_def["outputs"].keys())

        step_entry: dict[str, Any] = {
            "run": f"#{tool_id}",
            "in": wiring,
            "out": step_out,
        }
        step_definitions.append((cwl_step_id, step_entry))

        # 5. Collect workflow outputs (from final steps — those not depended on)
        # We'll filter after processing all steps

    # Determine which steps are "terminal" (not depended on by any other step)
    all_step_ids = set(topo_order)
    depended_on: set[str] = set()
    for step_id in topo_order:
        step = completed_steps[step_id]
        for dep in step.depends_on:
            depended_on.add(dep)

    terminal_steps = all_step_ids - depended_on

    # Add workflow outputs from terminal steps
    for step_id in topo_order:
        if step_id in terminal_steps:
            step = completed_steps[step_id]
            tool_def = _find_tool_in_graph(graph, step_id)
            cwl_step_id = _step_id_to_workflow_step_id(step_id)
            if tool_def:
                for output_name in tool_def["outputs"]:
                    wf_output_name = (
                        output_name
                        if len(terminal_steps) == 1
                        else f"{cwl_step_id}_{output_name}"
                    )
                    output_type = tool_def["outputs"][output_name].get("type", "File")
                    workflow_outputs[wf_output_name] = {
                        "type": output_type,
                        "outputSource": f"{cwl_step_id}/{output_name}",
                    }

    # Build the Workflow entry
    workflow_entry: dict[str, Any] = {
        "id": "main",
        "class": "Workflow",
        "inputs": {name: cwl_type for name, cwl_type in workflow_inputs.items()},
        "steps": {sid: sdef for sid, sdef in step_definitions},
        "outputs": workflow_outputs,
    }

    graph.append(workflow_entry)

    # Assemble the full CWL document
    cwl_doc: dict[str, Any] = {
        "cwlVersion": CWL_VERSION,
        "$namespaces": {
            "gowe": GOWE_NAMESPACE,
        },
        "$graph": graph,
    }

    return cwl_doc


# ---------------------------------------------------------------------------
# Submission inputs generation
# ---------------------------------------------------------------------------


def generate_submission_inputs(
    completed_steps: dict[str, Any],
    user_id: str | None = None,
) -> dict[str, Any]:
    """Generate the inputs dict for POST /api/v1/submissions.

    Converts parameter values to CWL-typed inputs with proper workspace
    path wrapping. The keys are workflow-level input names (matching the
    "main" Workflow's input definitions).

    Args:
        completed_steps: Dict of {step_id: ValidatedStep} from Phase 2.
        user_id: Optional user ID (unused currently).

    Returns:
        Dict of workflow-level input values, with workspace paths wrapped
        as CWL File/Directory objects.
    """
    inputs: dict[str, Any] = {}

    for step_id, step in completed_steps.items():
        for param_name, value in step.params.items():
            # Skip output_of references — these are wired between steps,
            # not provided as submission inputs
            if isinstance(value, str) and _OUTPUT_OF_PATTERN.match(value):
                continue
            if isinstance(value, list) and any(
                isinstance(v, str) and _OUTPUT_OF_PATTERN.match(v) for v in value
            ):
                continue

            input_name = _make_workflow_input_name(step_id, param_name)
            cwl_value = python_to_cwl_value(value, param_name=param_name)
            inputs[input_name] = cwl_value

    return inputs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _step_id_to_tool_id(step_id: str) -> str:
    """Convert a step_id to a CWL tool ID.

    Convention: "bvbrc-<step_id>" with underscores replaced by hyphens.
    E.g., "assemble" -> "bvbrc-assemble"
    """
    return f"bvbrc-{step_id.replace('_', '-')}"


def _step_id_to_workflow_step_id(step_id: str) -> str:
    """Convert a step_id to a CWL workflow step ID.

    Uses the step_id directly (underscores preserved) since CWL step IDs
    in workflow.steps allow underscores.
    """
    return step_id


def _make_workflow_input_name(step_id: str, param_name: str) -> str:
    """Create a unique workflow-level input name for a step parameter.

    For single-step workflows or common params like output_path/output_file,
    this could be simplified to just param_name. For multi-step workflows,
    prefixing with step_id avoids collisions.

    Strategy: If the param is a common one (output_path, output_file), just
    use the param name (they're typically the same across steps). Otherwise,
    prefix with step_id.
    """
    # Common params shared across steps
    _COMMON_PARAMS = {"output_path", "output_file"}
    if param_name in _COMMON_PARAMS:
        return param_name
    return f"{step_id}_{param_name}"


def _find_tool_in_graph(
    graph: list[dict[str, Any]], step_id: str
) -> dict[str, Any] | None:
    """Find a tool definition in the $graph by step_id."""
    tool_id = _step_id_to_tool_id(step_id)
    for entry in graph:
        if entry.get("id") == tool_id:
            return entry
    return None
