"""CWL v1.2 generation library for BV-BRC service agent.

Produces CWL $graph packed documents from validated service step data
for ingestion by GoWe (CWL v1.2 workflow engine).

Public API:
    generate_cwl_workflow     - Main entry point: steps -> CWL document
    generate_tool_definition  - Single step -> CommandLineTool dict
    wire_step_inputs          - Resolve inter-step data wiring
    generate_submission_inputs - Steps -> submission input values dict
"""

from .generator import (
    generate_cwl_workflow,
    generate_submission_inputs,
    generate_tool_definition,
    wire_step_inputs,
)
from .types import (
    infer_cwl_type,
    is_workspace_path,
    python_to_cwl_value,
    wrap_workspace_path,
)

__all__ = [
    "generate_cwl_workflow",
    "generate_submission_inputs",
    "generate_tool_definition",
    "wire_step_inputs",
    "infer_cwl_type",
    "is_workspace_path",
    "python_to_cwl_value",
    "wrap_workspace_path",
]
