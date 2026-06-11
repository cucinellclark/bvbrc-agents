"""
OpenAI-compatible function/tool schema definitions for the Analysis Agent.

Five tools for read-only output analysis:
  1. workspace_browse    -- List/search files in workspace directories (reused)
  2. get_file_metadata   -- Get detailed metadata for a single file (reused)
  3. read_file_preview   -- Read the first N bytes of a file (reused)
  4. get_expected_outputs -- Look up expected output patterns for a service (new)
  5. get_job_details      -- Query job metadata by task ID (output path, status, params)

The three workspace tools are imported from workspace_agent's tool_registry
to share the same schemas. The get_expected_outputs and get_job_details tools
are local to the analysis agent.
"""

from __future__ import annotations

# Import workspace tool schemas to reuse them
from workspace_agent.tool_registry import (
    WORKSPACE_BROWSE,
    GET_FILE_METADATA,
    READ_FILE_PREVIEW,
)


# ---------------------------------------------------------------------------
# New tool schema: get_expected_outputs
# ---------------------------------------------------------------------------

GET_EXPECTED_OUTPUTS = {
    "type": "function",
    "function": {
        "name": "get_expected_outputs",
        "description": (
            "Look up the expected output file patterns and metric extraction "
            "hints for a BV-BRC service type. Returns file path templates "
            "that can be resolved using output_path and output_file values "
            "from the workflow context. Also returns hints about which "
            "metrics to extract from each file. This is a local lookup "
            "(no API call) -- use it before browsing to know which files "
            "to look for."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "service_name": {
                    "type": "string",
                    "description": (
                        "The BV-BRC service app name (e.g. 'GenomeAssembly2', "
                        "'GenomeAnnotation', 'Homology', 'CodonTree'). "
                        "Use the exact app name from the workflow context."
                    ),
                },
            },
            "required": ["service_name"],
        },
    },
}


# ---------------------------------------------------------------------------
# New tool schema: get_job_details
# ---------------------------------------------------------------------------

GET_JOB_DETAILS = {
    "type": "function",
    "function": {
        "name": "get_job_details",
        "description": (
            "Query BV-BRC job details by task ID. Returns job metadata "
            "including status, app name, parameters (with output_path and "
            "output_file), execution times, and optionally stdout/stderr. "
            "Use this to resolve a task ID into the workspace output path "
            "so you can then browse and analyze the output files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of task IDs to query (e.g. ['22429455']). "
                        "Can be one or more task IDs."
                    ),
                },
                "stdout": {
                    "type": "boolean",
                    "description": (
                        "If true, fetch and include the last 100 lines of "
                        "stdout from the job. Default: false."
                    ),
                },
                "stderr": {
                    "type": "boolean",
                    "description": (
                        "If true, fetch and include the last 100 lines of "
                        "stderr from the job. Useful for debugging failures. "
                        "Default: false."
                    ),
                },
            },
            "required": ["task_ids"],
        },
    },
}


# ---------------------------------------------------------------------------
# Complete tool list for the agent
# ---------------------------------------------------------------------------
TOOL_SCHEMAS: list[dict] = [
    WORKSPACE_BROWSE,
    GET_FILE_METADATA,
    READ_FILE_PREVIEW,
    GET_EXPECTED_OUTPUTS,
    GET_JOB_DETAILS,
]

# Dispatch table: tool name -> schema
TOOL_MAP: dict[str, dict] = {
    schema["function"]["name"]: schema for schema in TOOL_SCHEMAS
}
