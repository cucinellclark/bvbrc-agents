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
            "Query BV-BRC job details by task ID. Returns status, parameters "
            "(including output paths), and optionally stdout/stderr logs."
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

SEARCH_LITERATURE = {
    "type": "function",
    "function": {
        "name": "search_literature",
        "description": "Search scientific literature using the literature RAG service and return raw source passages with metadata.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language query (organism/gene/topic).",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of sources to return (default 10).",
                },
                "use_graph": {
                    "type": "boolean",
                    "description": "Enable knowledge-graph-augmented retrieval (default false).",
                },
            },
            "required": ["query"],
        },
    },
}


FIND_SIMILAR_GENOMES = {
    "type": "function",
    "function": {
        "name": "find_similar_genomes",
        "description": (
            "Find public BV-BRC genomes similar to a query genome using "
            "Mash/MinHash distance estimation. Provide a genome_id OR a "
            "workspace fasta_file path (not both). Returns genome IDs "
            "ranked by genomic distance."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "genome_id": {
                    "type": "string",
                    "description": "A BV-BRC genome ID (e.g. '83332.12').",
                },
                "fasta_file": {
                    "type": "string",
                    "description": "FULL workspace path to a FASTA/contigs file (e.g. '/user@patricbrc.org/home/file.fasta'). Must start with /.",
                },
                "max_pvalue": {"type": "number", "description": "Max p-value (default 0.01)."},
                "max_distance": {"type": "number", "description": "Max Mash distance (default 0.01)."},
                "max_hits": {"type": "integer", "description": "Max results (default 50)."},
                "scope": {"type": "string", "description": "'reference' or 'all' (default 'reference')."},
                "include_bacterial": {"type": "boolean", "description": "Include bacterial genomes (default true)."},
                "include_viral": {"type": "boolean", "description": "Include viral genomes (default true)."},
            },
            "required": ["genome_id"],
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
    FIND_SIMILAR_GENOMES,
    SEARCH_LITERATURE,
]

# Dispatch table: tool name -> schema
TOOL_MAP: dict[str, dict] = {
    schema["function"]["name"]: schema for schema in TOOL_SCHEMAS
}
