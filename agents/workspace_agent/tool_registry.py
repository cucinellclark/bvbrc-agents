"""
OpenAI-compatible function/tool schema definitions for the Workspace Agent.

Three tools for read-only workspace exploration:
  1. workspace_browse  -- List/search files in workspace directories
  2. get_file_metadata -- Get detailed metadata for a single file or folder
  3. read_file_preview -- Read the first N bytes of a file to inspect contents
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Supported workspace file types (for workspace_types enum)
# ---------------------------------------------------------------------------
WORKSPACE_TYPES = [
    "csv",
    "diffexp_input_data",
    "diffexp_input_metadata",
    "doc",
    "docx",
    "embl",
    "feature_dna_fasta",
    "feature_protein_fasta",
    "genbank_file",
    "gff",
    "gif",
    "graph",
    "jpg",
    "json",
    "nwk",
    "pdf",
    "phyloxml",
    "png",
    "pdb",
    "ppt",
    "pptx",
    "reads",
    "string",
    "svg",
    "tar_gz",
    "tbi",
    "tsv",
    "txt",
    "unspecified",
    "vcf",
    "vcf_gz",
    "wig",
    "xls",
    "xlsx",
    "xml",
]


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function calling format)
# ---------------------------------------------------------------------------

WORKSPACE_BROWSE = {
    "type": "function",
    "function": {
        "name": "workspace_browse",
        "description": (
            "Browse and search files in the user's BV-BRC cloud workspace. "
            "Use this to list folder contents, find files by name, type, or "
            "extension. Without filters, lists the immediate contents of the "
            "given path. With filters, performs a recursive search."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Workspace path to browse. Use relative paths like "
                        "'Genome Groups' or 'my_project/results' -- they are "
                        "resolved relative to the user's home directory. "
                        "Leave empty or omit to browse the home directory root."
                    ),
                },
                "name_contains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Literal substrings that must appear in the filename "
                        "(AND logic). ONLY use for actual filename text the "
                        "user specifies (e.g., 'sample1', 'ecoli'). NEVER "
                        "put file type categories here -- use workspace_types "
                        "or file_extensions instead."
                    ),
                },
                "file_extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "File extensions to match (OR logic). Example: "
                        "['fastq', 'fq'] finds .fastq OR .fq files. "
                        "Do NOT combine with workspace_types for the same "
                        "file category."
                    ),
                },
                "workspace_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": WORKSPACE_TYPES,
                    },
                    "description": (
                        "Workspace object types to match (OR logic). This is "
                        "the PREFERRED way to find files by category. Examples: "
                        "'reads' for sequencing data, 'feature_protein_fasta' "
                        "for protein FASTA files, 'gff' for annotations. "
                        "When the user asks for files of a type, use ONLY "
                        "workspace_types -- do not also set name_contains."
                    ),
                },
                "sort_by": {
                    "type": "string",
                    "description": (
                        "Sort field. Valid: creation_time, name, size, type."
                    ),
                    "enum": ["creation_time", "name", "size", "type"],
                },
                "sort_order": {
                    "type": "string",
                    "description": "Sort direction.",
                    "enum": ["asc", "desc"],
                },
                "num_results": {
                    "type": "integer",
                    "description": (
                        "Maximum number of results to return. Default 50."
                    ),
                    "default": 50,
                },
            },
            "required": [],
        },
    },
}

GET_FILE_METADATA = {
    "type": "function",
    "function": {
        "name": "get_file_metadata",
        "description": (
            "Get detailed metadata for a single file or folder in the "
            "workspace. Returns name, type, size, creation time, owner, "
            "and other properties. Use this when you need specific "
            "information about one item, not for listing directories."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Full workspace path to the file or folder. "
                        "Relative paths are resolved from the user's home "
                        "directory."
                    ),
                },
            },
            "required": ["path"],
        },
    },
}

READ_FILE_PREVIEW = {
    "type": "function",
    "function": {
        "name": "read_file_preview",
        "description": (
            "Read the first portion of a workspace file to inspect its "
            "contents. Returns up to max_bytes of the file as text (or "
            "base64 for binary files). Use this to determine file format, "
            "check headers, or preview data contents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Full workspace path to the file to read. Relative "
                        "paths are resolved from the user's home directory."
                    ),
                },
                "max_bytes": {
                    "type": "integer",
                    "description": (
                        "Maximum bytes to read. Default 8192 (8 KB). "
                        "Max 1048576 (1 MB)."
                    ),
                    "default": 8192,
                },
            },
            "required": ["path"],
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
]

# Dispatch table: tool name -> schema
TOOL_MAP: dict[str, dict] = {
    schema["function"]["name"]: schema for schema in TOOL_SCHEMAS
}
