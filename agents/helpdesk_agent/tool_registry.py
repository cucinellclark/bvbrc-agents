"""
OpenAI-compatible function/tool schema definitions for the Helpdesk Agent.

These schemas tell the LLM what tools are available and how to call them.
The helpdesk agent has three tools:
  1. query_helpdesk  -- Search the BV-BRC helpdesk RAG knowledge base
  2. list_services   -- List available BV-BRC services with descriptions
  3. get_service_schema -- Get parameter schema for a specific service (read-only)
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function calling format)
# ---------------------------------------------------------------------------

QUERY_HELPDESK = {
    "type": "function",
    "function": {
        "name": "query_helpdesk",
        "description": (
            "Search the BV-BRC helpdesk knowledge base (FAQs, tutorials, guides, "
            "and documentation) to find information about how to use BV-BRC features, "
            "services, tools, and workflows. Returns relevant document excerpts with "
            "relevance scores.\n\n"
            "USE THIS TOOL FOR:\n"
            "- How to use BV-BRC services, applications, and workflows\n"
            "- FAQ-style questions about platform features and capabilities\n"
            "- Troubleshooting guidance and parameter explanations\n"
            "- Questions about BV-BRC website pages, tools, and documentation\n"
            "- General 'how does this work?' questions about BV-BRC usage\n"
            "- Understanding what a particular service does and when to use it\n\n"
            "This should be your FIRST tool to call for most questions. Ground "
            "your answers in the retrieved documents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A natural language search query about BV-BRC usage. "
                        "Be specific to get the most relevant results. "
                        "Examples:\n"
                        "  'How do I run genome assembly?'\n"
                        "  'What is the BLAST service?'\n"
                        "  'How to upload files to workspace'\n"
                        "  'What parameters does RNA-seq analysis need?'\n"
                        "  'How to build a phylogenetic tree'"
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": (
                        "Number of top matching documents to return. Default 5. "
                        "Use higher values (8-10) for broad topics, lower (3) "
                        "for specific questions."
                    ),
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}

LIST_SERVICES = {
    "type": "function",
    "function": {
        "name": "list_services",
        "description": (
            "List all available BV-BRC bioinformatics services with their "
            "descriptions. Use this when the user asks what services are "
            "available, or when you need to confirm a service name before "
            "looking up its schema.\n\n"
            "Returns a structured list of service names and descriptions."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

GET_SERVICE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_service_schema",
        "description": (
            "Get the full parameter schema for a specific BV-BRC service. "
            "Use this when the user asks about what parameters a service "
            "accepts, what input formats are required, or how to configure "
            "a specific analysis.\n\n"
            "This is READ-ONLY -- it returns the schema for informational "
            "purposes. It does NOT submit or run any jobs.\n\n"
            "Call list_services first if you are unsure of the exact service name."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "service_name": {
                    "type": "string",
                    "description": (
                        "The name of the BV-BRC service to get the schema for. "
                        "Use the exact service name as returned by list_services. "
                        "Examples: 'genome_assembly', 'genome_annotation', 'blast', "
                        "'comprehensive_genome_analysis', 'rnaseq', 'variation'"
                    ),
                },
            },
            "required": ["service_name"],
        },
    },
}


# ---------------------------------------------------------------------------
# Shared tool schemas (workspace, data, groups, GoWe)
# ---------------------------------------------------------------------------

WORKSPACE_BROWSE = {
    "type": "function",
    "function": {
        "name": "workspace_browse",
        "description": (
            "Browse the user's BV-BRC workspace to find files. Use this "
            "to provide contextual help about their files and workspace."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": ["string", "null"],
                    "description": "Workspace path to browse. Null defaults to home.",
                },
                "name_contains": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": "Filter by filename substrings.",
                },
                "workspace_types": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": (
                        "Filter by workspace type (e.g. 'reads', 'contigs', "
                        "'genbank_file')."
                    ),
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
        "description": "Get metadata for a specific workspace file (type, size, date).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Full workspace path to the file.",
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
            "Read the first portion of a workspace file. Use this to "
            "preview file contents when helping users understand their data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace path to the file.",
                },
                "max_bytes": {
                    "type": "integer",
                    "description": "Maximum bytes to read (default 8192).",
                    "default": 8192,
                },
            },
            "required": ["path"],
        },
    },
}

SEARCH_DATA = {
    "type": "function",
    "function": {
        "name": "search_data",
        "description": (
            "Query BV-BRC Solr collections to look up data. Use this "
            "to provide contextual help about genomes, features, etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": "Solr collection (e.g. 'genome').",
                },
                "query": {
                    "type": "string",
                    "description": "Solr query string.",
                },
                "select": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": "Fields to return.",
                },
                "limit": {
                    "type": ["integer", "null"],
                    "description": "Max records (default 25).",
                },
                "count_only": {
                    "type": ["boolean", "null"],
                    "description": "Return only count if true.",
                },
            },
            "required": ["collection", "query"],
        },
    },
}

GET_GENOME_GROUP = {
    "type": "function",
    "function": {
        "name": "get_genome_group",
        "description": "Get genome IDs from a named genome group in the user's workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "group_name": {
                    "type": "string",
                    "description": "Name of the genome group.",
                },
            },
            "required": ["group_name"],
        },
    },
}

GET_FEATURE_GROUP = {
    "type": "function",
    "function": {
        "name": "get_feature_group",
        "description": "Get feature IDs from a named feature group in the user's workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "group_name": {
                    "type": "string",
                    "description": "Name of the feature group.",
                },
            },
            "required": ["group_name"],
        },
    },
}

LIST_GOWE_WORKFLOWS = {
    "type": "function",
    "function": {
        "name": "list_gowe_workflows",
        "description": (
            "List available GoWe workflows. Use this to tell users "
            "what workflows are available."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

GET_WORKFLOW_INPUTS = {
    "type": "function",
    "function": {
        "name": "get_workflow_inputs",
        "description": "Get the input schema for a GoWe workflow.",
        "parameters": {
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "The GoWe workflow ID.",
                },
            },
            "required": ["workflow_id"],
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
    # Helpdesk-specific tools
    QUERY_HELPDESK,
    LIST_SERVICES,
    GET_SERVICE_SCHEMA,
    # Shared tools
    WORKSPACE_BROWSE,
    GET_FILE_METADATA,
    READ_FILE_PREVIEW,
    SEARCH_DATA,
    # GET_GENOME_GROUP,   # disabled – group tools temporarily removed
    # GET_FEATURE_GROUP,  # disabled – group tools temporarily removed
    LIST_GOWE_WORKFLOWS,
    GET_WORKFLOW_INPUTS,
    FIND_SIMILAR_GENOMES,
    SEARCH_LITERATURE,
]

# Dispatch table: tool name -> schema
TOOL_MAP: dict[str, dict] = {
    schema["function"]["name"]: schema for schema in TOOL_SCHEMAS
}
