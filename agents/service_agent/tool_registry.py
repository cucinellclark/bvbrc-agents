"""
OpenAI-compatible function/tool schema definitions for the Service Agent v2.

Tools are organized by phase:
  - Phase 1 (Decompose): create_workflow_plan, list_services, get_sra_metadata
  - Phase 2 (Build):     get_service_schema, plan_service, workspace_browse,
                          read_file_info, search_data, get_genome_group,
                          get_feature_group, get_sra_metadata
  - Phase 3 (Compose):   Programmatic -- no LLM tools needed
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Service name enum -- all friendly service names from service_mapping.json
# ---------------------------------------------------------------------------
SERVICE_NAMES = [
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
]


# ---------------------------------------------------------------------------
# Phase 1 tools
# ---------------------------------------------------------------------------

CREATE_WORKFLOW_PLAN = {
    "type": "function",
    "function": {
        "name": "create_workflow_plan",
        "strict": True,
        "description": (
            "Create a structured workflow plan (DAG) from a decomposed analysis "
            "request. Each step specifies a BV-BRC service, its intent, "
            "dependencies on other steps, and input sources. The plan is "
            "validated for unique step IDs, valid service names, valid "
            "dependency references, and absence of cycles. Call this after you "
            "have determined all the steps needed for the workflow."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "workflow_name": {
                    "type": "string",
                    "description": (
                        "Short descriptive name for the workflow "
                        "(e.g., 'ecoli-assembly-annotation')."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": "Brief description of what the workflow accomplishes.",
                },
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step_id": {
                                "type": "string",
                                "description": (
                                    "Unique identifier for this step "
                                    "(short, descriptive, snake_case)."
                                ),
                            },
                            "service_name": {
                                "type": "string",
                                "description": "BV-BRC service name.",
                                "enum": SERVICE_NAMES,
                            },
                            "intent": {
                                "type": "string",
                                "description": (
                                    "What this step accomplishes "
                                    "(e.g., 'Assemble E. coli reads into contigs')."
                                ),
                            },
                            "depends_on": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Step IDs this step depends on. "
                                    "Empty array for root steps."
                                ),
                            },
                            "input_sources": {
                                "type": "array",
                                "description": (
                                    "Parameter-to-source mappings. Each entry "
                                    "maps a parameter name to its source. "
                                    "Source values: 'user_provided', "
                                    "'output_of:<step_id>:<output_key>', "
                                    "'search:<description>', "
                                    "'workspace:<path_hint>'. "
                                    "Empty array if no sources to declare."
                                ),
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "param_name": {
                                            "type": "string",
                                            "description": "Parameter name.",
                                        },
                                        "source": {
                                            "type": "string",
                                            "description": (
                                                "Where this parameter's value "
                                                "comes from."
                                            ),
                                        },
                                    },
                                    "required": ["param_name", "source"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": [
                            "step_id", "service_name", "intent",
                            "depends_on", "input_sources",
                        ],
                        "additionalProperties": False,
                    },
                    "description": "List of workflow steps forming a DAG.",
                },
            },
            "required": ["workflow_name", "description", "steps"],
            "additionalProperties": False,
        },
    },
}

LIST_SERVICES = {
    "type": "function",
    "function": {
        "name": "list_services",
        "strict": True,
        "description": (
            "List all available BV-BRC services with short descriptions. "
            "Use this when you need to discover which service to use for "
            "a particular analysis task."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
}

GET_SRA_METADATA = {
    "type": "function",
    "function": {
        "name": "get_sra_metadata",
        "strict": True,
        "description": (
            "Retrieve metadata for one or more SRA run accession IDs (SRR IDs). "
            "Returns organism name, sequencing platform, library strategy, sample "
            "details, and more for each SRA ID. ALWAYS call this tool BEFORE "
            "planning any services when the user provides SRA accessions. This "
            "lets you verify the organism, check for mismatched samples, and "
            "auto-fill parameters like scientific_name."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sra_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of SRA run accession IDs to look up "
                        "(e.g., ['SRR37956035', 'SRR37956031'])."
                    ),
                },
            },
            "required": ["sra_ids"],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# Phase 2 tools
# ---------------------------------------------------------------------------

GET_SERVICE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_service_schema",
        "strict": True,
        "description": (
            "Get the full parameter schema for a specific BV-BRC service, "
            "including required parameters, defaults, enum constraints, and "
            "conditional requirements. Call this BEFORE plan_service to "
            "understand what parameters are needed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "service_name": {
                    "type": "string",
                    "description": "Friendly service name (e.g., 'genome_assembly', 'blast').",
                    "enum": SERVICE_NAMES,
                },
            },
            "required": ["service_name"],
            "additionalProperties": False,
        },
    },
}

PLAN_SERVICE = {
    "type": "function",
    "function": {
        "name": "plan_service",
        "description": (
            "Validate and plan a single BV-BRC service job. Checks required "
            "parameters, validates enum values (with fuzzy matching), applies "
            "defaults, and returns the validated parameter set ready for "
            "submission. Call get_service_schema first to understand the "
            "required parameters."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "service_name": {
                    "type": "string",
                    "description": "Friendly service name (e.g., 'genome_assembly', 'blast').",
                    "enum": SERVICE_NAMES,
                },
                "params": {
                    "type": "object",
                    "description": (
                        "Service parameters. Include all required params plus any "
                        "optional params you want to override. Missing optional "
                        "params get defaults applied automatically."
                    ),
                },
            },
            "required": ["service_name", "params"],
        },
    },
}

WORKSPACE_BROWSE = {
    "type": "function",
    "function": {
        "name": "workspace_browse",
        "strict": True,
        "description": (
            "Search and list files in the user's BV-BRC workspace. Use this "
            "to find input files (reads, contigs, FASTA files) for service "
            "parameters."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": ["string", "null"],
                    "description": (
                        "Workspace path to browse (e.g., '/username/home'). "
                        "Null defaults to user's home directory."
                    ),
                },
                "type_filter": {
                    "type": ["string", "null"],
                    "description": (
                        "Filter by file type. Examples: 'reads', 'contigs', "
                        "'genome_group', 'feature_group', 'unspecified'. "
                        "Null for no filter."
                    ),
                },
                "search": {
                    "type": ["string", "null"],
                    "description": (
                        "Search term to filter files by name. "
                        "Null for no filter."
                    ),
                },
            },
            "required": ["path", "type_filter", "search"],
            "additionalProperties": False,
        },
    },
}

READ_FILE_INFO = {
    "type": "function",
    "function": {
        "name": "read_file_info",
        "strict": True,
        "description": (
            "Get metadata about a workspace file (size, type, creation date). "
            "Use this to verify a file is the right type before using it as a "
            "service input."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Full workspace path to the file.",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}

SEARCH_DATA = {
    "type": "function",
    "function": {
        "name": "search_data",
        "strict": True,
        "description": (
            "Query BV-BRC Solr collections to find genome IDs, feature IDs, "
            "or other data. Use this when you need to resolve organism names "
            "to genome IDs for service parameters."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": "The Solr collection to query (e.g., 'genome').",
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Solr query string. Examples:\n"
                        "  genus:Salmonella AND host_name:Human\n"
                        "  genome_id:(83332.12 OR 208964.12)\n"
                        "  taxon_lineage_ids:1763\n"
                        "  genome_name:*Escherichia*"
                    ),
                },
                "select": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": (
                        "Fields to return. Only request fields you need. "
                        "Null returns default fields."
                    ),
                },
                "limit": {
                    "type": ["integer", "null"],
                    "description": (
                        "Maximum records to return. Max 50. "
                        "Null defaults to 25."
                    ),
                },
                "count_only": {
                    "type": ["boolean", "null"],
                    "description": (
                        "If true, return only the count of matching records. "
                        "Null defaults to false."
                    ),
                },
            },
            "required": ["collection", "query", "select", "limit", "count_only"],
            "additionalProperties": False,
        },
    },
}

GET_GENOME_GROUP = {
    "type": "function",
    "function": {
        "name": "get_genome_group",
        "strict": True,
        "description": (
            "Retrieve genome IDs from a named genome group in the user's "
            "workspace. Use this when the user refers to 'my genomes' or a "
            "named group."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "group_name": {
                    "type": "string",
                    "description": "The name of the genome group (fuzzy matched).",
                },
            },
            "required": ["group_name"],
            "additionalProperties": False,
        },
    },
}

GET_FEATURE_GROUP = {
    "type": "function",
    "function": {
        "name": "get_feature_group",
        "strict": True,
        "description": (
            "Retrieve feature IDs from a named feature group in the user's "
            "workspace. Use this when the user refers to a named feature group."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "group_name": {
                    "type": "string",
                    "description": "The name of the feature group (fuzzy matched).",
                },
            },
            "required": ["group_name"],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# Submission tool (available in Phase 1 for submit-by-id requests)
# ---------------------------------------------------------------------------

SUBMIT_WORKFLOW = {
    "type": "function",
    "function": {
        "name": "submit_workflow",
        "description": (
            "Submit an already-planned workflow for execution by its workflow_id. "
            "Only use this when the user explicitly asks to submit/run/execute "
            "a planned workflow. The workflow must have been previously planned "
            "and persisted to the engine."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "The engine-issued workflow ID (e.g. 'wf_abc123').",
                },
            },
            "required": ["workflow_id"],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# Tool sets organized by phase
# ---------------------------------------------------------------------------

PHASE_1_TOOLS: list[dict] = [
    CREATE_WORKFLOW_PLAN,
    LIST_SERVICES,
    GET_SRA_METADATA,
    SUBMIT_WORKFLOW,
]

PHASE_2_TOOLS: list[dict] = [
    GET_SERVICE_SCHEMA,
    PLAN_SERVICE,
    WORKSPACE_BROWSE,
    READ_FILE_INFO,
    SEARCH_DATA,
    GET_GENOME_GROUP,
    GET_FEATURE_GROUP,
    GET_SRA_METADATA,  # Also available in Phase 2
]

# All tools (for reference / backwards compatibility)
ALL_TOOL_SCHEMAS: list[dict] = [
    CREATE_WORKFLOW_PLAN,
    LIST_SERVICES,
    GET_SERVICE_SCHEMA,
    PLAN_SERVICE,
    WORKSPACE_BROWSE,
    READ_FILE_INFO,
    SEARCH_DATA,
    GET_GENOME_GROUP,
    GET_FEATURE_GROUP,
    GET_SRA_METADATA,
]

# Name -> schema lookup
TOOL_MAP: dict[str, dict] = {
    schema["function"]["name"]: schema for schema in ALL_TOOL_SCHEMAS
}
