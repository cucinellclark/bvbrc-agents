"""
OpenAI-compatible function/tool schema definitions for the Service Agent.

Tools for the GoWe-first workflow: discover workflows, inspect inputs,
gather context from workspace/data, and submit jobs.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# GoWe workflow tools
# ---------------------------------------------------------------------------

LIST_GOWE_WORKFLOWS = {
    "type": "function",
    "function": {
        "name": "list_gowe_workflows",
        "strict": True,
        "description": (
            "List all available workflows registered in the GoWe workflow "
            "engine. Returns workflow id, name, description, and step count "
            "for each. Call this first to discover which workflow matches "
            "the user's request."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
}

GET_WORKFLOW_INPUTS = {
    "type": "function",
    "function": {
        "name": "get_workflow_inputs",
        "strict": True,
        "description": (
            "Get the full input schema for a specific GoWe workflow. "
            "Returns each input's id, type, required flag, default value, "
            "and documentation. Call this after selecting a workflow to "
            "understand what inputs need to be provided."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": (
                        "The GoWe workflow ID (e.g., 'wf_abc123'). "
                        "Get this from list_gowe_workflows."
                    ),
                },
            },
            "required": ["workflow_id"],
            "additionalProperties": False,
        },
    },
}

SUBMIT_GOWE_JOB = {
    "type": "function",
    "function": {
        "name": "submit_gowe_job",
        "description": (
            "Submit a job to the GoWe workflow engine with populated inputs. "
            "Call this after you have gathered all required input values for "
            "the selected workflow. The inputs dict must match the workflow's "
            "input schema."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "The GoWe workflow ID to run.",
                },
                "inputs": {
                    "type": "object",
                    "description": (
                        "Input values matching the workflow's input schema. "
                        "Include all required inputs and any optional inputs "
                        "you want to override from defaults."
                    ),
                },
            },
            "required": ["workflow_id", "inputs"],
        },
    },
}


# ---------------------------------------------------------------------------
# Context-gathering tools (shared with other agents)
# ---------------------------------------------------------------------------

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
                        "Search term to filter files by name. Null for no filter."
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
                        "Maximum records to return. Max 50. Null defaults to 25."
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

GET_SRA_METADATA = {
    "type": "function",
    "function": {
        "name": "get_sra_metadata",
        "strict": True,
        "description": (
            "Retrieve metadata for one or more SRA run accession IDs (SRR IDs). "
            "Returns organism name, sequencing platform, library strategy, sample "
            "details, and more for each SRA ID. ALWAYS call this tool BEFORE "
            "planning any services when the user provides SRA accessions."
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

FIND_SIMILAR_GENOMES = {
    "type": "function",
    "function": {
        "name": "find_similar_genomes",
        "description": (
            "Find public genomes in BV-BRC that are similar to a query genome "
            "using Mash/MinHash distance estimation. Returns genome IDs ranked "
            "by distance.\n\n"
            "Provide exactly ONE of genome_id or fasta_file (not both).\n\n"
            "USE THIS TOOL FOR:\n"
            "- Finding the closest public genomes to a query genome\n"
            "- Identifying related or similar organisms by genomic distance\n"
            "- Pre-screening reference genomes before downstream analysis\n"
            "- Answering 'what genomes are similar to X?'\n\n"
            "DO NOT USE THIS TOOL FOR:\n"
            "- BLAST sequence similarity searches (use submit_gowe_job with a BLAST workflow)\n"
            "- Phylogenetic tree building (use submit_gowe_job with a tree workflow)\n"
            "- Genome annotation or assembly (use the appropriate workflow tools)"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "genome_id": {
                    "type": ["string", "null"],
                    "description": (
                        "A BV-BRC genome ID to search against (e.g. '83332.12'). "
                        "Mutually exclusive with fasta_file."
                    ),
                },
                "fasta_file": {
                    "type": ["string", "null"],
                    "description": (
                        "FULL workspace path to a FASTA or contigs file "
                        "(e.g. '/user@patricbrc.org/home/my_contigs.fasta'). "
                        "Must be an absolute workspace path starting with /, "
                        "NOT just a filename. Use workspace_browse to find "
                        "the full path first if needed. "
                        "Mutually exclusive with genome_id."
                    ),
                },
                "max_pvalue": {
                    "type": ["number", "null"],
                    "description": (
                        "Maximum p-value threshold (default 0.01). "
                        "Options: 0.001, 0.01, 0.1, 1.0"
                    ),
                },
                "max_distance": {
                    "type": ["number", "null"],
                    "description": (
                        "Maximum Mash distance threshold (default 0.01). "
                        "Lower = more stringent. Options: 0.01, 0.05, 0.1, 0.5, 1.0"
                    ),
                },
                "max_hits": {
                    "type": ["integer", "null"],
                    "description": (
                        "Maximum number of similar genomes to return (default 50). "
                        "Options: 1, 10, 50, 100, 500"
                    ),
                },
                "scope": {
                    "type": ["string", "null"],
                    "description": (
                        "Which public genomes to search. "
                        "'reference' = reference and representative only (default). "
                        "'all' = all public genomes."
                    ),
                },
                "include_bacterial": {
                    "type": ["boolean", "null"],
                    "description": "Include bacterial/archaeal genomes (default true).",
                },
                "include_viral": {
                    "type": ["boolean", "null"],
                    "description": "Include viral genomes (default true).",
                },
            },
            "required": ["genome_id", "fasta_file"],
            "additionalProperties": False,
        },
    },
}

SEARCH_LITERATURE = {
    "type": "function",
    "function": {
        "name": "search_literature",
        "strict": True,
        "description": (
            "Search scientific literature using the literature RAG service and return raw "
            "source passages with bibliographic metadata. Use this when the user asks "
            "for published evidence, papers about an organism/gene, or literature-backed "
            "facts (PPI, mutations, phenotypes)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language query (organism/gene/topic).",
                },
                "top_k": {
                    "type": ["integer", "null"],
                    "description": "Maximum number of sources to return (default 10).",
                },
                "use_graph": {
                    "type": ["boolean", "null"],
                    "description": "Enable knowledge-graph-augmented retrieval (default false).",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# Tool set for the GoWe populate flow (the only active flow)
# ---------------------------------------------------------------------------

POPULATE_TOOLS: list[dict] = [
    LIST_GOWE_WORKFLOWS,
    GET_WORKFLOW_INPUTS,
    SUBMIT_GOWE_JOB,
    WORKSPACE_BROWSE,
    READ_FILE_INFO,
    SEARCH_DATA,
    # GET_GENOME_GROUP,   # disabled – group tools temporarily removed
    # GET_FEATURE_GROUP,  # disabled – group tools temporarily removed
    GET_SRA_METADATA,
    FIND_SIMILAR_GENOMES,
    SEARCH_LITERATURE,
]

# Name -> schema lookup
TOOL_MAP: dict[str, dict] = {
    schema["function"]["name"]: schema for schema in POPULATE_TOOLS
}
