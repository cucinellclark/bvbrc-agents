"""Canonical OpenAI function-calling tool schemas for all BV-BRC agent tools.

This is the SINGLE SOURCE OF TRUTH for tool schemas. Every agent uses these
definitions. No per-agent schema variants.

26 tools total:
  Data:      search_data, facet_query, probe_data, list_collections, get_collection_fields
  Workspace: workspace_browse, get_file_metadata, read_file_preview
  GoWe:      list_gowe_workflows, get_workflow_inputs, submit_gowe_job
  Groups:    create_group, list_groups, get_group_ids
  SRA:       get_sra_metadata
  Genome:    find_similar_genomes
  Literature: search_literature
  Helpdesk:  query_helpdesk, list_services, get_service_schema
  Analysis:  get_expected_outputs, get_job_details, list_jobs
  Planning:  ask_clarification, create_plan, list_agents
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Enums / constants
# ---------------------------------------------------------------------------

COLLECTIONS = [
    "antibiotics",
    "bioset",
    "bioset_results",
    "enzyme_class_ref",
    "epitope",
    "epitope_assay",
    "experiment",
    "feature_sequence",
    "gene_ontology_ref",
    "genome",
    "genome_amr",
    "genome_feature",
    "genome_sequence",
    "id_ref",
    "misc_niaid_sgc",
    "pathway",
    "pathway_ref",
    "ppi",
    "protein_family_ref",
    "protein_feature",
    "protein_structure",
    "serology",
    "sequence_feature",
    "sequence_feature_vt",
    "sp_gene",
    "sp_gene_ref",
    "spike_lineage",
    "spike_variant",
    "strain",
    "structured_assertion",
    "subsystem",
    "subsystem_ref",
    "surveillance",
    "taxonomy",
]

WORKSPACE_TYPES = [
    "csv",
    "diffexp_input_data",
    "diffexp_input_metadata",
    "doc",
    "docx",
    "embl",
    "feature_dna_fasta",
    "feature_group",
    "feature_protein_fasta",
    "genbank_file",
    "genome_group",
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

VALID_AGENTS = [
    "data",
    "service",
    "workspace",
    "helpdesk",
    "analysis",
    "direct",
]


# ===================================================================
# DATA TOOLS
# ===================================================================

SEARCH_DATA = {
    "type": "function",
    "function": {
        "name": "search_data",
        "description": (
            "Search a BV-BRC Solr data collection with filters. Use this to query "
            "genomes, genome features, AMR data, pathways, taxonomy, specialty genes, "
            "subsystems, and all other BV-BRC collections. Returns matching records. "
            "Always specify the fields you need in 'select' to reduce response size."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": "The Solr collection to query.",
                    "enum": COLLECTIONS,
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Solr query string using Solr syntax. Examples:\n"
                        "  genome_name:Salmonella AND host_name:Human\n"
                        "  resistant_phenotype:Resistant AND antibiotic:ciprofloxacin\n"
                        "  genome_id:(83332.12 OR 208964.12)\n"
                        "  taxon_lineage_ids:1763 (all Mycobacterium)\n"
                        "  product:*kinase* (wildcard search)\n"
                        "  collection_year:[2020 TO 2024] (year range)\n"
                        "  genome_length:[4000000 TO 5000000] (numeric range)\n"
                        "  gc_content:[60 TO *] (open-ended range)\n"
                        "Use * for all records."
                    ),
                },
                "select": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Fields to return (Solr fl parameter). Only request fields "
                        "you actually need. If omitted, returns all fields."
                    ),
                },
                "sort": {
                    "type": "string",
                    "description": (
                        "Sort order. Format: 'field_name asc' or 'field_name desc'. "
                        "Example: 'genome_name asc'"
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of records to return. Default 25. Max 1000.",
                    "default": 25,
                },
                "count_only": {
                    "type": "boolean",
                    "description": (
                        "If true, return only the count of matching records without "
                        "fetching data. Use this first to gauge result set size."
                    ),
                    "default": False,
                },
            },
            "required": ["collection", "query"],
        },
    },
}

FACET_QUERY = {
    "type": "function",
    "function": {
        "name": "facet_query",
        "description": (
            "Get faceted counts (value distributions) for fields in a BV-BRC "
            "collection. Use this to understand data distributions, get breakdowns "
            "by category, or answer 'how many X per Y' questions. Returns counts "
            "grouped by field values without returning individual records."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": "The Solr collection to query.",
                    "enum": COLLECTIONS,
                },
                "query": {
                    "type": "string",
                    "description": "Solr query to filter records before faceting. Use * for all.",
                },
                "facet_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Fields to get value distributions for. Example: "
                        "['host_name', 'isolation_country'] to see counts by host and country."
                    ),
                },
                "facet_limit": {
                    "type": "integer",
                    "description": "Max number of facet values to return per field. Default 20.",
                    "default": 20,
                },
                "facet_mincount": {
                    "type": "integer",
                    "description": "Minimum count for a facet value to be included. Default 1.",
                    "default": 1,
                },
            },
            "required": ["collection", "query", "facet_fields"],
        },
    },
}

PROBE_DATA = {
    "type": "function",
    "function": {
        "name": "probe_data",
        "description": (
            "Do a keyword-based reconnaissance search against a BV-BRC collection "
            "to discover what field values actually exist in the data. Returns the "
            "total match count and faceted value distributions for requested fields. "
            "Use this BEFORE constructing a structured query when you are unsure "
            "about exact field values, taxonomic rank, or canonical spelling. "
            "Also use this when a structured query returns 0 results, to find out "
            "what values actually exist."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": "The collection to probe.",
                    "enum": COLLECTIONS,
                },
                "keywords": {
                    "type": "string",
                    "description": (
                        "Search keywords matched against all text fields in the "
                        "collection (full-text search). Examples: "
                        "'Deltacoronavirus', 'ciprofloxacin resistant', "
                        "'SARS-CoV-2 human'."
                    ),
                },
                "facet_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Fields to get value distributions for. Returns unique "
                        "values and their counts. Examples: "
                        "['genus', 'species', 'genome_status'] for taxonomy "
                        "questions, ['antibiotic', 'resistant_phenotype'] for "
                        "AMR questions."
                    ),
                },
                "facet_limit": {
                    "type": "integer",
                    "description": "Max number of values per facet field. Default 20.",
                    "default": 20,
                },
            },
            "required": ["collection", "keywords"],
        },
    },
}

LIST_COLLECTIONS = {
    "type": "function",
    "function": {
        "name": "list_collections",
        "description": (
            "List all available BV-BRC Solr data collections with descriptions. "
            "Use this when you need to discover which collection to query."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

GET_COLLECTION_FIELDS = {
    "type": "function",
    "function": {
        "name": "get_collection_fields",
        "description": (
            "Get the full list of queryable fields and their types for a specific "
            "BV-BRC Solr collection. Use this to discover valid field names before "
            "building a query, or when a query fails due to an invalid field."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": "The collection to inspect.",
                    "enum": COLLECTIONS,
                },
            },
            "required": ["collection"],
        },
    },
}


# ===================================================================
# WORKSPACE TOOLS
# ===================================================================

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
                    "description": "Maximum number of results to return. Default 50.",
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


# ===================================================================
# GOWE WORKFLOW TOOLS
# ===================================================================

LIST_GOWE_WORKFLOWS = {
    "type": "function",
    "function": {
        "name": "list_gowe_workflows",
        "description": (
            "List all available bioinformatics workflows. Returns workflow "
            "id, name, description, and step count for each. Call this "
            "first to discover which workflow matches the user's request."
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
        "description": (
            "Get the full input schema for a specific workflow. "
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
                        "The workflow ID (e.g., 'wf_abc123'). "
                        "Get this from list_gowe_workflows."
                    ),
                },
            },
            "required": ["workflow_id"],
        },
    },
}

SUBMIT_GOWE_JOB = {
    "type": "function",
    "function": {
        "name": "submit_gowe_job",
        "description": (
            "Submit a job with populated inputs. Call this after you have "
            "gathered all required input values for the selected workflow. "
            "The inputs dict must match the workflow's input schema."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "The workflow ID to run.",
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


# ===================================================================
# GROUP TOOLS
# ===================================================================

CREATE_GROUP = {
    "type": "function",
    "function": {
        "name": "create_group",
        "description": (
            "Create a genome or feature group in the user's BV-BRC workspace "
            "from a Solr query. Runs the query to fetch matching IDs, then "
            "creates the group. Use this when the user asks to save search "
            "results as a group, or when a downstream service needs a genome "
            "or feature group as input. For mixed exact subsets (e.g. 5 of A "
            "and 5 of B), pass a genome_id:(id1 OR id2 OR ...) query built "
            "from prior search_data calls; a single A OR B query with limit "
            "will NOT balance subsets."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "group_name": {
                    "type": "string",
                    "description": "Name for the new group.",
                },
                "group_type": {
                    "type": "string",
                    "enum": ["genome_group", "feature_group"],
                    "description": "Type of group to create.",
                },
                "collection": {
                    "type": "string",
                    "description": (
                        "Solr collection to query for IDs. "
                        "Use 'genome' for genome groups, "
                        "'genome_feature' for feature groups."
                    ),
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Solr query string (same syntax as search_data). "
                        "Example: 'genus:Salmonella AND host_name:Human'"
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Maximum number of IDs to include in the group. "
                        "Default 500. Some services have input limits -- "
                        "use this to cap the group size accordingly."
                    ),
                    "default": 500,
                },
            },
            "required": ["group_name", "group_type", "collection", "query"],
        },
    },
}

LIST_GROUPS = {
    "type": "function",
    "function": {
        "name": "list_groups",
        "description": (
            "List all genome groups or feature groups in the user's BV-BRC "
            "workspace. Returns group names and count. Use this to discover "
            "which groups exist before retrieving their contents with "
            "get_group_ids."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "group_type": {
                    "type": "string",
                    "enum": ["genome_group", "feature_group"],
                    "description": (
                        "Type of groups to list. Use 'genome_group' for "
                        "genome groups, 'feature_group' for feature groups."
                    ),
                },
            },
            "required": ["group_type"],
        },
    },
}

GET_GROUP_IDS = {
    "type": "function",
    "function": {
        "name": "get_group_ids",
        "description": (
            "Get the member IDs (genome IDs or feature IDs) from a genome "
            "group or feature group by name. The group is looked up by name "
            "automatically -- you do NOT need to provide a workspace path. "
            "If the name is ambiguous, returns candidates for clarification."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "group_name": {
                    "type": "string",
                    "description": (
                        "Name of the group (e.g. 'My E. coli genomes'). "
                        "Do NOT provide a workspace path -- just the name."
                    ),
                },
                "group_type": {
                    "type": "string",
                    "enum": ["genome_group", "feature_group"],
                    "description": (
                        "Type of group. Use 'genome_group' for genome "
                        "groups, 'feature_group' for feature groups."
                    ),
                },
            },
            "required": ["group_name", "group_type"],
        },
    },
}


# ===================================================================
# SRA TOOLS
# ===================================================================

GET_SRA_METADATA = {
    "type": "function",
    "function": {
        "name": "get_sra_metadata",
        "description": (
            "Retrieve metadata for one or more SRA run accession IDs (SRR IDs) "
            "from NCBI. Returns organism name, sequencing platform, library "
            "strategy, sample details, and more for each SRA ID. Use this when "
            "the user asks about an SRA sample or provides SRA accession IDs "
            "(e.g., SRR..., SRX..., ERR..., DRR...). ALWAYS call this tool "
            "BEFORE planning any services when the user provides SRA accessions."
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
        },
    },
}


# ===================================================================
# GENOME SIMILARITY
# ===================================================================

FIND_SIMILAR_GENOMES = {
    "type": "function",
    "function": {
        "name": "find_similar_genomes",
        "description": (
            "Find public genomes in BV-BRC that are similar to a query genome "
            "using Mash/MinHash genomic distance estimation. Returns genome IDs "
            "ranked by distance. This is NOT a Solr query -- it uses a dedicated "
            "MinHash service for sequence-level similarity.\n\n"
            "Provide exactly ONE of genome_id or fasta_file (not both).\n\n"
            "USE THIS TOOL FOR:\n"
            "- Finding the closest public genomes to a query genome\n"
            "- Identifying related or similar organisms by genomic distance\n"
            "- Pre-screening reference genomes before downstream analysis\n"
            "- Answering 'what genomes are similar to X?'\n\n"
            "DO NOT USE THIS TOOL FOR:\n"
            "- BLAST sequence similarity searches (use submit_gowe_job with a BLAST workflow)\n"
            "- Phylogenetic tree building (use submit_gowe_job with a tree workflow)\n"
            "- Genome annotation or assembly (use the appropriate workflow submission tools)"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "genome_id": {
                    "type": "string",
                    "description": "A BV-BRC genome ID (e.g. '83332.12'). Mutually exclusive with fasta_file.",
                },
                "fasta_file": {
                    "type": "string",
                    "description": "FULL workspace path to a FASTA/contigs file (e.g. '/user@patricbrc.org/home/file.fasta'). Must start with /. Use workspace_browse to find the path if needed.",
                },
                "max_pvalue": {
                    "type": "number",
                    "description": "Max p-value threshold (default 0.01). Options: 0.001, 0.01, 0.1, 1.0",
                },
                "max_distance": {
                    "type": "number",
                    "description": "Max Mash distance (default 0.01). Options: 0.01, 0.05, 0.1, 0.5, 1.0",
                },
                "max_hits": {
                    "type": "integer",
                    "description": "Max results to return (default 50). Options: 1, 10, 50, 100, 500",
                },
                "scope": {
                    "type": "string",
                    "description": "'reference' (ref+rep only, default) or 'all' (all public genomes).",
                },
                "include_bacterial": {
                    "type": "boolean",
                    "description": "Include bacterial/archaeal genomes (default true).",
                },
                "include_viral": {
                    "type": "boolean",
                    "description": "Include viral genomes (default true).",
                },
            },
            "required": ["genome_id"],
        },
    },
}


# ===================================================================
# LITERATURE
# ===================================================================

SEARCH_LITERATURE = {
    "type": "function",
    "function": {
        "name": "search_literature",
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


# ===================================================================
# HELPDESK TOOLS
# ===================================================================

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
            "- Understanding what a particular service does and when to use it"
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


# ===================================================================
# ANALYSIS TOOLS
# ===================================================================

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

LIST_JOBS = {
    "type": "function",
    "function": {
        "name": "list_jobs",
        "description": (
            "List the user's BV-BRC jobs with optional filtering, sorting, and "
            "pagination. Use this to answer questions like 'show my recent jobs', "
            "'what jobs are running', 'list failed jobs', or 'find my assembly jobs'. "
            "Returns job summaries (status, service, submit time, etc.).\n\n"
            "USE THIS TOOL FOR:\n"
            "- Listing recent jobs or job history\n"
            "- Finding jobs by status (completed, running, failed)\n"
            "- Finding jobs by service type (genome_assembly, blast, etc.)\n"
            "- Searching jobs by name or description\n"
            "- Paginated browsing of the job queue\n\n"
            "DO NOT USE THIS TOOL FOR:\n"
            "- Deep inspection of a specific job (use get_job_details with task IDs)\n"
            "- Submitting new jobs (use the workflow submission tools)"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of jobs to return. Default 20.",
                    "default": 20,
                },
                "offset": {
                    "type": "integer",
                    "description": "Number of jobs to skip for pagination. Default 0.",
                    "default": 0,
                },
                "sort_by": {
                    "type": "string",
                    "description": (
                        "Field to sort by. Default 'submit_time'. "
                        "Options: submit_time, start_time, status, app, id"
                    ),
                    "default": "submit_time",
                },
                "sort_dir": {
                    "type": "string",
                    "description": "Sort direction: 'asc' or 'desc'. Default 'desc'.",
                    "enum": ["asc", "desc"],
                    "default": "desc",
                },
                "status": {
                    "type": "string",
                    "description": (
                        "Filter by job status. Examples: 'completed', "
                        "'failed', 'running', 'queued'."
                    ),
                },
                "service": {
                    "type": "string",
                    "description": (
                        "Filter by service name. Examples: 'genome_assembly', "
                        "'blast', 'genome_annotation', 'rnaseq', 'variation'."
                    ),
                },
                "search": {
                    "type": "string",
                    "description": "Search term to filter jobs by name or description.",
                },
                "include_archived": {
                    "type": "boolean",
                    "description": "Whether to include archived jobs. Default false.",
                    "default": False,
                },
            },
            "required": [],
        },
    },
}


# ===================================================================
# PLANNING TOOLS
# ===================================================================

ASK_CLARIFICATION = {
    "type": "function",
    "function": {
        "name": "ask_clarification",
        "description": (
            "Ask the user clarification questions when information is "
            "missing or ambiguous.  Use this during plan creation when "
            "the request lacks key details, OR during plan step execution "
            "when you cannot complete your assigned task without "
            "additional input from the user.  Calling this tool pauses "
            "plan execution and presents the questions to the user.  "
            "Each question should have 2-5 suggested options."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The question to ask the user",
                            },
                            "options": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "2-5 suggested answer options. "
                                    "The user can also type a custom answer."
                                ),
                            },
                            "required": {
                                "type": "boolean",
                                "description": "Whether an answer is required",
                                "default": True,
                            },
                        },
                        "required": ["question", "options"],
                    },
                    "description": "List of questions to ask the user",
                },
            },
            "required": ["questions"],
        },
    },
}

CREATE_PLAN = {
    "type": "function",
    "function": {
        "name": "create_plan",
        "description": (
            "Create a step-by-step execution plan. Each step is assigned "
            "to a specific agent. Steps can declare dependencies on other "
            "steps via depends_on. The plan will be presented to the user "
            "for review and editing before execution."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short, descriptive title for the plan",
                },
                "description": {
                    "type": "string",
                    "description": "Brief rationale explaining the plan",
                },
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step_id": {
                                "type": "string",
                                "description": (
                                    "Unique step identifier "
                                    "(e.g. 'search_genomes', 'run_assembly')"
                                ),
                            },
                            "description": {
                                "type": "string",
                                "description": "What this step does",
                            },
                            "agent": {
                                "type": "string",
                                "enum": VALID_AGENTS,
                                "description": (
                                    "Which agent handles this step. "
                                    "'data' for database queries, "
                                    "'service' for running BV-BRC services, "
                                    "'workspace' for browsing workspace files, "
                                    "'helpdesk' for documentation/FAQ, "
                                    "'analysis' for post-job analysis, "
                                    "'direct' for steps you can answer yourself"
                                ),
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Why this step is needed",
                            },
                            "depends_on": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "step_ids that must complete before "
                                    "this step can run"
                                ),
                                "default": [],
                            },
                        },
                        "required": [
                            "step_id",
                            "description",
                            "agent",
                            "reasoning",
                        ],
                    },
                },
            },
            "required": ["title", "description", "steps"],
        },
    },
}

LIST_AGENTS = {
    "type": "function",
    "function": {
        "name": "list_agents",
        "description": (
            "List available BV-BRC agents and their capabilities. "
            "Call this to understand what agents are available before "
            "creating a plan."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
}


# ===================================================================
# MASTER LISTS
# ===================================================================

# All 26 tools in a single list
ALL_TOOL_SCHEMAS: list[dict] = [
    # Data
    SEARCH_DATA,
    FACET_QUERY,
    PROBE_DATA,
    LIST_COLLECTIONS,
    GET_COLLECTION_FIELDS,
    # Workspace
    WORKSPACE_BROWSE,
    GET_FILE_METADATA,
    READ_FILE_PREVIEW,
    # GoWe
    LIST_GOWE_WORKFLOWS,
    GET_WORKFLOW_INPUTS,
    SUBMIT_GOWE_JOB,
    # Groups
    CREATE_GROUP,
    LIST_GROUPS,
    GET_GROUP_IDS,
    # SRA
    GET_SRA_METADATA,
    # Genome similarity
    FIND_SIMILAR_GENOMES,
    # Literature
    SEARCH_LITERATURE,
    # Helpdesk
    QUERY_HELPDESK,
    LIST_SERVICES,
    GET_SERVICE_SCHEMA,
    # Analysis
    GET_EXPECTED_OUTPUTS,
    GET_JOB_DETAILS,
    LIST_JOBS,
    # Planning
    ASK_CLARIFICATION,
    CREATE_PLAN,
    LIST_AGENTS,
]

# Name -> schema lookup
TOOL_SCHEMA_MAP: dict[str, dict] = {
    schema["function"]["name"]: schema for schema in ALL_TOOL_SCHEMAS
}
