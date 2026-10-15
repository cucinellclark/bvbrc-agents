"""Canonical OpenAI function-calling tool schemas for all BV-BRC agent tools.

This is the SINGLE SOURCE OF TRUTH for tool schemas. Every agent uses these
definitions. No per-agent schema variants.

31 tools total:
  Data:      search_data, facet_query, probe_data, list_collections, get_collection_fields
  Workspace: workspace_browse, get_file_metadata, read_file_preview, search_file, summarize_file,
             list_samples
  GoWe:      list_gowe_workflows, get_workflow_inputs, submit_gowe_job, rerun_jobs, prepare_batch
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
                    "description": (
                        "Maximum number of records to return. Default 25. Capped at 200 "
                        "(results are truncated before you see them anyway): use "
                        "count_only or facet_query for counts, and give the user the "
                        "download link for full lists."
                    ),
                    "default": 25,
                },
                "allow_large": {
                    "type": "boolean",
                    "description": (
                        "Lift the 200-row cap. Only when you must read more rows "
                        "yourself, e.g. collecting more than 200 IDs for one step."
                    ),
                    "default": False,
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
            "grouped by field values without returning individual records. "
            "To answer 'how many DISTINCT X' (e.g. how many genomes have a "
            "matching genome_amr or genome_feature record), set count_distinct=true "
            "and read distinct_counts[field] — never request a large facet_limit "
            "and count the values yourself."
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
                "count_distinct": {
                    "type": "boolean",
                    "description": (
                        "When true, return the exact number of distinct values per "
                        "facet field in distinct_counts (all buckets are counted "
                        "server-side; only the top 20 are returned). Use this for "
                        "'how many genomes/features/...' questions where records "
                        "are not one-per-entity. facet_limit is ignored."
                    ),
                    "default": False,
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
            "Read a workspace file in pages. Returns up to max_bytes "
            "(≤32 KB) starting at start_byte; call again with "
            "start_byte = next_start until is_complete is true. Extract "
            "what you need from each page as you go — earlier pages may "
            "be trimmed from your context. Compressed .gz files are read "
            "transparently; offsets refer to the uncompressed text. PDFs "
            "return extracted text. To find something specific in a large "
            "file use search_file and read from the match's byte_offset "
            "instead of paging end to end."
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
                        "Maximum bytes to read per page. Default 8192. "
                        "Max 32768 (32 KB)."
                    ),
                    "default": 8192,
                },
                "start_byte": {
                    "type": "integer",
                    "description": (
                        "Byte offset to start reading from. Default 0. "
                        "For gzip files this is the decompressed-stream "
                        "offset. Set to next_start from the previous "
                        "page to continue reading."
                    ),
                    "default": 0,
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
                        "The workflow NAME exactly as list_gowe_workflows "
                        "reported it (e.g. 'MetaCATS', 'HA Subtype Numbering "
                        "Conversion'). PREFER THE NAME over the wf_<uuid> id: "
                        "both work, but a 36-character UUID is easy to "
                        "mistype and a wrong one cannot be recovered from."
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
                    "description": (
                        "The workflow NAME exactly as list_gowe_workflows "
                        "reported it. PREFER THE NAME over the wf_<uuid> id."
                    ),
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

RERUN_JOBS = {
    "type": "function",
    "function": {
        "name": "rerun_jobs",
        "description": (
            "Prepare a rerun of earlier BV-BRC jobs with their original "
            "parameters. Give a workspace folder (every job result under it, "
            "up to 4 levels deep) and/or task ids. The tool reads each job's "
            "original app and parameters, finds the matching workflow, and "
            "returns prepared_actions: one job per original, outputs in this "
            "chat's folder named <original>_rerun. It submits NOTHING; the "
            "user confirms the whole set with one question. Call it ONCE for a "
            "rerun / resubmit / run-again request instead of browsing the "
            "folder, calling get_job_details or building submit_gowe_job "
            "calls yourself. Then list each prepared job, each skipped job "
            "with its reason, and any collapsed duplicates. errorType "
            "TOO_MANY (more than 10 jobs) prepares nothing: ask the user which "
            "subfolder or task ids to rerun."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "folder": {
                    "type": "string",
                    "description": (
                        "Workspace folder holding the job results, as the "
                        "user gave it or as a tool returned it."
                    ),
                },
                "task_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "BV-BRC job (task) ids, e.g. ['23428307'].",
                },
                "include_failed": {
                    "type": "boolean",
                    "description": (
                        "Also rerun jobs whose original failed (default true; "
                        "they are marked in the list). False when the user "
                        "asks for the successful ones only."
                    ),
                },
            },
            "required": [],
        },
    },
}


LIST_SAMPLES = {
    "type": "function",
    "function": {
        "name": "list_samples",
        "description": (
            "List the samples in a folder of sequencing reads, paired by file "
            "name (R1/R2, _1/_2), and/or check a list of SRA run accessions. "
            "Reads one folder level (.fastq/.fq, optionally .gz). Returns "
            "paired [{sample_id, read1, read2}], single [{sample_id, read}], "
            "sra [{sample_id, srr_accession}] and unpaired files (one mate "
            "missing: NOT a sample). Use it instead of pairing files from a "
            "workspace_browse listing yourself, whenever a workflow is to run "
            "on every sample in a folder or on several SRA runs; pass its "
            "entries to prepare_batch as samples."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace folder holding the read files.",
                },
                "sra_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "SRA run accessions, e.g. ['SRR1234567', 'ERR7916233'].",
                },
            },
            "required": [],
        },
    },
}


PREPARE_BATCH = {
    "type": "function",
    "function": {
        "name": "prepare_batch",
        "description": (
            "Prepare one job per sample for ONE workflow (or one job holding "
            "every sample, one_job=true) without submitting anything. First "
            "call get_workflow_inputs and populate every input EXCEPT the "
            "reads/SRA inputs and output_file -- that is template_inputs. "
            "samples are the entries of list_samples (paired, single, sra). "
            "The tool puts each sample's reads into the right input, names "
            "each job's output_file <template output_file>_<sample_id>, checks "
            "each job with GoWe and returns prepared_actions; the user "
            "confirms the set with one question. Do NOT call submit_gowe_job "
            "for these samples yourself. errorType TOO_MANY (more than 10 "
            "jobs) prepares nothing: ask which samples to run first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "Workflow NAME exactly as list_gowe_workflows reported it.",
                },
                "template_inputs": {
                    "type": "object",
                    "description": (
                        "Every input the jobs share (recipe, platform, "
                        "organism, output_path ...), as you would pass to "
                        "submit_gowe_job, without the read / SRA inputs."
                    ),
                },
                "samples": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": (
                        "list_samples entries: {sample_id, read1, read2}, "
                        "{sample_id, read} or {sample_id, srr_accession}."
                    ),
                },
                "one_job": {
                    "type": "boolean",
                    "description": (
                        "Put every sample in ONE job (RNA-Seq and other "
                        "workflows that compare samples). Default false: one "
                        "job per sample."
                    ),
                },
            },
            "required": ["workflow_id", "template_inputs", "samples"],
        },
    },
}


CREATE_GROUP = {
    "type": "function",
    "function": {
        "name": "create_group",
        "description": (
            "Create a genome or feature group in the user's BV-BRC workspace. "
            "Give EITHER a Solr query (any collection: genome, genome_amr, "
            "genome_feature, sp_gene, ...; the tool collects the genome_id or "
            "feature_id of every matching row, deduplicated, up to limit) OR "
            "an explicit list of ids -- never both. Use the query when the "
            "selection is described by fields; use ids for pasted lists or "
            "ids an earlier step found. Use this when the user asks to save "
            "results as a group, or when a downstream service needs a genome "
            "or feature group as input. For mixed exact subsets (e.g. 5 of A "
            "and 5 of B), pass the chosen ids; a single A OR B query with "
            "limit will NOT balance subsets."
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
                        "Solr collection the query runs on (with query). "
                        "'genome' or 'genome_amr' for genome groups, "
                        "'genome_feature' for feature groups."
                    ),
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Solr query string (same syntax as search_data). "
                        "Example: 'genus:Salmonella AND host_name:Human'. "
                        "Omit when ids is given."
                    ),
                },
                "ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Explicit member ids instead of a query: genome ids "
                        "like 83332.12 for a genome group; for a feature "
                        "group, feature_id values (PATRIC.83332.12...) or "
                        "patric_ids (fig|83332.12.peg.3062), which the tool "
                        "converts to feature_id. Order is kept, duplicates "
                        "dropped."
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
                "if_exists": {
                    "type": "string",
                    "enum": ["error", "append", "replace"],
                    "description": (
                        "What to do when a group with this name already exists. "
                        "'error' (default): fail with errorType ALREADY_EXISTS and "
                        "change nothing. 'append': add the query's IDs to the "
                        "existing group (duplicates skipped) -- use when the user "
                        "says add/put these into MY group. 'replace': overwrite the "
                        "group with only the new IDs -- only when the user explicitly "
                        "asks to replace it. Never retry a failed create_group with a "
                        "different query; report the error instead."
                    ),
                    "default": "error",
                },
            },
            "required": ["group_name", "group_type"],
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
                    "description": (
                        "Max p-value threshold (default 0.01). Options: 0.001, 0.01, 0.1, "
                        "1.0. For broad searches with many hits, use 0.1 or higher."
                    ),
                },
                "max_distance": {
                    "type": "number",
                    "description": (
                        "Max Mash distance threshold (default 0.01). Options: 0.01, 0.05, "
                        "0.1, 0.5, 1.0. NOTE: The default 0.01 is very restrictive (>99% "
                        "identity). For broad searches (scope='all', high max_hits, or viral "
                        "genomes), use 0.5 to get meaningful results."
                    ),
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
            "Search published scientific literature (ASM journals + PubMed Central "
            "open-access) and return distinct papers, each with title, DOI/URL, year, "
            "journal, authors when known, and the retrieved passages. Use this when the "
            "user asks for published evidence, papers about an organism/gene, or "
            "literature-backed facts (PPI, mutations, phenotypes). Cite only papers it "
            "returns."
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
                    "description": "Maximum number of distinct papers to return (default 10, max 20).",
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
            "Look up how to analyse a BV-BRC service's outputs: which small "
            "files hold the key numbers (read_first), which large files to "
            "avoid, the checks and typical ranges to judge the results "
            "against, and next steps. Path templates use ${output_folder} = "
            "output_path + '/.' + output_file. Local lookup (no API call) -- "
            "call it before reading a job's files."
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
                            "expects": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Named identifiers this step must find for "
                                    "later steps, from: plfam_id, pgfam_id, "
                                    "patric_id, feature_id, genome_id, "
                                    "srr_accession. They "
                                    "are copied exactly into the steps that "
                                    "depend on this one. Groups, job ids and "
                                    "output folders are passed on automatically."
                                ),
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


SEARCH_FILE = {
    "type": "function",
    "function": {
        "name": "search_file",
        "description": (
            "Search inside one workspace text file for a string or regex, "
            "like grep. Returns matching lines, each with a byte_offset you "
            "can pass to read_file_preview(start_byte=...) to read from that "
            "point. Use this instead of paging through a large file to find "
            "a gene ID, sample name, column value, error message, or FASTA "
            "header. For FASTA header matches, record_length gives the "
            "sequence length. Compressed .gz files are searched transparently. "
            "Stops after max_matches or after scanning 25 MB (next_start "
            "tells you where to resume)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Full workspace path to the file to search. Relative "
                        "paths are resolved from the user's home directory."
                    ),
                },
                "pattern": {
                    "type": "string",
                    "description": (
                        "Text to find. Matched as a literal substring unless "
                        "regex is true. Case-insensitive by default."
                    ),
                },
                "regex": {
                    "type": "boolean",
                    "description": "Treat pattern as a Python regular expression. Default false.",
                    "default": False,
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Match case exactly. Default false.",
                    "default": False,
                },
                "max_matches": {
                    "type": "integer",
                    "description": "Stop after this many matches. Default 50, max 200.",
                    "default": 50,
                },
                "context_lines": {
                    "type": "integer",
                    "description": (
                        "Lines of context to include before and after each "
                        "match. Default 0, max 5."
                    ),
                    "default": 0,
                },
                "start_byte": {
                    "type": "integer",
                    "description": (
                        "Byte offset to start searching from (decompressed "
                        "offset for gzip). Default 0. Use next_start from a "
                        "previous truncated search to continue."
                    ),
                    "default": 0,
                },
            },
            "required": ["path", "pattern"],
        },
    },
}

SUMMARIZE_FILE = {
    "type": "function",
    "function": {
        "name": "summarize_file",
        "description": (
            "Structure and statistics of one workspace file without reading "
            "it page by page. Tables (tsv/csv/vcf, tab-separated tool output): "
            "exact row_count, columns with type, min/max/mean/sum or top "
            "values. FASTA: sequence count, total length, N50, GC. FASTQ: read "
            "count, length stats. Use it to COUNT rows: where filters rows "
            "(e.g. \"padj < 0.05 and abs(log2FoldChange) >= 1\") and returns "
            "matched_rows plus column stats over the matching rows only; "
            "sort_by + top_n return the top rows. Compressed .gz files are "
            "read transparently. Scans up to 25 MB; if partial is true, counts "
            "are named *_scanned and cover only that part."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Full workspace path to the file.",
                },
                "where": {
                    "type": "string",
                    "description": (
                        "Tables only. Conditions joined by 'and': <col> <op> "
                        "<value> with ops < <= > >= == != contains, not "
                        "contains, startswith, not startswith, is null, is not "
                        "null; abs(<col>) for absolute values. Quote column "
                        "names with spaces in backticks. A table with no "
                        "header row has columns column_1, column_2, ... "
                        "Rows whose tested cell is NA do not match. "
                        "Examples: 'padj < 0.05', 'column_1 startswith __', "
                        "'`% Identity` >= 90 and Class contains BETA-LACTAM'."
                    ),
                },
                "sort_by": {
                    "type": "string",
                    "description": (
                        "Tables only. Numeric column to rank matching rows by, "
                        "ascending; prefix '-' for descending; abs(<col>) "
                        "allowed (e.g. '-abs(log2FoldChange)')."
                    ),
                },
                "top_n": {
                    "type": "integer",
                    "description": "Rows to return with sort_by. Default 10, max 50.",
                    "default": 10,
                },
                "format": {
                    "type": "string",
                    "enum": ["table", "csv", "tsv", "fasta", "fastq", "text"],
                    "description": "Override the detected format.",
                },
                "has_header": {
                    "type": "boolean",
                    "description": "Override header detection for a table.",
                },
            },
            "required": ["path"],
        },
    },
}

# ===================================================================
# MASTER LISTS
# ===================================================================

# All 31 tools in a single list
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
    SEARCH_FILE,
    SUMMARIZE_FILE,
    LIST_SAMPLES,
    # GoWe
    LIST_GOWE_WORKFLOWS,
    GET_WORKFLOW_INPUTS,
    SUBMIT_GOWE_JOB,
    RERUN_JOBS,
    PREPARE_BATCH,
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
