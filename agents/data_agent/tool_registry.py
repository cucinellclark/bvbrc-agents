"""
OpenAI-compatible function/tool schema definitions for Phase 1 tools.

These schemas tell the LLM what tools are available and how to call them.
In plan-only mode, the LLM produces tool_calls but they are not executed --
only captured and displayed for inspection.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Collection enum -- all 33 BV-BRC Solr collections
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


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function calling format)
# ---------------------------------------------------------------------------

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

GET_GENOME_GROUP = {
    "type": "function",
    "function": {
        "name": "get_genome_group",
        "description": (
            "Retrieve the genome IDs from a named genome group in the user's "
            "workspace. Use this when the user refers to 'my genomes' or a named "
            "group. Returns a list of genome_id values that can be used as filters "
            "in search_data queries."
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
        },
    },
}

GET_FEATURE_GROUP = {
    "type": "function",
    "function": {
        "name": "get_feature_group",
        "description": (
            "Retrieve feature IDs from a named feature group in the user's "
            "workspace. Returns a list of feature_id / patric_id values that can "
            "be used as filters in search_data queries."
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


# ---------------------------------------------------------------------------
# Complete tool list for the agent
# ---------------------------------------------------------------------------
TOOL_SCHEMAS: list[dict] = [
    SEARCH_DATA,
    LIST_COLLECTIONS,
    GET_COLLECTION_FIELDS,
    GET_GENOME_GROUP,
    GET_FEATURE_GROUP,
    FACET_QUERY,
    PROBE_DATA,
]

# Dispatch table: tool name -> schema (for future execution mapping)
TOOL_MAP: dict[str, dict] = {
    schema["function"]["name"]: schema for schema in TOOL_SCHEMAS
}
