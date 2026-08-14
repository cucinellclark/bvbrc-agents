"""Tool schemas for the BV-BRC Planning Agent.

These are OpenAI function-calling format tool definitions used during
the planning phase of the agent loop.
"""

VALID_AGENTS = [
    "data",
    "service",
    "workspace",
    "helpdesk",
    "analysis",
    "review",
    "direct",
]

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "ask_clarification",
            "description": (
                "Ask the user clarification questions before creating a plan. "
                "Use this when the request is ambiguous or missing key details "
                "needed to determine which agents and steps are required. "
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
    },
    {
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
                                        "'review' for user review/checkpoint "
                                        "(pause to show data and collect decisions), "
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
                                "review_config": {
                                    "type": "object",
                                    "description": (
                                        "Required when agent is 'review'. "
                                        "Configures the review checkpoint."
                                    ),
                                    "properties": {
                                        "data_source_step": {
                                            "type": "string",
                                            "description": (
                                                "step_id of the step whose "
                                                "results to present for review"
                                            ),
                                        },
                                        "review_type": {
                                            "type": "string",
                                            "enum": [
                                                "data_selection",
                                                "workflow_choice",
                                                "parameter_config",
                                                "group_management",
                                            ],
                                            "description": (
                                                "Type of review: "
                                                "'data_selection' to filter/select data, "
                                                "'workflow_choice' to pick an analysis, "
                                                "'parameter_config' to set parameters, "
                                                "'group_management' to create/add-to/confirm "
                                                "a genome or feature group"
                                            ),
                                        },
                                        "prompt": {
                                            "type": "string",
                                            "description": (
                                                "Question/instruction to present "
                                                "to the user during review"
                                            ),
                                        },
                                        "suggested_workflows": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": (
                                                "GoWe workflow names to suggest "
                                                "(for workflow_choice reviews)"
                                            ),
                                            "default": [],
                                        },
                                        "suggested_group_name": {
                                            "type": "string",
                                            "description": (
                                                "Pre-filled group name suggestion "
                                                "(for group_management reviews). "
                                                "Generate from context, e.g. "
                                                "'Salmonella AMR Genomes'."
                                            ),
                                        },
                                        "group_type": {
                                            "type": "string",
                                            "enum": [
                                                "genome_group",
                                                "feature_group",
                                            ],
                                            "description": (
                                                "Type of group to manage "
                                                "(for group_management reviews). "
                                                "Infer from collection: genome -> "
                                                "genome_group, genome_feature -> "
                                                "feature_group."
                                            ),
                                        },
                                        "group_action": {
                                            "type": "string",
                                            "enum": [
                                                "create",
                                                "add_to",
                                                "use_existing",
                                            ],
                                            "description": (
                                                "Default action for group management: "
                                                "'create' a new group, 'add_to' an "
                                                "existing group, or 'use_existing' to "
                                                "confirm an existing group for use."
                                            ),
                                        },
                                        "id_field": {
                                            "type": "string",
                                            "enum": [
                                                "genome_id",
                                                "feature_id",
                                            ],
                                            "description": (
                                                "ID field name for the items "
                                                "(for group_management reviews). "
                                                "Infer from group_type."
                                            ),
                                        },
                                    },
                                    "required": [
                                        "data_source_step",
                                        "review_type",
                                        "prompt",
                                    ],
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
    },
    {
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
    },
    # ------------------------------------------------------------------
    # Shared reconnaissance tools
    # ------------------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "workspace_browse",
            "description": (
                "Browse the user's BV-BRC workspace to find files, genome "
                "groups, feature groups, and job output folders. Use this "
                "during planning to discover what data the user already has "
                "before asking clarification questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": ["string", "null"],
                        "description": (
                            "Workspace path to browse. Null defaults to home. "
                            "Use 'Genome Groups' to browse genome groups directly, "
                            "or 'Feature Groups' for feature groups."
                        ),
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
                            "Filter by workspace object type. Common types: "
                            "'genome_group', 'feature_group', 'reads', "
                            "'contigs', 'genbank_file', 'gff', 'nwk', "
                            "'csv', 'json', 'folder'."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_file_metadata",
            "description": (
                "Get metadata for a specific workspace file (type, size, date)."
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
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_data",
            "description": (
                "Query BV-BRC Solr collections. Use during planning to "
                "verify data availability or resolve organism names."
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
    },
    {
        "type": "function",
        "function": {
            "name": "create_group",
            "description": (
                "Create a genome or feature group in the user's BV-BRC "
                "workspace from a Solr query. Runs the query to fetch "
                "matching IDs, then creates the group. Use this during "
                "planning when a step needs to create a group for "
                "downstream services."
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
                            "Solr collection to query. Use 'genome' for "
                            "genome groups, 'genome_feature' for feature groups."
                        ),
                    },
                    "query": {
                        "type": "string",
                        "description": (
                            "Solr query string (same syntax as search_data)."
                        ),
                    },
                    "limit": {
                        "type": ["integer", "null"],
                        "description": (
                            "Max IDs to include (default 500). Use this "
                            "when a downstream service has an input cap."
                        ),
                    },
                },
                "required": ["group_name", "group_type", "collection", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_gowe_workflows",
            "description": (
                "List available GoWe workflows. Use during planning to "
                "verify which workflows exist before assigning service steps."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_similar_genomes",
            "description": (
                "Find public BV-BRC genomes similar to a query genome using "
                "Mash/MinHash distance estimation. Use during planning to identify "
                "reference genomes or related organisms before downstream analysis steps."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "genome_id": {
                        "type": ["string", "null"],
                        "description": "A BV-BRC genome ID (e.g. '83332.12').",
                    },
                    "fasta_file": {
                        "type": ["string", "null"],
                        "description": "FULL workspace path to a FASTA/contigs file (e.g. '/user@patricbrc.org/home/file.fasta'). Must start with /.",
                    },
                    "max_pvalue": {"type": ["number", "null"], "description": "Max p-value (default 0.01)."},
                    "max_distance": {"type": ["number", "null"], "description": "Max Mash distance (default 0.01)."},
                    "max_hits": {"type": ["integer", "null"], "description": "Max results (default 50)."},
                    "scope": {"type": ["string", "null"], "description": "'reference' or 'all'."},
                    "include_bacterial": {"type": ["boolean", "null"], "description": "Include bacterial genomes (default true)."},
                    "include_viral": {"type": ["boolean", "null"], "description": "Include viral genomes (default true)."},
                },
                "required": ["genome_id", "fasta_file"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_literature",
            "description": (
                "Search scientific literature using the literature RAG service and return raw "
                "source passages with bibliographic metadata. Use during planning when the plan "
                "needs literature-backed context (papers, evidence, citations)."
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
            },
        },
    },
]
