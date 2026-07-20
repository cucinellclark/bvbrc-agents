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
]
