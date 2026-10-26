"""Tool schemas for the BV-BRC agents: the BV-BRC tools from ``bvbrc_mcp``
plus the harness's own planning tools.

32 tools: the 29 in ``bvbrc_mcp.tools.schemas`` (data, workspace, GoWe,
groups, SRA, genome similarity, literature, helpdesk, jobs) and
  Planning:  ask_clarification, create_plan, list_agents
"""

from __future__ import annotations

from bvbrc_mcp.tools.schemas import *  # noqa: F401,F403
from bvbrc_mcp.tools.schemas import ALL_TOOL_SCHEMAS as _bvbrc_tool_schemas


VALID_AGENTS = [
    "data",
    "service",
    "workspace",
    "helpdesk",
    "analysis",
    "direct",
]


# ===================================================================
# PLANNING TOOLS (harness only)
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


ALL_TOOL_SCHEMAS: list[dict] = [
    *_bvbrc_tool_schemas,
    # Planning
    ASK_CLARIFICATION,
    CREATE_PLAN,
    LIST_AGENTS,
]

# Name -> schema lookup
TOOL_SCHEMA_MAP: dict[str, dict] = {
    schema["function"]["name"]: schema for schema in ALL_TOOL_SCHEMAS
}
