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
# Complete tool list for the agent
# ---------------------------------------------------------------------------
TOOL_SCHEMAS: list[dict] = [
    QUERY_HELPDESK,
    LIST_SERVICES,
    GET_SERVICE_SCHEMA,
]

# Dispatch table: tool name -> schema
TOOL_MAP: dict[str, dict] = {
    schema["function"]["name"]: schema for schema in TOOL_SCHEMAS
}
