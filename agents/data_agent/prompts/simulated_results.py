"""Simulated tool results for plan-only mode.

This module builds structured fake responses for each tool so the LLM
can reason about what it *would* get and plan subsequent steps. The
``_note`` strings within each result are LLM-facing instructions that
guide the agent's planning behavior -- they are evolution targets.

Moved here from agent.py so the prompt evolver can discover and target
these strings alongside the system prompt sections.
"""

from __future__ import annotations

from typing import Any


def build_simulated_result(tc: Any) -> dict:
    """Build a tool-specific simulated result for plan-only mode.

    Instead of a generic "assume it succeeded" message, this returns a
    structured response that mimics the shape of real tool output so the
    LLM can reason about what it would get and plan subsequent steps.

    Args:
        tc: A ToolCall object with .name and .arguments attributes.

    Returns:
        A dict mimicking the shape of a real tool result, including a
        ``_note`` string that guides the LLM's next action.
    """
    base: dict[str, Any] = {
        "_mode": "plan_only",
        "_tool": tc.name,
        "_args": tc.arguments,
    }

    if tc.name == "search_data":
        if tc.arguments.get("count_only"):
            base["count"] = "<count of matching records>"
            base["_note"] = (
                "This count query was recorded. If you have enough information "
                "to answer the user's question, provide your final answer now. "
                "Otherwise, plan your next query."
            )
        else:
            fields = tc.arguments.get("select", ["genome_id", "genome_name"])
            sample_record = {f: f"<{f}_value>" for f in fields}
            base["count"] = "<total matching records>"
            base["records"] = [sample_record]
            base["_note"] = (
                "This search was recorded. The result would contain records "
                "with the fields shown. If you now have the data you need, "
                "provide your final answer. Otherwise, plan your next step."
            )

    elif tc.name == "facet_query":
        facet_fields = tc.arguments.get("facet_fields", [])
        base["facets"] = {
            f: {"<value_1>": "<count>", "<value_2>": "<count>"}
            for f in facet_fields
        }
        base["_note"] = (
            "This facet query was recorded. It would return value distributions "
            "for the requested fields. If this answers the user's question, "
            "provide your final answer now."
        )

    elif tc.name == "list_collections":
        base["collections"] = ["genome", "genome_feature", "genome_amr", "..."]
        base["_note"] = (
            "Collection list was recorded. Now query the appropriate collection."
        )

    elif tc.name == "get_collection_fields":
        collection = tc.arguments.get("collection", "unknown")
        base["fields"] = [f"<field_1 for {collection}>", f"<field_2 for {collection}>"]
        base["_note"] = (
            "Field list was recorded. Use the correct field names in your query."
        )

    elif tc.name == "probe_data":
        facet_fields = tc.arguments.get("facet_fields", [])
        base["numFound"] = "<total matching records>"
        base["facets"] = {
            f: [
                {"value": "<value_1>", "count": "<count>"},
                {"value": "<value_2>", "count": "<count>"},
            ]
            for f in facet_fields
        }
        base["_note"] = (
            "This probe query was recorded. It would return the total count "
            "and value distributions for the requested fields. Use the facet "
            "results to determine the correct field:value pairs for your "
            "structured query. If numFound already answers the user's question, "
            "provide your final answer now."
        )

    elif tc.name in ("get_genome_group", "get_feature_group"):
        id_field = "genome_id" if tc.name == "get_genome_group" else "feature_id"
        base["ids"] = [f"<{id_field}_1>", f"<{id_field}_2>", f"<{id_field}_3>"]
        base["count"] = "<number of IDs in group>"
        base["_note"] = (
            f"Group IDs were recorded. Use these {id_field} values as filters "
            "in a search_data query for your next step."
        )

    else:
        base["_note"] = (
            "Tool call was recorded. Plan your next step or provide your "
            "final answer if you have enough information."
        )

    return base
