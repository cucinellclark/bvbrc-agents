"""Planning tool implementations.

These handle the local validation logic for the planning agent's tools.
No external API calls — all validation is programmatic.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Any

from shared.tools.schemas import VALID_AGENTS


def handle_ask_clarification(arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate and structure clarification questions.

    Args:
        arguments: {"questions": [{"question": str, "options": [str], ...}]}

    Returns:
        Validated questions with generated IDs, or error dict.
    """
    questions = arguments.get("questions", [])

    if not questions:
        return {"error": "At least one question is required."}

    validated = []
    for i, q in enumerate(questions):
        question_text = q.get("question", "").strip()
        options = q.get("options", [])
        required = q.get("required", True)

        if not question_text:
            return {"error": f"Question {i + 1} has empty text."}

        if not options or len(options) < 2:
            return {
                "error": (
                    f"Question {i + 1} needs at least 2 options. Got {len(options)}."
                ),
            }

        if len(options) > 6:
            return {
                "error": (
                    f"Question {i + 1} has too many options ({len(options)}). "
                    f"Maximum is 6."
                ),
            }

        validated.append(
            {
                "id": f"q_{i + 1}",
                "question": question_text,
                "options": [str(o).strip() for o in options if str(o).strip()],
                "required": bool(required),
            }
        )

    return {
        "status": "valid",
        "questions": validated,
        "count": len(validated),
    }


def handle_create_plan(arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate a plan's structure and DAG dependencies.

    Performs:
    - Unique step_id check
    - Valid agent names
    - Valid dependency references (no dangling, no self-refs)
    - Cycle detection via topological sort (Kahn's algorithm)
    - Non-empty descriptions

    Args:
        arguments: {"title": str, "description": str, "steps": [...]}

    Returns:
        Validated plan dict with plan_id, or error dict with hints.
    """
    title = arguments.get("title", "").strip()
    description = arguments.get("description", "").strip()
    steps = arguments.get("steps", [])

    # Basic validation
    if not title:
        return {"error": "Plan title is required."}
    if not description:
        return {"error": "Plan description is required."}
    if not steps:
        return {"error": "Plan must have at least one step."}

    # Validate each step
    seen_ids: set[str] = set()
    errors: list[str] = []

    for i, step in enumerate(steps):
        step_id = step.get("step_id", "").strip()
        step_desc = step.get("description", "").strip()
        agent = step.get("agent", "").strip()
        reasoning = step.get("reasoning", "").strip()
        depends_on = step.get("depends_on", [])

        if not step_id:
            errors.append(f"Step {i + 1}: step_id is required.")
            continue

        if step_id in seen_ids:
            errors.append(
                f"Step {i + 1}: duplicate step_id '{step_id}'. "
                f"Each step must have a unique ID."
            )
            continue
        seen_ids.add(step_id)

        if not step_desc:
            errors.append(f"Step '{step_id}': description is required.")

        if agent not in VALID_AGENTS:
            errors.append(
                f"Step '{step_id}': invalid agent '{agent}'. "
                f"Valid agents: {', '.join(VALID_AGENTS)}."
            )

        if not reasoning:
            errors.append(f"Step '{step_id}': reasoning is required.")

        # Review step validation
        if agent == "review":
            review_config = step.get("review_config")
            if not review_config:
                errors.append(
                    f"Step '{step_id}': review steps require a review_config "
                    f"with data_source_step, review_type, and prompt."
                )
            else:
                src = review_config.get("data_source_step", "")
                if src and src not in seen_ids:
                    later_ids = {s.get("step_id", "") for s in steps[i + 1 :]}
                    if src in later_ids:
                        errors.append(
                            f"Step '{step_id}': review_config.data_source_step "
                            f"'{src}' is a forward reference."
                        )
                    else:
                        errors.append(
                            f"Step '{step_id}': review_config.data_source_step "
                            f"'{src}' does not match any step_id."
                        )
                if not review_config.get("review_type"):
                    errors.append(
                        f"Step '{step_id}': review_config.review_type is required."
                    )
                if not review_config.get("prompt"):
                    errors.append(
                        f"Step '{step_id}': review_config.prompt is required."
                    )
            if not depends_on:
                errors.append(
                    f"Step '{step_id}': review steps must depend on at least "
                    f"one prior step (the data source to review)."
                )

        # Check dependency references
        for dep in depends_on:
            if dep == step_id:
                errors.append(f"Step '{step_id}': cannot depend on itself.")
            elif dep not in seen_ids:
                # Check if it's a forward reference to a later step
                later_ids = {s.get("step_id", "") for s in steps[i + 1 :]}
                if dep in later_ids:
                    errors.append(
                        f"Step '{step_id}': depends_on '{dep}' is a "
                        f"forward reference. A step can only depend on "
                        f"earlier steps."
                    )
                else:
                    errors.append(
                        f"Step '{step_id}': depends_on '{dep}' does not "
                        f"match any step_id in the plan."
                    )

    if errors:
        return {
            "error": "Plan validation failed.",
            "issues": errors,
            "hint": (
                "Fix the issues above and call create_plan again. "
                "Ensure all step_ids are unique, all agents are valid, "
                "and all depends_on references point to earlier steps."
            ),
        }

    # Cycle detection via Kahn's algorithm (topological sort)
    in_degree: dict[str, int] = {s.get("step_id", ""): 0 for s in steps}
    adjacency: dict[str, list[str]] = defaultdict(list)

    for step in steps:
        step_id = step["step_id"]
        for dep in step.get("depends_on", []):
            adjacency[dep].append(step_id)
            in_degree[step_id] += 1

    # BFS / Kahn's
    queue = [sid for sid, deg in in_degree.items() if deg == 0]
    topo_order: list[str] = []

    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        for neighbor in adjacency[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(topo_order) != len(steps):
        cycle_nodes = [sid for sid, deg in in_degree.items() if deg > 0]
        return {
            "error": "Plan contains a dependency cycle.",
            "cycle_nodes": cycle_nodes,
            "hint": (
                "Remove circular dependencies. Steps involved in the "
                f"cycle: {', '.join(cycle_nodes)}."
            ),
        }

    # Build validated plan
    plan_id = str(uuid.uuid4())
    validated_steps = []
    for step in steps:
        validated_step: dict[str, Any] = {
            "step_id": step["step_id"],
            "description": step["description"].strip(),
            "agent": step["agent"].strip(),
            "reasoning": step["reasoning"].strip(),
            "depends_on": step.get("depends_on", []),
            "status": "pending",
            "result_summary": None,
            "result_data": None,
        }
        if step.get("review_config"):
            validated_step["review_config"] = step["review_config"]
        validated_steps.append(validated_step)

    return {
        "status": "valid",
        "plan": {
            "plan_id": plan_id,
            "title": title,
            "description": description,
            "steps": validated_steps,
            "status": "draft",
            "current_step_index": 0,
        },
        "topological_order": topo_order,
        "step_count": len(validated_steps),
    }


def handle_list_agents(
    agent_catalog: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the available agent catalog.

    Args:
        agent_catalog: List of agent info dicts from the system prompt context.

    Returns:
        Agent catalog dict.
    """
    if not agent_catalog:
        # Provide the static fallback catalog
        return {
            "agents": [
                {
                    "key": "data",
                    "name": "Data Agent",
                    "description": (
                        "Queries the BV-BRC database (Solr). Searches genomes, "
                        "features, AMR data, taxonomy, specialty genes, and more. "
                        "Returns structured data results."
                    ),
                    "capabilities": [
                        "data_retrieval",
                        "solr_query",
                        "facet_query",
                    ],
                },
                {
                    "key": "service",
                    "name": "Service Agent",
                    "description": (
                        "Runs BV-BRC computational services: genome assembly, "
                        "annotation, comparative genomics, phylogenetics, "
                        "metagenomics, and more. Builds and submits workflow jobs."
                    ),
                    "capabilities": [
                        "genome_assembly",
                        "annotation",
                        "phylogenetics",
                        "comparative_genomics",
                        "metagenomics",
                    ],
                },
                {
                    "key": "workspace",
                    "name": "Workspace Agent",
                    "description": (
                        "Browses the user's BV-BRC workspace. Lists files, "
                        "reads metadata, explores directories. Read-only access."
                    ),
                    "capabilities": [
                        "workspace_browse",
                        "file_metadata",
                        "directory_listing",
                    ],
                },
                {
                    "key": "helpdesk",
                    "name": "Helpdesk Agent",
                    "description": (
                        "Answers questions about BV-BRC features, services, "
                        "documentation, and how-to guides using the knowledge base."
                    ),
                    "capabilities": [
                        "documentation",
                        "faq",
                        "how_to",
                    ],
                },
                {
                    "key": "analysis",
                    "name": "Analysis Agent",
                    "description": (
                        "Analyzes results from completed BV-BRC jobs. Reads "
                        "output files, generates summaries, and creates reports."
                    ),
                    "capabilities": [
                        "job_analysis",
                        "output_parsing",
                        "report_generation",
                    ],
                },
                {
                    "key": "review",
                    "name": "Review Checkpoint",
                    "description": (
                        "Pauses plan execution to present intermediate results "
                        "to the user for review and decision-making. Use for "
                        "data selection/filtering, choosing an analysis type, "
                        "or confirming parameters before proceeding."
                    ),
                    "capabilities": [
                        "data_review",
                        "user_checkpoint",
                        "workflow_selection",
                    ],
                },
                {
                    "key": "direct",
                    "name": "Direct (Planning Agent)",
                    "description": (
                        "Steps that the planning agent can answer directly "
                        "without delegating to another agent. Use for "
                        "summarization, explanation, or synthesis steps."
                    ),
                    "capabilities": [
                        "summarization",
                        "explanation",
                        "synthesis",
                    ],
                },
            ],
        }

    return {"agents": agent_catalog}
