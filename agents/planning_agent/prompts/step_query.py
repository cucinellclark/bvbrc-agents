"""Prompt template for formulating step execution queries.

When the planning agent is in 'execute step' mode, it uses this prompt
to formulate a focused, self-contained query for the target agent,
incorporating context from prior step results.
"""

import json

STEP_EXECUTION_SYSTEM_PROMPT = """\
You are helping execute a step in a multi-step plan. Your job is to
formulate a clear, self-contained query for the target agent based on
the step description and any available context from prior steps.

## Plan Context

**Plan title:** {plan_title}
**Plan description:** {plan_description}
**Original user request:** {original_query}

## Current Step

**Step {step_number} of {total_steps}:** {step_description}
**Target agent:** {target_agent}
**Step reasoning:** {step_reasoning}

{prior_results_section}

## Your Task

Formulate a single, clear query for the {target_agent} agent that:

1. Is self-contained — the agent should understand what to do without
   needing to see the full plan.
2. Incorporates relevant information from prior step results (if any).
3. Is specific enough for the agent to act on immediately.
4. Does NOT mention "the plan" or "step N" — write it as if the user
   is asking the agent directly.

Respond with ONLY the formulated query text. No explanations, no
preamble — just the query the agent should receive.
"""


def build_step_execution_prompt(
    plan_title: str,
    plan_description: str,
    original_query: str,
    step_number: int,
    total_steps: int,
    step_description: str,
    target_agent: str,
    step_reasoning: str,
    prior_results: dict[str, dict] | None = None,
) -> str:
    """Build the system prompt for step execution query formulation.

    Args:
        plan_title: Title of the plan.
        plan_description: Description of the plan.
        original_query: The user's original question.
        step_number: 1-based index of the current step.
        total_steps: Total number of steps in the plan.
        step_description: Description of the current step.
        target_agent: Agent key for this step.
        step_reasoning: Why this step is needed.
        prior_results: Dict of step_id -> result summary from completed steps.

    Returns:
        Complete system prompt for the step execution LLM call.
    """
    prior_results_section = ""
    if prior_results:
        lines = ["## Results from Prior Steps", ""]
        for step_id, result in prior_results.items():
            summary = result.get("answer", result.get("result_summary", ""))
            if summary:
                if len(summary) > 4000:
                    summary = summary[:4000] + "... [truncated]"
                lines.append(f"**{step_id}:** {summary}")
                lines.append("")

            # Include structured data (IDs, counts, facets) without truncation
            structured = result.get("structured_data")
            if structured and isinstance(structured, dict):
                sd_lines = []
                if structured.get("record_count") is not None:
                    sd_lines.append(
                        f"Record count: {structured['record_count']}"
                    )
                if structured.get("collection"):
                    sd_lines.append(
                        f"Collection: {structured['collection']}"
                    )
                if structured.get("record_ids"):
                    ids = structured["record_ids"]
                    sd_lines.append(f"Record IDs ({len(ids)} total): {json.dumps(ids)}")
                if structured.get("facets"):
                    sd_lines.append(
                        f"Facets: {json.dumps(structured['facets'])}"
                    )
                if sd_lines:
                    lines.append(f"**{step_id} structured data:**")
                    lines.extend(sd_lines)
                    lines.append("")
        prior_results_section = "\n".join(lines)
    else:
        prior_results_section = (
            "## Results from Prior Steps\n\nNo prior steps have completed yet."
        )

    return STEP_EXECUTION_SYSTEM_PROMPT.format(
        plan_title=plan_title,
        plan_description=plan_description,
        original_query=original_query,
        step_number=step_number,
        total_steps=total_steps,
        step_description=step_description,
        target_agent=target_agent,
        step_reasoning=step_reasoning,
        prior_results_section=prior_results_section,
    )
