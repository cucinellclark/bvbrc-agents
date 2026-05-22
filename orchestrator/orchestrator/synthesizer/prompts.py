"""Synthesis prompt templates.

Used by the synthesizer to generate a final user-facing response from
agent results. For single-agent, single-step calls, synthesis is often
skipped (the agent's answer is passed through directly).

The synthesis prompt is used when:
- Multiple agents contributed results (Phase 4)
- The agent's raw answer needs reformatting
- Additional context needs to be woven in
"""

SYNTHESIS_SYSTEM_PROMPT = """\
You are the BV-BRC Copilot. You help researchers with biological data \
and bioinformatics analyses using the BV-BRC platform.

One or more agents have processed the user's request and produced results. \
Your job is to synthesize a clear, unified response for the user based on \
the agent outputs and the original question.

Rules:
- Be concise and informative.
- If a data agent found data, summarize the key findings.
- If a service agent planned a workflow, explain what it will do.
- If a workspace agent browsed files, summarize what was found.
- If multiple agents contributed, weave their results into a coherent \
  narrative. Explain how the outputs relate to each other (e.g., "I found \
  N genomes matching your criteria, and prepared an annotation workflow \
  for them").
- If any agent encountered an error, explain what went wrong and what \
  did succeed.
- Do not fabricate data. Only use information from the agent results.
- Use markdown formatting for readability (tables, lists, bold).
- IMPORTANT: If a service agent produced a workflow manifest (JSON), you \
  MUST include it in your response as a fenced JSON code block. Do NOT \
  omit or summarize the manifest -- the user needs the exact JSON. Place \
  the manifest at the end of your response after your explanatory text.
"""


def build_synthesis_prompt(
    query: str,
    agent_results: list[dict],
    conversation_context: str | None = None,
) -> tuple[str, str]:
    """Build the system and user prompts for synthesis.

    Args:
        query: The user's original question.
        agent_results: List of agent result dicts with 'agent', 'answer',
                       'status', 'sources', etc.
        conversation_context: Optional conversation summary.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    user_parts = []

    if conversation_context:
        user_parts.append(f"## Conversation Context\n{conversation_context}\n")

    user_parts.append(f"## User's Question\n{query}\n")

    for result in agent_results:
        agent = result.get("agent", "unknown")
        answer = result.get("answer", "(no answer)")
        status = result.get("status", "unknown")
        sources = result.get("sources", [])
        manifest = result.get("manifest")

        user_parts.append(f"## Agent Result: {agent} (status: {status})")
        if sources:
            user_parts.append(f"Sources consulted: {', '.join(sources)}")
        user_parts.append(f"\n{answer}\n")

        if manifest:
            import json as _json
            user_parts.append(
                "### Workflow Manifest (include this JSON in your response)\n"
                f"```json\n{_json.dumps(manifest, indent=2, default=str)}\n```\n"
            )

    if len(agent_results) > 1:
        user_parts.append(
            "## Instructions\n"
            "Multiple agents contributed to this response as part of a pipeline. "
            "Synthesize their results into a single, coherent response. Explain "
            "how the steps connect to each other and address the user's original "
            "question."
        )
    else:
        user_parts.append(
            "## Instructions\n"
            "Synthesize a clear, helpful response for the user based on the "
            "agent result above. Respond directly to their question."
        )

    return SYNTHESIS_SYSTEM_PROMPT, "\n".join(user_parts)
