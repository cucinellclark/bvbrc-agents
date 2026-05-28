"""Lightweight intent classifier for the BV-BRC Service Agent v2.

Uses a cheap/fast LLM call (configurable model) to determine whether the
user wants to plan a new workflow or perform a lifecycle operation (submit,
status, cancel, modify) on an existing one.  The classifier resolves
workflow IDs from conversation context so the user can say "submit that"
instead of providing an explicit ID.

When ``AgentConfig.classifier_model`` is set, the classifier uses that model
(e.g. gpt41mini).  When it is ``None``, the classifier falls back to the
agent's primary ``llm_model``.  The UI can also override both via
``llm_override`` in the request context -- that override always wins.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from service_agent.models import AgentConfig, Intent
from llm_config import (
    get_excluded_params,
    get_temperature_override,
    uses_max_completion_tokens,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classifier prompt (kept small — ~250 tokens)
# ---------------------------------------------------------------------------

CLASSIFIER_SYSTEM_PROMPT = """\
You are an intent classifier for a bioinformatics workflow system.

Given the user's message and conversation context, determine what they want.

## Actions
- "plan"   : Build a NEW workflow (assembly, annotation, BLAST, etc.)
- "submit" : Submit / run / execute an ALREADY-PLANNED workflow
- "status" : Check the status of a submitted workflow or job
- "cancel" : Cancel / abort a running or pending workflow
- "modify" : Change parameters of a planned (not yet submitted) workflow
- "unknown": Cannot determine intent

## submit_after_plan
If the user wants to build a NEW workflow AND ALSO submit/run/execute it \
immediately, set action to "plan" and set "submit_after_plan" to true.
Phrases that indicate this:
- "assemble X and submit the job"
- "run assembly on X and execute it"
- "annotate this genome and submit"
- "... and run it", "... and start it", "... and launch it"
When in doubt (user only says "assemble X" with no submit language), \
set "submit_after_plan" to false.

## Workflow context
Conversation context may contain workflow references like:
  [workflow: wf_abc123 | ServiceName | planned]
  [workflow: wf_def456 | ServiceName | submitted]

Use these to resolve which workflow the user is referring to.

## Disambiguation rules
- If the user says "run assembly on X" and there is NO matching planned \
workflow in context, this is a "plan" request.
- If there IS a matching planned workflow, this is a "submit" request.
- If the user explicitly mentions a workflow_id (wf_...), use it directly.
- If there is exactly ONE planned workflow and the user says "submit that" \
or "run it", resolve to that workflow.
- If multiple planned workflows exist and the user is ambiguous, set \
action to "unknown" so the agent can ask for clarification.

## Response format
Return ONLY a JSON object (no markdown, no explanation):
{"action": "plan|submit|status|cancel|modify|unknown", \
"workflow_id": "wf_... or null", \
"confidence": 0.0-1.0, \
"submit_after_plan": true|false, \
"reasoning": "brief explanation"}
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def classify_intent(
    query: str,
    context: dict[str, Any],
    config: AgentConfig,
) -> Intent:
    """Classify the user's intent with a lightweight LLM call.

    Args:
        query: The user's natural language request.
        context: Conversation context dict (may contain workflow refs,
            conversation_summary, recent_messages, etc.).
        config: Agent configuration (provides model settings).

    Returns:
        An Intent object with action, workflow_id, confidence, reasoning.
    """
    # Determine which model to use for classification
    classifier_model = config.classifier_model or config.llm_model

    client = AsyncOpenAI(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
    )

    # Build a compact context summary for the classifier
    context_summary = _build_context_summary(context)

    user_message = query
    if context_summary:
        user_message = f"## Context\n{context_summary}\n\n## User Request\n{query}"

    messages = [
        {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # Build model-aware kwargs
    kwargs = _build_classifier_kwargs(classifier_model, config, messages)

    try:
        response = await client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        intent = _parse_classifier_response(content)
        logger.info(
            "Intent classified: action=%s, workflow_id=%s, confidence=%.2f, "
            "model=%s, reasoning=%s",
            intent.action, intent.workflow_id, intent.confidence,
            classifier_model, intent.reasoning,
        )
        return intent

    except Exception as e:
        logger.warning(
            "Intent classifier failed (%s: %s), defaulting to 'plan'",
            type(e).__name__, e,
        )
        return Intent(
            action="plan",
            confidence=0.5,
            reasoning=f"Classifier error: {e}; defaulting to plan",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_context_summary(context: dict[str, Any]) -> str:
    """Extract workflow references and recent conversation from context."""
    parts: list[str] = []

    # Conversation summary (if available)
    summary = context.get("conversation_summary")
    if summary:
        parts.append(summary)

    # Recent messages (last 3 for brevity)
    recent = context.get("recent_messages")
    if recent and isinstance(recent, list):
        for msg in recent[-3:]:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if content:
                # Truncate long messages
                if len(content) > 200:
                    content = content[:200] + "..."
                parts.append(f"{role}: {content}")

    return "\n".join(parts)


def _build_classifier_kwargs(
    model: str,
    config: AgentConfig,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build API kwargs with model-specific parameter handling."""
    excluded = get_excluded_params(model)

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }

    # Temperature — use low value for deterministic classification
    temp_override = get_temperature_override(model)
    if "temperature" not in excluded:
        kwargs["temperature"] = temp_override if temp_override is not None else 0.0

    # Token limit — classification responses are tiny (~50 tokens)
    if uses_max_completion_tokens(model):
        kwargs["max_completion_tokens"] = 256
    elif "max_tokens" not in excluded:
        kwargs["max_tokens"] = 256

    return kwargs


def _parse_classifier_response(content: str) -> Intent:
    """Parse the classifier's JSON response into an Intent.

    Handles common LLM quirks: markdown code fences, trailing text, etc.
    """
    # Strip markdown code fences if present
    text = content.strip()
    if text.startswith("```"):
        # Remove opening fence (with optional language tag)
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Classifier returned non-JSON: %r", content[:200])
        return Intent(
            action="plan",
            confidence=0.3,
            reasoning=f"Could not parse classifier response: {content[:100]}",
        )

    action = data.get("action", "plan")
    valid_actions = {"plan", "submit", "status", "cancel", "modify", "unknown"}
    if action not in valid_actions:
        action = "plan"

    return Intent(
        action=action,
        workflow_id=data.get("workflow_id"),
        confidence=float(data.get("confidence", 0.8)),
        reasoning=data.get("reasoning", ""),
        submit_after_plan=bool(data.get("submit_after_plan", False)),
    )
