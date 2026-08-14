"""LLM-powered router — decides which agent handles a request.

Takes an OrchestratorRequest + agent catalog from the registry, calls
the routing LLM, and returns a RoutingDecision.
"""

from __future__ import annotations

import json
import logging
import re

from orchestrator.llm.client import LLMClient
from orchestrator.models import OrchestratorRequest
from orchestrator.registry.agent_registry import AgentRegistry
from orchestrator.router.models import Plan, RoutingDecision, Step
from orchestrator.router.prompts import build_routing_prompt

logger = logging.getLogger(__name__)


async def route(
    request: OrchestratorRequest,
    registry: AgentRegistry,
    llm: LLMClient,
) -> RoutingDecision:
    """Route a user request to the appropriate agent.

    If the request has a target_agent override, skip LLM routing and
    go directly to that agent.

    Args:
        request: The inbound orchestrator request.
        registry: Agent registry with discovered agents.
        llm: LLM client for the routing call.

    Returns:
        A RoutingDecision indicating how to handle this request.
    """
    # --- Override: forced routing ---
    if request.target_agent:
        agent_key = request.target_agent
        if agent_key not in registry.agents:
            return RoutingDecision(
                decision="direct",
                direct_response=(
                    f"Requested agent '{agent_key}' is not available. "
                    f"Available agents: {', '.join(registry.agent_keys)}"
                ),
                confidence=1.0,
            )
        return RoutingDecision(
            decision="agent",
            plan=Plan(
                reasoning=f"Forced routing to '{agent_key}' via target_agent override.",
                steps=[Step(agent_key=agent_key, task=request.query)],
            ),
            confidence=1.0,
        )

    # --- No healthy agents: respond directly ---
    if not registry.healthy_agents:
        return RoutingDecision(
            decision="direct",
            direct_response=(
                "I'm sorry, but no agents are currently available to help "
                "with your request. Please try again later."
            ),
            confidence=1.0,
        )

    # --- Build conversation context ---
    context_parts: list[str] = []
    if request.conversation_summary:
        context_parts.append(request.conversation_summary)
    if request.recent_messages:
        recent = request.recent_messages[-5:]  # Last 5 messages
        for msg in recent:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                context_parts.append(f"{role}: {content[:200]}")
    conversation_context = "\n".join(context_parts) if context_parts else None

    # --- Call the routing LLM ---
    catalog = registry.catalog()
    system_prompt, user_prompt = build_routing_prompt(
        query=request.query,
        agent_catalog=catalog,
        conversation_context=conversation_context,
        page_context=request.page_context,
    )

    logger.info(
        f"Routing prompt sizes: system={len(system_prompt)} chars, "
        f"user={len(user_prompt)} chars, "
        f"llm_model={llm.config.model!r} @ {llm.config.base_url!r}, "
        f"page_context={'yes' if request.page_context else 'no'} "
        f"({len(request.page_context or '')} chars)"
    )

    max_routing_attempts = 2
    try:
        raw_response = ""
        for attempt in range(1, max_routing_attempts + 1):
            raw_response = await llm.complete(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=4096,
            )
            logger.info(
                f"Routing LLM raw response (attempt {attempt}/{max_routing_attempts}): "
                f"{len(raw_response)} chars, preview={raw_response[:120]!r}"
            )
            if raw_response and raw_response.strip():
                break
            logger.warning(
                "Routing LLM returned empty response "
                f"(attempt {attempt}/{max_routing_attempts})"
            )
        else:
            # All attempts returned empty — fall through to error handling
            raise ValueError(
                f"Routing LLM returned empty response after {max_routing_attempts} attempts"
            )
        decision = _parse_routing_response(raw_response, request.query, registry)
        dr_preview = (
            repr(decision.direct_response[:80]) if decision.direct_response else "None"
        )
        logger.info(
            f"Routing decision: type={decision.decision}, "
            f"confidence={decision.confidence}, "
            f"has_direct_response={bool(decision.direct_response)}, "
            f"direct_response_preview={dr_preview}"
        )
        return decision

    except Exception as e:
        logger.error(f"Routing LLM call failed: {e}", exc_info=True)
        return RoutingDecision(
            decision="direct",
            direct_response=(
                "I'm sorry, I encountered an error while processing your "
                "request. Please try again."
            ),
            confidence=0.0,
        )


def _parse_routing_response(
    raw: str,
    query: str,
    registry: AgentRegistry,
) -> RoutingDecision:
    """Parse the LLM's JSON response into a RoutingDecision.

    Handles common LLM quirks like markdown code fences, extra text,
    and Qwen-style ``<think>...</think>`` reasoning blocks.
    """
    # Strip Qwen3-style <think>...</think> reasoning blocks before parsing.
    # These contain chain-of-thought text that is not part of the JSON output.
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        # Remove ```json or ``` at start and ``` at end
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    # Try to find JSON in the response
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from surrounding text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end])
            except json.JSONDecodeError:
                logger.error(f"Could not parse routing response: {text[:200]}")
                raise ValueError(f"Unparseable routing response: {text[:200]}")
        elif start >= 0:
            # LLM truncated the JSON (missing closing brace) — try to repair
            fragment = text[start:]
            for suffix in ["}", '"}', '""}']:
                try:
                    data = json.loads(fragment + suffix)
                    logger.info("Repaired truncated routing JSON with '%s'", suffix)
                    break
                except json.JSONDecodeError:
                    continue
            else:
                logger.error(f"Could not repair truncated routing JSON: {text[:200]}")
                raise ValueError(f"Truncated routing response: {text[:200]}")
        else:
            logger.error(f"No JSON found in routing response: {text[:200]}")
            raise ValueError(f"No JSON in routing response: {text[:200]}")

    decision_type = data.get("decision", "direct")
    reasoning = data.get("reasoning", "")

    if decision_type == "direct":
        return RoutingDecision(
            decision="direct",
            direct_response=data.get("direct_response")
            or "I can help with that. Could you provide a bit more detail?",
            confidence=0.9,
        )

    elif decision_type == "agent":
        agent_key = data.get("agent_key", "")
        task = data.get("task", query)

        # Validate agent exists
        if agent_key not in registry.agents:
            # Try to find a close match
            for key in registry.agent_keys:
                if key in agent_key or agent_key in key:
                    agent_key = key
                    break
            else:
                logger.error(
                    f"Router selected unknown agent '{agent_key}', "
                    f"available agents: {list(registry.agent_keys)}"
                )
                raise ValueError(f"Router selected unknown agent '{agent_key}'")

        return RoutingDecision(
            decision="agent",
            plan=Plan(
                reasoning=reasoning,
                steps=[Step(agent_key=agent_key, task=task)],
            ),
            confidence=0.9,
        )

    elif decision_type == "pipeline":
        raw_steps = data.get("steps", [])
        if not raw_steps:
            logger.error("Router returned pipeline with no steps")
            raise ValueError("Router returned pipeline with no steps")

        steps: list[Step] = []
        for raw_step in raw_steps:
            agent_key = raw_step.get("agent_key", "")
            task = raw_step.get("task", query)
            depends_on = raw_step.get("depends_on", [])

            # Validate agent exists (with fuzzy match)
            if agent_key not in registry.agents:
                for key in registry.agent_keys:
                    if key in agent_key or agent_key in key:
                        agent_key = key
                        break
                else:
                    logger.error(
                        f"Pipeline step references unknown agent '{agent_key}', "
                        f"available agents: {list(registry.agent_keys)}"
                    )
                    raise ValueError(
                        f"Pipeline step references unknown agent '{agent_key}'"
                    )

            steps.append(Step(agent_key=agent_key, task=task, depends_on=depends_on))

        # Validate depends_on indices are in range
        for i, step in enumerate(steps):
            step.depends_on = [d for d in step.depends_on if 0 <= d < i]

        return RoutingDecision(
            decision="pipeline",
            plan=Plan(reasoning=reasoning, steps=steps),
            confidence=0.85,
        )

    else:
        # Unknown decision type — treat as direct
        logger.warning(
            f"Router returned unknown decision type '{decision_type}', "
            "treating as direct"
        )
        return RoutingDecision(
            decision="direct",
            direct_response=data.get("direct_response")
            or "I can help with that. Could you provide a bit more detail?",
            confidence=0.5,
        )
