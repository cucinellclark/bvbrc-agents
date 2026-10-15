"""LLM-powered router — decides which agent handles a request.

Takes an OrchestratorRequest + agent catalog from the registry, calls
the routing LLM, and returns a RoutingDecision.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

from orchestrator.llm.client import LLMClient
from orchestrator.models import OrchestratorRequest
from orchestrator.registry.agent_registry import AgentRegistry
from orchestrator.router.models import Plan, RoutingDecision, Step
from orchestrator.router.prompts import build_routing_prompt

# The orchestrator's import chain runs before agent_dispatch.py puts the
# bvbrc-agents repo root on sys.path (same as documents.py).
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from shared.agent_utils import format_recent_messages  # noqa: E402

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
    # Only use recent_messages (already token-budgeted by the gateway's
    # selectRecentMessages, ~4000 tokens).  conversation_summary is
    # unbounded and can contain the full content of every message in the
    # session, which overflows the routing LLM's context window on long
    # conversations.  The router only needs enough context to classify
    # intent — the current query (in ## User Request) plus a few
    # truncated recent messages is sufficient.
    # Last 5 messages, start and end of each: the router only classifies,
    # but "do the next steps" / "submit it" refer to the END of the last
    # reply, which a head-only cut dropped.
    conversation_context = format_recent_messages(
        request.recent_messages,
        max_per_message=700,
        max_messages=5,
        max_last_assistant=1500,
    ) or None

    # --- Call the routing LLM ---
    catalog = registry.catalog()
    system_prompt, user_prompt = build_routing_prompt(
        query=request.query,
        agent_catalog=catalog,
        conversation_context=conversation_context,
        page_context=request.page_context,
        has_images=bool(request.images),
        has_documents=bool(request.parsed_documents),
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
            # max_tokens is deliberately not pinned here: it comes from the
            # routing client's own config (orchestrator.routing_max_tokens).
            # The old hard-coded 16384 existed to leave room for a <think>
            # block that is now disabled, and it silently overrode the
            # client's configured budget.
            raw_response = await llm.complete(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                usage_site="router",
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


# One rule for "direct" answers (2026-10-12).  The router model answers
# direct from memory, and on Oct 9 it got BV-BRC facts wrong even after the
# prompt said not to: genome ids "look like GCF_…" (calling 5482.50 a typo),
# the microbiome pipeline is "16S/SILVA".  A message that mentions BV-BRC or
# carries a BV-BRC-style id goes to the helpdesk agent instead.  Each
# override is logged as ROUTER_DIRECT_OVERRIDE and stored on the message.
_BVBRC_REFERENCE_RE = re.compile(
    r"bv-?brc|patric|\b\d+\.\d+\b|fig\||\b[SED]RR\d+",
    re.IGNORECASE,
)

FALLBACK_AGENT = "helpdesk"


def _direct_or_helpdesk(
    direct_response: str | None,
    query: str,
    registry: AgentRegistry,
    confidence: float,
) -> RoutingDecision:
    """Return the router's direct answer, or send the query to helpdesk.

    Helpdesk is used when the router gave no text (this used to reach users
    as "I can help with that. Could you provide a bit more detail?") or when
    the query refers to BV-BRC.
    """
    text = (direct_response or "").strip()
    if not text:
        override = "empty_direct"
    elif _BVBRC_REFERENCE_RE.search(query):
        override = "bvbrc_reference"
    else:
        return RoutingDecision(decision="direct", direct_response=text, confidence=confidence)

    if FALLBACK_AGENT not in registry.agents:
        logger.warning("ROUTER_DIRECT_OVERRIDE skipped (%s): no %s agent", override, FALLBACK_AGENT)
        return RoutingDecision(
            decision="direct",
            direct_response=text or "Sorry, I couldn't work out how to help with that. Could you rephrase it?",
            confidence=confidence,
        )
    logger.warning("ROUTER_DIRECT_OVERRIDE reason=%s query=%r", override, query[:120])
    return RoutingDecision(
        decision="agent",
        plan=Plan(
            reasoning=f"Direct answer replaced ({override}); sent to {FALLBACK_AGENT}.",
            steps=[Step(agent_key=FALLBACK_AGENT, task=query)],
        ),
        confidence=confidence,
        routing_override=override,
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
        return _direct_or_helpdesk(data.get("direct_response"), query, registry, 0.9)

    elif decision_type == "agent":
        agent_key = data.get("agent_key", "")
        # Always use the original user query so pasted data (FASTA sequences,
        # file paths, accession IDs, etc.) is never lost.  The router's
        # summarised "task" is only used for logging/debugging.
        router_task = data.get("task", "")
        if router_task and router_task != query:
            logger.info(
                "Router produced a summarised task; using original query instead. "
                "Router task: %s",
                router_task[:120],
            )
        task = query

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
        # Retired (D4 Phase 3).  A model still answering in the old format
        # wanted several agents in a row: that is the planning agent's job.
        logger.info("Router returned the retired 'pipeline' decision; routing to planning")
        if "planning" in registry.agents:
            return RoutingDecision(
                decision="agent",
                plan=Plan(reasoning=reasoning, steps=[Step(agent_key="planning", task=query)]),
                confidence=0.8,
                routing_override="pipeline_to_planning",
            )
        raise ValueError("Router returned 'pipeline' and no planning agent is registered")

    else:
        # Unknown decision type — treat as direct
        logger.warning(
            f"Router returned unknown decision type '{decision_type}', "
            "treating as direct"
        )
        return _direct_or_helpdesk(data.get("direct_response"), query, registry, 0.5)
