"""Response synthesizer.

After agent execution, generates a final natural-language response.
For single-agent single-tool calls with a good answer, passes through
the agent's response directly (skips the unnecessary LLM call).
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from orchestrator.events.events import Event, EventType
from orchestrator.llm.client import LLMClient
from orchestrator.models import OrchestratorRequest
from orchestrator.synthesizer.prompts import build_synthesis_prompt

logger = logging.getLogger(__name__)


_LINK_LABELS = (
    ("viewer_url", "view results"),
    ("download_tsv_url", "download TSV"),
    ("workspace_browser_url", "open in workspace"),
    ("jobs_page_url", "jobs page"),
)


def _describe_tool_result(call: dict[str, Any]) -> str | None:
    """One user-facing line from a tool trace entry's result_summary."""
    summary = call.get("result_summary") or {}
    tool = call.get("tool", "lookup")
    if call.get("error") and not summary:
        return None
    facts = []
    for key in ("numFound", "count", "total"):
        if isinstance(summary.get(key), int):
            facts.append(f"{summary[key]:,} records")
            break
    distinct = summary.get("distinct_counts")
    if isinstance(distinct, dict):
        for field, n in distinct.items():
            if isinstance(n, int):
                facts.append(f"{n:,} distinct {field}")
    links = [
        f"[{label}](<{summary[key]}>)"
        for key, label in _LINK_LABELS
        if isinstance(summary.get(key), str)
    ]
    if not facts and not links:
        return None
    return f"- `{tool}`: " + "; ".join(facts + links)


def _build_fallback(
    agent_results: list[dict[str, Any]],
    error: str | None = None,
) -> str:
    """Build a fallback response when synthesis fails or returns empty."""
    parts: list[str] = []

    for result in agent_results:
        answer = result.get("answer", "")
        agent = result.get("agent", "unknown")
        if answer:
            parts.append(f"**{agent}**: {answer}")
            continue

        # No textual answer: say what the tools found, not what they were
        # called with (the old fallback printed the raw query arguments).
        found_lines = []
        for call in result.get("tool_trace") or []:
            line = _describe_tool_result(call)
            if line:
                found_lines.append(line)
        if found_lines:
            parts.append(
                "I ran the lookups for this but wasn't able to finish writing "
                "up the answer. Here is what they returned:\n\n"
                + "\n".join(found_lines[-8:])
                + "\n\nAsk me to summarize these results, or try the "
                "question again."
            )

    if parts:
        return "\n\n".join(parts)

    if error:
        return (
            "I encountered an error generating a response. "
            "Please try your question again."
        )

    return (
        "The agent completed its work but was unable to produce a summary. "
        "Please try rephrasing your question or breaking it into smaller steps."
    )


async def synthesize(
    request: OrchestratorRequest,
    agent_results: list[dict[str, Any]],
    llm: LLMClient,
    force_llm: bool = False,
) -> AsyncGenerator[Event, None]:
    """Synthesize a final response from agent results.

    For single-agent calls with a successful result, passes through the
    agent's answer directly. For multi-agent results, error recovery, or
    when force_llm is True, uses the LLM to synthesize a unified response.

    Args:
        request: The original orchestrator request.
        agent_results: List of result dicts from agent execution.
        llm: LLM client for synthesis calls.
        force_llm: Force LLM synthesis even for single results.

    Yields:
        SYNTHESIS_START, SYNTHESIS_CHUNK/SYNTHESIS_DONE events.
    """
    yield Event(
        type=EventType.SYNTHESIS_START,
        data={"result_count": len(agent_results)},
    )

    # --- Pass-through for a single agent result ---
    # Also pass through "needs_input" so the agent's clarification
    # question reaches the user verbatim (no LLM reformulation), and
    # "error": error answers are user-facing already (agent_executor's
    # user_facing_agent_error, or the agent's own explanation). Sending
    # them through LLM synthesis cost another slow call on an already
    # failed turn, and when that came back empty the fallback printed
    # "**data**: <raw error>".
    if (
        not force_llm
        and len(agent_results) == 1
        and agent_results[0].get("status")
        in ("completed", "max_iterations", "needs_input", "error")
        and agent_results[0].get("answer")
    ):
        answer = agent_results[0]["answer"]
        logger.info(
            f"Synthesis pass-through: {len(answer)} chars, "
            f"preview={answer[:80]!r}"
        )

        # Append workflow manifest if the agent produced one.
        # If the manifest contains a cwl_document, format it as YAML
        # (idiomatic for CWL v1.2).  Otherwise fall back to JSON.
        manifest = agent_results[0].get("manifest")
        if manifest:
            cwl_doc = manifest.get("cwl_document") if isinstance(manifest, dict) else None
            if cwl_doc:
                import yaml
                answer += (
                    "\n\n### CWL Workflow\n```yaml\n"
                    + yaml.dump(cwl_doc, default_flow_style=False, sort_keys=False)
                    + "```"
                )
            else:
                answer += (
                    "\n\n### Workflow Manifest\n"
                    f"```json\n{json.dumps(manifest, indent=2, default=str)}\n```"
                )

        # Emit the answer as a single synthesis chunk so the gateway
        # always receives a synthesis_chunk -> final_response mapping.
        yield Event(
            type=EventType.SYNTHESIS_CHUNK,
            data={
                "chunk": answer,
                "agent": agent_results[0].get("agent", "unknown"),
            },
        )

        yield Event(
            type=EventType.SYNTHESIS_DONE,
            data={
                "response_text": answer,
                "method": "pass_through",
                "agent": agent_results[0].get("agent", "unknown"),
            },
        )
        return

    # --- LLM synthesis for complex cases (streamed) ---
    try:
        # Build bounded conversation context from recent_messages
        # (conversation_summary is unbounded and not forwarded to agents)
        conversation_context: str | None = None
        if request.recent_messages:
            from shared.agent_utils import format_recent_messages
            conversation_context = format_recent_messages(
                request.recent_messages, max_per_message=500, max_messages=5,
            ) or None

        system_prompt, user_prompt = build_synthesis_prompt(
            query=request.query,
            agent_results=agent_results,
            conversation_context=conversation_context,
        )

        # Stream tokens as SYNTHESIS_CHUNK events so the gateway can
        # relay them to the frontend as they arrive, dramatically
        # reducing time-to-first-token for remote endpoints.
        response_text = ""
        async for chunk in llm.complete_stream(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.1,
            usage_site="synthesizer",
            # No max_tokens override: the model's configured budget
            # (Mongo modelList / llm.yaml). A thinking model spent the old
            # hardcoded 2048 on reasoning and returned empty content.
        ):
            response_text += chunk
            yield Event(
                type=EventType.SYNTHESIS_CHUNK,
                data={"chunk": chunk},
            )

        logger.info(
            f"Synthesis LLM result (streamed): {len(response_text)} chars, "
            f"preview={response_text[:80]!r}"
        )

        # If the LLM returned nothing, build a fallback from agent results
        if not response_text.strip():
            logger.warning("Synthesis LLM returned empty response, using fallback")
            response_text = _build_fallback(agent_results)
            yield Event(
                type=EventType.SYNTHESIS_CHUNK,
                data={"chunk": response_text},
            )

        yield Event(
            type=EventType.SYNTHESIS_DONE,
            data={
                "response_text": response_text,
                "method": "llm_synthesis" if response_text else "fallback",
            },
        )

    except Exception as e:
        logger.error(f"Synthesis LLM call failed: {e}")
        fallback = _build_fallback(agent_results, error=str(e))

        yield Event(
            type=EventType.SYNTHESIS_DONE,
            data={
                "response_text": fallback,
                "method": "fallback",
                "error": str(e),
            },
        )
