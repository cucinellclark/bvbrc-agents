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

        # No textual answer — try to summarise the tool trace
        tool_trace = result.get("tool_trace") or []
        if tool_trace:
            summary_lines = []
            for call in tool_trace:
                tool_name = call.get("tool", "unknown")
                args = call.get("arguments", {})
                brief = json.dumps(args, default=str)
                if len(brief) > 200:
                    brief = brief[:200] + "..."
                summary_lines.append(f"- `{tool_name}({brief})`")
            elapsed = result.get("elapsed_seconds", "?")
            iters = result.get("iterations_used", "?")
            parts.append(
                f"The **{agent}** agent executed {len(tool_trace)} tool call(s) "
                f"over {iters} iterations ({elapsed}s) but was unable to produce "
                f"a textual summary.\n\n"
                + "\n".join(summary_lines[:10])
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

    # --- Pass-through for single successful agent result ---
    # Also pass through "needs_input" so the agent's clarification
    # question reaches the user verbatim (no LLM reformulation).
    if (
        not force_llm
        and len(agent_results) == 1
        and agent_results[0].get("status") in ("completed", "max_iterations", "needs_input")
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
        # Build conversation context
        context_parts: list[str] = []
        if request.conversation_summary:
            context_parts.append(request.conversation_summary)
        conversation_context = "\n".join(context_parts) if context_parts else None

        system_prompt, user_prompt = build_synthesis_prompt(
            query=request.query,
            agent_results=agent_results,
            conversation_context=conversation_context,
        )

        # Stream tokens as SYNTHESIS_CHUNK events so the gateway can
        # relay them to the frontend as they arrive, dramatically
        # reducing time-to-first-token for Argo endpoints.
        response_text = ""
        async for chunk in llm.complete_stream(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=2048,
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
