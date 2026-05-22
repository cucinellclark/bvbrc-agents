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
    if (
        not force_llm
        and len(agent_results) == 1
        and agent_results[0].get("status") in ("completed", "max_iterations")
        and agent_results[0].get("answer")
    ):
        answer = agent_results[0]["answer"]
        logger.info(
            f"Synthesis pass-through: {len(answer)} chars, "
            f"preview={answer[:80]!r}"
        )

        # Append workflow manifest JSON if the agent produced one
        manifest = agent_results[0].get("manifest")
        if manifest:
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

        yield Event(
            type=EventType.SYNTHESIS_DONE,
            data={
                "response_text": response_text,
                "method": "llm_synthesis",
            },
        )

    except Exception as e:
        logger.error(f"Synthesis LLM call failed: {e}")

        # Fallback: concatenate agent answers
        fallback_parts = []
        for result in agent_results:
            answer = result.get("answer", "")
            if answer:
                agent = result.get("agent", "unknown")
                fallback_parts.append(f"**{agent}**: {answer}")

        fallback = "\n\n".join(fallback_parts) if fallback_parts else (
            "I encountered an error generating a response. "
            "Please try your question again."
        )

        yield Event(
            type=EventType.SYNTHESIS_DONE,
            data={
                "response_text": fallback,
                "method": "fallback",
                "error": str(e),
            },
        )
