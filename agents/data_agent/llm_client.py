"""OpenAI-compatible LLM client wrapper using the OpenAI Python SDK."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from openai import AsyncOpenAI

from data_agent.models import AgentConfig
from llm_config import get_excluded_params, get_temperature_override, uses_max_completion_tokens

logger = logging.getLogger(__name__)


def create_client(config: AgentConfig) -> AsyncOpenAI:
    """Create an AsyncOpenAI client from agent config."""
    return AsyncOpenAI(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
    )


def _build_kwargs(
    config: AgentConfig,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
) -> dict[str, Any]:
    """Build kwargs for chat.completions.create().

    Centralises model-specific parameter handling so that both blocking
    and streaming callers share the same logic.
    """
    excluded = get_excluded_params(config.llm_model)

    kwargs: dict[str, Any] = {
        "model": config.llm_model,
        "messages": messages,
    }

    # Temperature handling
    temp_override = get_temperature_override(config.llm_model)
    if "temperature" not in excluded:
        kwargs["temperature"] = temp_override if temp_override is not None else config.temperature

    # Use max_completion_tokens for models that require it
    if uses_max_completion_tokens(config.llm_model):
        kwargs["max_completion_tokens"] = config.max_tokens
    elif "max_tokens" not in excluded:
        kwargs["max_tokens"] = config.max_tokens

    if excluded or uses_max_completion_tokens(config.llm_model):
        logger.debug(
            f"Model {config.llm_model!r}: excluded={excluded}, "
            f"uses_max_completion_tokens={uses_max_completion_tokens(config.llm_model)}"
        )

    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice or "auto"

    return kwargs


async def chat_completion(
    client: AsyncOpenAI,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    config: AgentConfig | None = None,
    tool_choice: str | None = None,
) -> Any:
    """
    Send a chat completion request with optional tool definitions.

    Args:
        client: AsyncOpenAI client instance.
        messages: Conversation messages.
        tools: Tool schemas (OpenAI function calling format).
        config: Agent configuration.
        tool_choice: Override tool_choice ("auto", "none", etc.).
            If None, defaults to "auto" when tools are provided.

    Returns the raw ChatCompletion response object from the SDK.
    """
    cfg = config or AgentConfig()
    kwargs = _build_kwargs(cfg, messages, tools, tool_choice)
    response = await client.chat.completions.create(**kwargs)
    return response


async def chat_completion_stream(
    client: AsyncOpenAI,
    messages: list[dict[str, Any]],
    config: AgentConfig | None = None,
    tool_choice: str | None = None,
) -> AsyncGenerator[str, None]:
    """Stream a chat completion, yielding text chunks as they arrive.

    This is used for the agent's final synthesis call (tool_choice="none")
    where no tool calls are expected — only text output. Streaming
    dramatically reduces time-to-first-token when using remote endpoints
    like the Argo Gateway API.

    Args:
        client: AsyncOpenAI client instance.
        messages: Conversation messages.
        config: Agent configuration.
        tool_choice: Should be "none" for synthesis calls.

    Yields:
        Text content deltas (strings).
    """
    cfg = config or AgentConfig()
    kwargs = _build_kwargs(cfg, messages, tools=None, tool_choice=tool_choice)
    kwargs["stream"] = True

    stream = await client.chat.completions.create(**kwargs)

    full_content = ""
    async for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            full_content += delta.content
            yield delta.content

    logger.info(
        f"LLM stream response: {len(full_content)} chars, "
        f"preview={full_content[:80]!r}"
    )
