"""Async LLM client for the orchestrator's routing and synthesis calls.

Wraps the OpenAI async client pointing at the vLLM instance (or any
OpenAI-compatible endpoint). Used by the router and synthesizer — NOT
by the sub-agents, which have their own LLM clients.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

from openai import AsyncOpenAI

from orchestrator.llm.config import LLMConfig
from llm_config import get_excluded_params, get_temperature_override, uses_max_completion_tokens

logger = logging.getLogger(__name__)


class LLMClient:
    """Async OpenAI-compatible LLM client for orchestrator use.

    Usage:
        config = LLMConfig()
        client = LLMClient(config)
        response = await client.complete("What is BV-BRC?")
    """

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        logger.info(
            f"Initializing LLM client: model={self.config.model!r} "
            f"base_url={self.config.base_url!r} "
            f"timeout={self.config.timeout_seconds}s"
        )
        self._client = AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout_seconds,
        )

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a completion request and return the response text.

        Args:
            prompt: The user message / prompt.
            system_prompt: Optional system message prepended to the conversation.
            temperature: Override the default temperature.
            max_tokens: Override the default max_tokens.

        Returns:
            The assistant's response text.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return await self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _build_create_kwargs(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Build the kwargs dict for chat.completions.create().

        Centralises model-specific parameter handling so that both the
        blocking ``chat()`` and streaming ``chat_stream()`` methods
        share the same logic.
        """
        excluded = get_excluded_params(self.config.model)
        resolved_temp = temperature if temperature is not None else self.config.temperature
        resolved_max = max_tokens or self.config.max_tokens

        # Apply forced temperature for models that require a specific value
        temp_override = get_temperature_override(self.config.model)
        if temp_override is not None:
            resolved_temp = temp_override

        create_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,  # type: ignore[arg-type]
        }
        if "temperature" not in excluded and resolved_temp is not None:
            create_kwargs["temperature"] = resolved_temp

        # Use max_completion_tokens for models that require it
        if resolved_max is not None:
            if uses_max_completion_tokens(self.config.model):
                create_kwargs["max_completion_tokens"] = resolved_max
            elif "max_tokens" not in excluded:
                create_kwargs["max_tokens"] = resolved_max

        if excluded or uses_max_completion_tokens(self.config.model):
            logger.debug(
                f"LLM request: model={self.config.model!r} "
                f"excluded_params={excluded} "
                f"uses_max_completion_tokens={uses_max_completion_tokens(self.config.model)} "
                f"messages={len(messages)}"
            )
        else:
            logger.debug(
                f"LLM request: model={self.config.model!r} "
                f"base_url={self.config.base_url!r} "
                f"messages={len(messages)} "
                f"temperature={resolved_temp} "
                f"max_tokens={resolved_max}"
            )

        return create_kwargs

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat completion request with explicit messages.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Override the default temperature.
            max_tokens: Override the default max_tokens.

        Returns:
            The assistant's response text.
        """
        try:
            create_kwargs = self._build_create_kwargs(messages, temperature, max_tokens)
            response = await self._client.chat.completions.create(**create_kwargs)

            content = response.choices[0].message.content or ""
            logger.info(
                f"LLM response: {len(content)} chars, "
                f"preview={content[:80]!r}"
            )
            return content.strip()

        except Exception as e:
            logger.error(
                f"LLM call failed: {e} | "
                f"base_url={self.config.base_url!r} "
                f"model={self.config.model!r}"
            )
            raise

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion, yielding text chunks as they arrive.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Override the default temperature.
            max_tokens: Override the default max_tokens.

        Yields:
            Text content deltas (strings). The caller is responsible for
            concatenating them into the full response.
        """
        try:
            create_kwargs = self._build_create_kwargs(messages, temperature, max_tokens)
            create_kwargs["stream"] = True

            stream = await self._client.chat.completions.create(**create_kwargs)

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

        except Exception as e:
            logger.error(
                f"LLM stream call failed: {e} | "
                f"base_url={self.config.base_url!r} "
                f"model={self.config.model!r}"
            )
            raise

    async def complete_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a completion request, yielding text chunks as they arrive.

        Convenience wrapper around ``chat_stream()`` that builds the
        messages list from a prompt and optional system prompt.

        Args:
            prompt: The user message / prompt.
            system_prompt: Optional system message.
            temperature: Override the default temperature.
            max_tokens: Override the default max_tokens.

        Yields:
            Text content deltas (strings).
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async for chunk in self.chat_stream(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield chunk

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()
