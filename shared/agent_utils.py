"""Shared utility functions for all BV-BRC agents.

These functions handle common patterns in the agent LLM loop:
parsing tool calls from OpenAI responses, normalizing arguments,
building conversation history messages, progress callbacks, and LLM
retry logic.

Imported by each agent via sys.path manipulation:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
    from agent_utils import parse_tool_calls, normalize_arguments, ...
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# LLM retry and error classification
# ---------------------------------------------------------------------------

# Phrases that indicate the request exceeded the model's context window.
_CONTEXT_ERROR_PHRASES = [
    "maximum context length",
    "context_length_exceeded",
    "too many tokens",
    "prompt is too long",
    "token limit",
    "max_tokens",
    "request too large",
]

# Phrases / status codes that indicate a transient (retryable) error.
_TRANSIENT_ERROR_PHRASES = [
    "rate_limit",
    "rate limit",
    "429",
    "too many requests",
    "server_error",
    "502",
    "503",
    "504",
    "bad gateway",
    "service unavailable",
    "gateway timeout",
    "connection error",
    "connection reset",
    "connection refused",
    "timed out",
    "timeout",
    "temporary failure",
    "overloaded",
]


def is_context_window_error(exc: BaseException) -> bool:
    """Return True if *exc* indicates the LLM's context window was exceeded.

    This is used by agent loops to catch context-overflow errors and
    gracefully synthesize a response from data collected so far, rather
    than crashing the entire agent run.
    """
    error_str = str(exc).lower()
    return any(phrase in error_str for phrase in _CONTEXT_ERROR_PHRASES)


def _is_transient_error(exc: BaseException) -> bool:
    """Return True if *exc* looks like a transient LLM API error."""
    error_str = str(exc).lower()
    # Also check for known exception types from the OpenAI SDK
    exc_type = type(exc).__name__.lower()
    return any(phrase in error_str for phrase in _TRANSIENT_ERROR_PHRASES) or any(
        phrase in exc_type
        for phrase in ["ratelimit", "timeout", "connection", "apierror"]
    )


async def llm_call_with_retry(
    fn: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    **kwargs: Any,
) -> Any:
    """Call an async LLM function with exponential backoff on transient errors.

    Retries on rate-limit (429), server errors (5xx), timeouts, and
    connection errors. Does NOT retry on context-window errors or other
    client errors (4xx) — those are re-raised immediately.

    Args:
        fn: The async callable to invoke (e.g., ``chat_completion``).
        *args: Positional arguments forwarded to *fn*.
        max_retries: Maximum number of retry attempts (default 3).
        base_delay: Initial delay in seconds (default 2.0).
        max_delay: Maximum delay cap in seconds (default 30.0).
        **kwargs: Keyword arguments forwarded to *fn*.

    Returns:
        The return value of *fn*.

    Raises:
        The last exception if all retries are exhausted, or any
        non-transient exception immediately.
    """
    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            last_exc = e

            # Context-window errors are not transient — re-raise immediately
            if is_context_window_error(e):
                raise

            # Only retry on transient errors
            if not _is_transient_error(e):
                raise

            if attempt >= max_retries:
                logger.error("LLM call failed after %d retries: %s", max_retries, e)
                raise

            delay = min(base_delay * (2**attempt), max_delay)
            logger.warning(
                "LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1,
                max_retries + 1,
                delay,
                e,
            )
            await asyncio.sleep(delay)

    # Should not reach here, but satisfy type checker
    raise last_exc  # type: ignore[misc]


def call_fingerprint(tc: Any) -> str:
    """Create a deterministic fingerprint for a tool call.

    Used to detect duplicate calls -- if the LLM emits the same tool name
    with the same arguments, it will produce the same fingerprint.

    Args:
        tc: A ToolCall object with .name and .arguments attributes.

    Returns:
        A string fingerprint in the form "tool_name::json_args".
    """
    return f"{tc.name}::{json.dumps(tc.arguments, sort_keys=True)}"


def normalize_arguments(args: dict[str, Any]) -> dict[str, Any]:
    """Normalize tool call arguments from the LLM.

    vLLM/Llama sometimes returns booleans as strings ("true"/"false"),
    integers as strings ("25"), and nulls as strings ("null"/"none").
    This coerces them to proper Python types.

    Args:
        args: Raw argument dict from the LLM.

    Returns:
        Dict with string booleans, integers, and nulls converted to
        their proper Python types.
    """
    normalized = {}
    for key, value in args.items():
        if isinstance(value, str):
            lower = value.lower()
            if lower == "true":
                normalized[key] = True
            elif lower == "false":
                normalized[key] = False
            elif lower == "null" or lower == "none":
                normalized[key] = None
            elif lower.isdigit() or (lower.startswith("-") and lower[1:].isdigit()):
                normalized[key] = int(value)
            else:
                normalized[key] = value
        else:
            normalized[key] = value
    return normalized


def parse_tool_calls(response: Any, tool_call_cls: type) -> list:
    """Extract ToolCall objects from an OpenAI ChatCompletion response.

    Args:
        response: OpenAI ChatCompletion response object.
        tool_call_cls: The ToolCall class to instantiate (avoids
            coupling this module to any specific agent's models).

    Returns:
        List of ToolCall instances parsed from the response.
    """
    choice = response.choices[0]
    if not choice.message.tool_calls:
        return []

    calls = []
    for tc in choice.message.tool_calls:
        try:
            args = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                "Failed to parse tool call arguments as JSON for %s: %s",
                tc.function.name,
                tc.function.arguments[:200] if tc.function.arguments else "",
            )
            args = {"_raw": tc.function.arguments}

        args = normalize_arguments(args)

        calls.append(
            tool_call_cls(
                id=tc.id,
                name=tc.function.name,
                arguments=args,
            )
        )
    return calls


def get_response_content(response: Any) -> str | None:
    """Extract text content from an OpenAI ChatCompletion response.

    Args:
        response: OpenAI ChatCompletion response object.

    Returns:
        The text content string, or None if empty.
    """
    choice = response.choices[0]
    return choice.message.content


def build_tool_calls_message(tool_calls: list) -> list[dict[str, Any]]:
    """Build the tool_calls list in OpenAI message format for conversation history.

    Args:
        tool_calls: List of ToolCall objects with .id, .name, .arguments.

    Returns:
        List of dicts in OpenAI tool_calls message format.
    """
    return [
        {
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.name,
                "arguments": json.dumps(tc.arguments),
            },
        }
        for tc in tool_calls
    ]


async def emit_progress(
    cb: Any, progress: float, total: float | None, message: str
) -> None:
    """Fire progress callback if provided, swallowing errors.

    Args:
        cb: Async callback with signature (progress, total, message) -> None,
            or None.
        progress: Current progress value.
        total: Total expected value (or None if indeterminate).
        message: Human-readable progress description.
    """
    if cb is not None:
        try:
            await cb(progress, total, message)
        except Exception:
            pass
