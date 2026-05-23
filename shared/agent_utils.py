"""Shared utility functions for all BV-BRC agents.

These functions handle common patterns in the agent LLM loop:
parsing tool calls from OpenAI responses, normalizing arguments,
building conversation history messages, and progress callbacks.

Imported by each agent via sys.path manipulation:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
    from agent_utils import parse_tool_calls, normalize_arguments, ...
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


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
