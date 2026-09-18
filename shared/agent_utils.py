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
import random
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
# NOTE: timeout phrases are deliberately excluded here — they are handled
# separately by _is_timeout_error() with a more conservative retry policy.
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
    "connection error",
    "connection reset",
    "connection refused",
    "temporary failure",
    "overloaded",
]

# Phrases that indicate a non-retryable admission rejection.  The holly
# admission proxy returns 429 with type "admission_queue_full" when its
# wait queue is full.  Retrying this would double-book the GPU — the queue
# is still full, and a retry just adds another waiter.
_ADMISSION_REJECT_PHRASES = [
    "admission_queue_full",
]

# Phrases that indicate a timeout (queue wait + generation exceeded budget).
# These get at most 1 retry with a long jittered delay to avoid piling onto
# an already-overloaded vLLM engine.
_TIMEOUT_ERROR_PHRASES = [
    "timed out",
    "timeout",
    "read timeout",
    "connect timeout",
]


def is_context_window_error(exc: BaseException) -> bool:
    """Return True if *exc* indicates the LLM's context window was exceeded.

    This is used by agent loops to catch context-overflow errors and
    gracefully synthesize a response from data collected so far, rather
    than crashing the entire agent run.
    """
    error_str = str(exc).lower()
    return any(phrase in error_str for phrase in _CONTEXT_ERROR_PHRASES)


def _is_timeout_error(exc: BaseException) -> bool:
    """Return True if *exc* looks like an HTTP / read timeout.

    Timeouts are separated from other transient errors because they
    indicate the vLLM engine is overloaded (queue wait + thinking tokens
    exceeded the client budget).  Retrying aggressively makes this worse.
    """
    error_str = str(exc).lower()
    exc_type = type(exc).__name__.lower()
    return (
        any(phrase in error_str for phrase in _TIMEOUT_ERROR_PHRASES)
        or "timeout" in exc_type
    )


def _is_admission_rejection(exc: BaseException) -> bool:
    """Return True if *exc* is a non-retryable admission proxy rejection.

    The admission proxy returns 429 with ``"type": "admission_queue_full"``
    when its bounded wait queue is full.  This must NOT be retried — the
    queue is full and retrying would just add another waiter, potentially
    double-booking GPU capacity.
    """
    error_str = str(exc).lower()
    return any(phrase in error_str for phrase in _ADMISSION_REJECT_PHRASES)


def _is_transient_error(exc: BaseException) -> bool:
    """Return True if *exc* looks like a transient LLM API error.

    NOTE: timeout errors are NOT classified as transient here — they
    are handled separately with a more conservative retry policy.
    See ``_is_timeout_error``.

    Admission proxy rejections (``admission_queue_full``) are also
    excluded — they are non-retryable by design.
    """
    # Admission rejections look like a 429 but must not be retried.
    if _is_admission_rejection(exc):
        return False

    error_str = str(exc).lower()
    # Also check for known exception types from the OpenAI SDK
    exc_type = type(exc).__name__.lower()
    return any(phrase in error_str for phrase in _TRANSIENT_ERROR_PHRASES) or any(
        phrase in exc_type
        for phrase in ["ratelimit", "connection", "apierror"]
    )


async def llm_call_with_retry(
    fn: Callable[..., Any],
    *args: Any,
    max_retries: int = 1,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    timeout_max_retries: int = 1,
    timeout_delay_min: float = 10.0,
    timeout_delay_max: float = 30.0,
    **kwargs: Any,
) -> Any:
    """Call an async LLM function with exponential backoff on transient errors.

    **Transient errors** (rate-limit 429, server errors 5xx, connection
    resets) are retried up to *max_retries* times with exponential backoff
    starting at *base_delay*.

    **Timeout errors** are retried at most *timeout_max_retries* time
    (default 1) with a long jittered delay (default 10-30 s).  Aggressive
    retry on timeouts worsens GPU overload — the vLLM engine is already
    backed up, and piling on more requests makes queue wait longer for
    everyone.

    Does NOT retry on context-window errors, admission proxy rejections
    (``admission_queue_full``), or other client errors (4xx) — those are
    re-raised immediately.

    Args:
        fn: The async callable to invoke (e.g., ``chat_completion``).
        *args: Positional arguments forwarded to *fn*.
        max_retries: Max retry attempts for transient (non-timeout) errors
            (default 1).
        base_delay: Initial delay in seconds for transient retries
            (default 2.0).
        max_delay: Maximum delay cap in seconds for transient retries
            (default 30.0).
        timeout_max_retries: Max retry attempts for timeout errors
            (default 1).
        timeout_delay_min: Minimum jittered delay for timeout retries
            (default 10.0 s).
        timeout_delay_max: Maximum jittered delay for timeout retries
            (default 30.0 s).
        **kwargs: Keyword arguments forwarded to *fn*.

    Returns:
        The return value of *fn*.

    Raises:
        The last exception if all retries are exhausted, or any
        non-transient/non-timeout exception immediately.
    """
    last_exc: BaseException | None = None
    transient_attempts = 0
    timeout_attempts = 0

    max_total_attempts = max(max_retries, timeout_max_retries) + 1 + max_retries + timeout_max_retries
    for _ in range(max_total_attempts):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            last_exc = e

            # Context-window errors are not transient — re-raise immediately
            if is_context_window_error(e):
                raise

            # --- Timeout errors: conservative retry (once, long delay) ---
            if _is_timeout_error(e):
                timeout_attempts += 1
                if timeout_attempts > timeout_max_retries:
                    logger.error(
                        "LLM call timed out after %d timeout retries: %s",
                        timeout_max_retries,
                        e,
                    )
                    raise

                delay = random.uniform(timeout_delay_min, timeout_delay_max)
                logger.warning(
                    "LLM call timed out (timeout attempt %d/%d), "
                    "retrying in %.1fs: %s",
                    timeout_attempts,
                    timeout_max_retries,
                    delay,
                    e,
                )
                await asyncio.sleep(delay)
                continue

            # --- Other transient errors: standard exponential backoff ---
            if not _is_transient_error(e):
                raise

            transient_attempts += 1
            if transient_attempts > max_retries:
                logger.error(
                    "LLM call failed after %d retries: %s", max_retries, e
                )
                raise

            delay = min(base_delay * (2 ** (transient_attempts - 1)), max_delay)
            logger.warning(
                "LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                transient_attempts,
                max_retries,
                delay,
                e,
            )
            await asyncio.sleep(delay)

    # Should not reach here, but satisfy type checker
    raise last_exc  # type: ignore[misc]


async def llm_stream_with_retry(
    fn: Callable[..., Any],
    *args: Any,
    max_retries: int = 1,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    timeout_max_retries: int = 1,
    timeout_delay_min: float = 10.0,
    timeout_delay_max: float = 30.0,
    label: str = "stream",
    **kwargs: Any,
) -> Any:
    """Open an LLM streaming connection with retry on transient errors.

    Retry logic applies only to the **stream-open** phase — the initial
    ``await client.chat.completions.create(stream=True, ...)`` call.
    Once the stream is established and chunks are flowing, errors are
    propagated immediately (retrying mid-stream would lose context).

    The returned value is the async stream object.  The caller iterates
    it with ``async for chunk in stream:``.

    Uses the same retry policy as ``llm_call_with_retry``:
    - Transient errors (429, 5xx, connection reset): up to *max_retries*
      with exponential backoff.
    - Timeout errors: up to *timeout_max_retries* (default 1) with
      jittered delay.
    - Context-window errors: never retried.
    - Admission proxy rejections (``admission_queue_full``): never retried.
    """
    last_exc: BaseException | None = None
    transient_attempts = 0
    timeout_attempts = 0

    max_total_attempts = (
        max(max_retries, timeout_max_retries)
        + 1
        + max_retries
        + timeout_max_retries
    )
    for _ in range(max_total_attempts):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            last_exc = e

            if is_context_window_error(e):
                raise

            if _is_timeout_error(e):
                timeout_attempts += 1
                if timeout_attempts > timeout_max_retries:
                    logger.error(
                        "[%s] Stream open timed out after %d timeout retries: %s",
                        label,
                        timeout_max_retries,
                        e,
                    )
                    raise

                delay = random.uniform(timeout_delay_min, timeout_delay_max)
                logger.warning(
                    "[%s] Stream open timed out (timeout attempt %d/%d), "
                    "retrying in %.1fs: %s",
                    label,
                    timeout_attempts,
                    timeout_max_retries,
                    delay,
                    e,
                )
                await asyncio.sleep(delay)
                continue

            if not _is_transient_error(e):
                raise

            transient_attempts += 1
            if transient_attempts > max_retries:
                logger.error(
                    "[%s] Stream open failed after %d retries: %s",
                    label,
                    max_retries,
                    e,
                )
                raise

            delay = min(base_delay * (2 ** (transient_attempts - 1)), max_delay)
            logger.warning(
                "[%s] Stream open failed (attempt %d/%d), retrying in %.1fs: %s",
                label,
                transient_attempts,
                max_retries,
                delay,
                e,
            )
            await asyncio.sleep(delay)

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


def build_user_content(
    query: str, images: list[str] | None = None
) -> str | list[dict[str, Any]]:
    """Build user message content, with multimodal blocks if images present.

    When *images* is empty or ``None`` the plain *query* string is returned
    (backward-compatible with all existing agents).  When images are provided
    the return value is an OpenAI-compatible list of content blocks::

        [
            {"type": "text", "text": "<query>"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
            ...
        ]

    Args:
        query: The user's text query.
        images: Optional list of base64 data-URI strings.

    Returns:
        Plain string or list of content-block dicts.
    """
    if not images:
        return query
    content: list[dict[str, Any]] = [{"type": "text", "text": query}]
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": img}})
    return content


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


# ---------------------------------------------------------------------------
# Conversation context formatting (for system prompt injection)
# ---------------------------------------------------------------------------


def format_attached_documents(
    documents: list[dict[str, Any]] | None,
) -> str:
    """Format ``parsed_documents`` into an ``=== ATTACHED DOCUMENTS ===`` section.

    For each document, renders the name, page/char counts, the workspace
    path, and the 25 KB excerpt.  For a **failed** document, renders the
    error plus an instruction to surface it to the user.

    Handles both PDFs (with ``page_count``) and text uploads (without).
    Reads ``workspace_path``, falling back to ``workspace_txt_path`` for
    backward compatibility.

    Returns an empty string if *documents* is falsy or empty.
    """
    if not documents:
        return ""

    parts: list[str] = ["=== ATTACHED DOCUMENTS ==="]

    for i, doc in enumerate(documents, 1):
        name = doc.get("name", f"Document {i}")

        if doc.get("error"):
            parts.append(
                f"\n### Document {i}: {name}\n"
                f"**ERROR:** {doc['error']}\n"
                f"You MUST tell the user this document could not be read "
                f"and explain the reason above."
            )
            continue

        page_count = doc.get("page_count")
        char_count = doc.get("char_count", "?")
        ws_path = doc.get("workspace_path") or doc.get("workspace_txt_path")
        excerpt = doc.get("excerpt", "")
        source = doc.get("source", "pdf")

        # Build header — include page count only for PDFs
        if page_count is not None:
            header = f"\n### Document {i}: {name} ({page_count} pages, {char_count} chars)"
        else:
            header = f"\n### Document {i}: {name} ({char_count} chars)"

        if ws_path:
            header += f"\nFull file saved to: {ws_path}"
            saved_as = doc.get("saved_as")
            if saved_as:
                header += (
                    f"\n(Stored as `{saved_as}` — another attachment in this "
                    f"message had the same filename.)"
                )
            header += (
                "\nIf the user asks about content beyond this excerpt, "
                "use `read_file_preview` on the path above."
            )
            if source == "upload":
                header += (
                    "\nThis is the original uploaded file (not an extract). "
                    "You may pass this path as a GoWe workflow input."
                )

        parts.append(header)
        if excerpt:
            parts.append(f"\n--- Excerpt (first {len(excerpt)} chars) ---\n{excerpt}")

    return "\n".join(parts)


def format_recent_messages(
    recent_messages: list[dict[str, Any]] | None,
    max_per_message: int = 500,
    max_messages: int = 5,
) -> str:
    """Format a ``recent_messages`` list into a compact conversation context.

    Used by agents to inject bounded conversation context into their system
    prompt.  Each message is truncated to *max_per_message* characters to
    keep the context concise.

    Returns an empty string if *recent_messages* is falsy or empty.
    """
    if not recent_messages:
        return ""
    parts: list[str] = []
    for msg in recent_messages[-max_messages:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not content:
            continue
        if len(content) > max_per_message:
            content = content[:max_per_message] + "..."
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Session workspace prompt injection
# ---------------------------------------------------------------------------


def build_session_workspace_path(
    workspace_path: str | None,
    session_id: str | None,
) -> str | None:
    """Construct the absolute session workspace path, or ``None`` if missing.

    This is the canonical location for uploads, PDF extracts, and GoWe job
    outputs created during this chat session::

        /<user>/home/chats/<session_id>/

    The same path is used by ``submit_gowe_job()`` and ``persist_session_file()``
    when writing files.
    """
    if not workspace_path or not session_id:
        return None
    return f"{workspace_path.rstrip('/')}/chats/{session_id}"


def format_session_workspace(context: dict[str, Any] | None) -> str:
    """Build an ``=== SESSION WORKSPACE ===`` section for the system prompt.

    Returns the section text when both ``workspace_path`` and ``session_id``
    are present in *context*, otherwise returns an empty string (MCP callers,
    tests, or missing gateway context).
    """
    if not context:
        return ""
    session_path = build_session_workspace_path(
        context.get("workspace_path"),
        context.get("session_id"),
    )
    if not session_path:
        return ""
    return (
        "=== SESSION WORKSPACE ===\n"
        "This chat's workspace folder is:\n"
        f"  {session_path}\n"
        "\n"
        "Uploads, PDF extracts, and jobs submitted from this chat land here.\n"
        "This folder may not exist yet if no files have been uploaded or jobs\n"
        "submitted in this chat. If you get a 'not found' error browsing it,\n"
        "skip it and fall back to home.\n"
        "\n"
        "When looking for files from this conversation, call workspace_browse\n"
        "on this path FIRST (copy it exactly). If nothing relevant is there,\n"
        "then browse home or the folder the user named.\n"
        "\n"
        "Genome Groups, Feature Groups, and user-named folders still live at\n"
        "home — do not look for those only under this path.\n"
        "\n"
        "Do not invent other session UUIDs. Do not search sibling folders\n"
        "under chats/ unless the user asks."
    )


def format_execution_mode(source: Any) -> str:
    """Build an ``=== EXECUTION MODE ===`` section for the system prompt.

    *source* may be an agent config (``config.execution_mode``) or a context
    dict (``context["execution_mode"]``).  Anything other than ``"execute"``
    is treated as ``"plan"`` — the safe default.  The gate itself lives in
    ``shared.tools.execute_tool``; this section only tells the LLM what to
    expect so it does not retry or hallucinate a submission.
    """
    if isinstance(source, dict):
        mode = source.get("execution_mode")
    else:
        mode = getattr(source, "execution_mode", None)
    if mode == "execute":
        return (
            "=== EXECUTION MODE ===\n"
            "This session is in EXECUTE mode. submit_gowe_job and create_group\n"
            "are enabled. Still ask before acting when the workflow or group\n"
            "choice is ambiguous or a required input is missing."
        )
    return (
        "=== EXECUTION MODE ===\n"
        "This session is in PLAN mode. submit_gowe_job and create_group are\n"
        "disabled and return an error if called; every other tool works.\n"
        "You may fully prepare a job or group (discover, browse, verify,\n"
        "populate inputs) and present it as ready, but you must never say\n"
        "it was submitted or created. Tell the user to switch to Execute\n"
        "mode using the Plan/Execute toggle next to the message box when\n"
        "they want it run."
    )


# ---------------------------------------------------------------------------
# Message trimming (context window management)
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token for English/JSON."""
    return len(text) // 4


def estimate_messages_tokens(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> int:
    """Estimate the total token count of a messages list + tool schemas."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if content:
            total += estimate_tokens(
                content if isinstance(content, str) else json.dumps(content, default=str)
            )
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            total += estimate_tokens(json.dumps(tool_calls, default=str))
    if tools:
        total += estimate_tokens(json.dumps(tools, default=str))
    return total


def trim_messages_to_fit(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    max_tokens: int,
) -> list[dict[str, Any]]:
    """Trim conversation history to fit within max_tokens.

    Strategy (preserves correctness of tool-call / tool-result pairing):
      1. Always keep the system message(s) and the initial user message.
      2. Always keep the most recent assistant+tool-result exchange (the LLM
         needs it to know what just happened).
      3. When over budget, progressively replace older tool-result messages
         with a compact summary, starting from the oldest.
      4. If an assistant message references tool_calls whose results are
         dropped, drop that assistant message too (the LLM would be confused
         by dangling references).
    """
    current = estimate_messages_tokens(messages, tools)
    if current <= max_tokens:
        return messages

    result = list(messages)

    # Find the boundary: everything from the last assistant message onward is pinned
    last_assistant_idx = None
    for i in range(len(result) - 1, -1, -1):
        if result[i].get("role") == "assistant":
            last_assistant_idx = i
            break

    # Shrinkable region: tool results between the first user message and the
    # last assistant block.
    first_shrinkable = None
    last_shrinkable = None
    for i, msg in enumerate(result):
        if msg.get("role") in ("tool",) and (
            last_assistant_idx is None or i < last_assistant_idx
        ):
            if first_shrinkable is None:
                first_shrinkable = i
            last_shrinkable = i

    if first_shrinkable is None:
        return result

    # Progressively compress old tool results until we fit
    for i in range(first_shrinkable, (last_shrinkable or first_shrinkable) + 1):
        if estimate_messages_tokens(result, tools) <= max_tokens:
            break

        msg = result[i]
        if msg.get("role") != "tool":
            continue

        content = msg.get("content", "")
        if len(content) <= 200:
            continue

        try:
            data = json.loads(content)
            summary_parts = []
            if isinstance(data, dict):
                if "_summary" in data:
                    summary_parts.append(
                        f"summary={json.dumps(data['_summary'], default=str)}"
                    )
                elif "result" in data and isinstance(data["result"], dict):
                    inner = data["result"]
                    if "_summary" in inner:
                        summary_parts.append(
                            f"summary={json.dumps(inner['_summary'], default=str)}"
                        )
                    count = inner.get("count", inner.get("total", "?"))
                    summary_parts.append(f"count={count}")
                    path = inner.get("path", "")
                    if path:
                        summary_parts.append(f"path={path}")
                else:
                    count = data.get("count", data.get("total", ""))
                    if count:
                        summary_parts.append(f"count={count}")
                    error = data.get("error", "")
                    if error:
                        summary_parts.append(f"error={error}")

            compressed = "[Previous tool result compressed] " + "; ".join(summary_parts)
        except (json.JSONDecodeError, TypeError):
            compressed = "[Previous tool result compressed]"

        result[i] = {
            "role": "tool",
            "tool_call_id": msg.get("tool_call_id", ""),
            "content": compressed,
        }

    # If still over budget, drop old assistant+tool exchanges entirely
    while estimate_messages_tokens(result, tools) > max_tokens:
        dropped = False
        for i, msg in enumerate(result):
            if msg.get("role") == "assistant" and i != last_assistant_idx:
                tc_ids = set()
                for tc in msg.get("tool_calls", []):
                    tc_ids.add(tc.get("id", ""))
                indices_to_drop = {i}
                for j in range(i + 1, len(result)):
                    if (
                        result[j].get("role") == "tool"
                        and result[j].get("tool_call_id", "") in tc_ids
                    ):
                        indices_to_drop.add(j)
                    elif result[j].get("role") == "assistant":
                        break
                result = [
                    m for idx, m in enumerate(result) if idx not in indices_to_drop
                ]
                last_assistant_idx = None
                for k in range(len(result) - 1, -1, -1):
                    if result[k].get("role") == "assistant":
                        last_assistant_idx = k
                        break
                dropped = True
                break
        if not dropped:
            break

    return result
