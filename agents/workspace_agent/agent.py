"""
Core agent loop for the BV-BRC Workspace Exploration Agent.

Follows the Data agent pattern: iterative plan-execute-evaluate loop.
On each iteration the LLM either:
  - Returns tool_calls -> execute them, feed results back, loop
  - Returns text (no tool_calls) -> done, that's the final answer

The agent collects structured data (file listings, metadata, previews)
from tool results alongside the natural language synthesis.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# Shared utilities -- deduplicated across all agents
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))
from agent_utils import (
    call_fingerprint,
    parse_tool_calls as _parse_tool_calls_raw,
    get_response_content,
    build_tool_calls_message,
    emit_progress,
)
from agent_messages import (
    DUPLICATE_CALL_WARNING,
    MAX_ITERATIONS_SYNTHESIS,
    MAX_ITERATIONS_FALLBACK,
)

from workspace_agent.llm_client import chat_completion, chat_completion_stream, create_client
from workspace_agent.models import AgentConfig, AgentResult, AgentState, ToolCall
from workspace_agent.prompts.system import SYSTEM_PROMPT
from workspace_agent.tool_registry import TOOL_SCHEMAS


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token for English/JSON."""
    return len(text) // 4


def _estimate_messages_tokens(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> int:
    """Estimate the total token count of a messages list + tool schemas."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if content:
            total += _estimate_tokens(content)
        # Tool calls in assistant messages
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            total += _estimate_tokens(json.dumps(tool_calls, default=str))
    if tools:
        total += _estimate_tokens(json.dumps(tools, default=str))
    return total


def _trim_messages_to_fit(
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
    current = _estimate_messages_tokens(messages, tools)
    if current <= max_tokens:
        return messages

    # Identify which messages are "pinned" (must keep)
    # Pinned: system messages, the first user message, and the last
    # assistant+tool exchange block.
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
        if msg.get("role") in ("tool",) and (last_assistant_idx is None or i < last_assistant_idx):
            if first_shrinkable is None:
                first_shrinkable = i
            last_shrinkable = i

    if first_shrinkable is None:
        # Nothing to trim (no old tool results)
        return result

    # Progressively compress old tool results until we fit
    for i in range(first_shrinkable, (last_shrinkable or first_shrinkable) + 1):
        if _estimate_messages_tokens(result, tools) <= max_tokens:
            break

        msg = result[i]
        if msg.get("role") != "tool":
            continue

        content = msg.get("content", "")
        if len(content) <= 200:
            continue  # Already small, skip

        # Replace with a compact summary
        try:
            data = json.loads(content)
            summary_parts = []
            if isinstance(data, dict):
                # Preserve key metadata
                if "_summary" in data:
                    summary_parts.append(f"summary={json.dumps(data['_summary'], default=str)}")
                elif "result" in data and isinstance(data["result"], dict):
                    inner = data["result"]
                    if "_summary" in inner:
                        summary_parts.append(f"summary={json.dumps(inner['_summary'], default=str)}")
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

    # If still over budget after compressing tool results, drop old
    # assistant+tool exchanges entirely (keep system + user + latest exchange)
    while _estimate_messages_tokens(result, tools) > max_tokens:
        # Find the first droppable assistant message (not the last one)
        dropped = False
        for i, msg in enumerate(result):
            if msg.get("role") == "assistant" and i != last_assistant_idx:
                # Collect the tool_call_ids from this assistant message
                tc_ids = set()
                for tc in msg.get("tool_calls", []):
                    tc_ids.add(tc.get("id", ""))
                # Drop this assistant message and its tool results
                indices_to_drop = {i}
                for j in range(i + 1, len(result)):
                    if result[j].get("role") == "tool" and result[j].get("tool_call_id", "") in tc_ids:
                        indices_to_drop.add(j)
                    elif result[j].get("role") == "assistant":
                        break
                result = [m for idx, m in enumerate(result) if idx not in indices_to_drop]
                # Recalculate last_assistant_idx
                last_assistant_idx = None
                for k in range(len(result) - 1, -1, -1):
                    if result[k].get("role") == "assistant":
                        last_assistant_idx = k
                        break
                dropped = True
                break
        if not dropped:
            break  # Nothing left to drop

    return result


def _parse_tool_calls(response: Any) -> list[ToolCall]:
    """Extract ToolCall objects from an OpenAI ChatCompletion response."""
    return _parse_tool_calls_raw(response, ToolCall)


ProgressCallback = Any  # async (progress: float, total: float|None, message: str) -> None


async def run_agent(
    query: str,
    config: AgentConfig | None = None,
    context: dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AgentResult:
    """
    Full agent loop: plan, execute, evaluate, repeat.

    On each iteration the LLM either:
      - Returns tool_calls -> execute them, feed results back, loop
      - Returns text (no tool_calls) -> done, that's the final answer

    The agent collects structured workspace data (file listings, metadata,
    file previews) from tool results. These are included in the AgentResult
    alongside the LLM's natural language summary.

    Args:
        query: Natural language workspace exploration question.
        config: Agent configuration.
        context: Optional additional context.
        progress_callback: Optional async callback for progress updates.
            Signature: async (progress, total, message) -> None.

    Returns:
        AgentResult with answer, structured data, and execution trace.
    """
    from workspace_agent.tools import execute_tool, truncate_result

    cfg = config or AgentConfig()
    state = AgentState(query=query, context=context or {})
    client = create_client(cfg)

    # Build initial messages
    system_content = SYSTEM_PROMPT
    if context:
        system_content += f"\n\n=== ADDITIONAL CONTEXT ===\n{json.dumps(context)}"

    state.add_system_message(system_content)
    state.add_user_message(query)

    await emit_progress(progress_callback, 0, None, "Analyzing your workspace question...")

    # Build auth headers if token is available
    headers: dict[str, str] | None = None
    if cfg.bvbrc_auth_token:
        headers = {"Authorization": cfg.bvbrc_auth_token}

    # Track executed call fingerprints for duplicate detection
    executed_fingerprints: set[str] = set()
    duplicate_count = 0

    for iteration in range(cfg.max_iterations):
        state.iteration = iteration + 1

        await emit_progress(
            progress_callback, iteration, cfg.max_iterations,
            "Planning next step...",
        )

        # 1. PLAN -- Ask the LLM what to do next
        # Trim message history to stay within context window
        trimmed = _trim_messages_to_fit(
            state.messages, TOOL_SCHEMAS, cfg.max_context_tokens
        )
        response = await chat_completion(
            client=client,
            messages=trimmed,
            tools=TOOL_SCHEMAS,
            config=cfg,
        )

        tool_calls = _parse_tool_calls(response)
        content = get_response_content(response)

        # 2. CHECK -- If no tool calls, the LLM produced a final answer
        if not tool_calls:
            await emit_progress(progress_callback, iteration + 1, cfg.max_iterations, "Composing answer...")
            state.final_answer = content or ""
            state.status = "completed"
            break

        # Add assistant message with tool_calls to conversation
        state.add_assistant_message(
            content=content,
            tool_calls=build_tool_calls_message(tool_calls),
        )

        # 3. EXECUTE -- Run each tool call and feed results back
        for tc in tool_calls:
            # Emit contextual progress per tool
            _tool_msg = f"Calling {tc.name}..."
            _tc_args = dict(tc.arguments)
            if tc.name == "workspace_browse":
                _path = _tc_args.get("path", "home")
                _tool_msg = f"Browsing workspace '{_path}'..."
            elif tc.name == "get_file_metadata":
                _tool_msg = f"Retrieving file metadata..."
            elif tc.name == "read_file_preview":
                _tool_msg = f"Reading file preview..."
            await emit_progress(progress_callback, iteration, cfg.max_iterations, _tool_msg)

            fp = call_fingerprint(tc)

            # --- Duplicate detection ---
            if fp in executed_fingerprints:
                duplicate_count += 1
                state.add_tool_result(
                    tc.id,
                    json.dumps({"_duplicate": True, "_message": DUPLICATE_CALL_WARNING}),
                )
                if duplicate_count >= 2:
                    break
                continue

            start = time.time()
            result = await execute_tool(
                tool_name=tc.name,
                arguments=dict(tc.arguments),
                timeout_seconds=cfg.tool_timeout_seconds,
                config=cfg,
                headers=headers,
            )
            duration_ms = (time.time() - start) * 1000

            # Record the execution (also extracts structured data)
            executed_fingerprints.add(fp)
            error = result.get("error") if isinstance(result, dict) else None
            state.record_execution(
                tc=tc,
                result=result,
                error=error,
                duration_ms=duration_ms,
            )

            # Emit result summary
            _result_msg = f"Tool {tc.name} completed."
            if isinstance(result, dict):
                _items = result.get("items")
                if isinstance(_items, list):
                    _result_msg = f"Found {len(_items)} items."
                elif result.get("error"):
                    _result_msg = f"Workspace query error, adjusting approach..."
            await emit_progress(progress_callback, iteration, cfg.max_iterations, _result_msg)

            # Serialize and truncate for the LLM context
            result_str = truncate_result(result, max_chars=cfg.max_tool_result_chars)
            state.add_tool_result(tc.id, result_str)

    else:
        # Hit max iterations without a final text response.
        # Force synthesis with tool_choice="none".  Use streaming to
        # reduce time-to-first-token for remote endpoints (Argo).
        state.status = "max_iterations"
        await emit_progress(
            progress_callback, cfg.max_iterations, cfg.max_iterations,
            f"Synthesizing answer from {len(state.tool_calls_executed)} queries...",
        )
        try:
            state.add_system_message(MAX_ITERATIONS_SYNTHESIS)
            trimmed = _trim_messages_to_fit(
                state.messages, TOOL_SCHEMAS, cfg.max_context_tokens
            )
            synthesis_content = ""
            async for chunk in chat_completion_stream(
                client=client,
                messages=trimmed,
                config=cfg,
                tool_choice="none",
            ):
                synthesis_content += chunk
            state.final_answer = synthesis_content or (
                MAX_ITERATIONS_FALLBACK.format(n=len(state.tool_calls_executed))
            )
        except Exception:
            state.final_answer = (
                MAX_ITERATIONS_FALLBACK.format(n=len(state.tool_calls_executed))
            )

    await emit_progress(progress_callback, cfg.max_iterations, cfg.max_iterations, "Workspace exploration complete.")
    return state.to_result()
