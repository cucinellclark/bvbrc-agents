"""
Core agent loop for the BV-BRC Analysis Agent.

Follows the same iterative plan-execute-evaluate loop as workspace_agent/agent.py
and data_agent/agent.py. On each iteration the LLM either:
  - Returns tool_calls -> execute them, feed results back, loop
  - Returns text (no tool_calls) -> done, that's the final answer

The agent collects structured analysis data (output files, metrics, previews,
report links, step summaries) from tool results alongside the natural language
synthesis.

When workflow_context is provided, it is injected as a structured context
message so the LLM knows what outputs to look for.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# Shared utilities -- deduplicated across all agents
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))
# Also add repo root so `shared` package imports work (shared.tools, shared.prompts)
if str(Path(__file__).resolve().parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from agent_utils import (  # noqa: E402
    call_fingerprint,
    parse_tool_calls as _parse_tool_calls_raw,
    get_response_content,
    build_tool_calls_message,
    emit_progress,
)
from agent_messages import (  # noqa: E402
    DUPLICATE_CALL_WARNING,
    MAX_ITERATIONS_SYNTHESIS,
    MAX_ITERATIONS_FALLBACK,
)

from analysis_agent.llm_client import (
    chat_completion,
    chat_completion_stream,
    create_client,
)  # noqa: E402
from analysis_agent.models import AgentConfig, AgentResult, AgentState, ToolCall  # noqa: E402
from analysis_agent.prompts import SYSTEM_PROMPT  # noqa: E402
from analysis_agent.tool_registry import TOOL_SCHEMAS  # noqa: E402


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
      2. Always keep the most recent assistant+tool-result exchange.
      3. When over budget, progressively replace older tool-result messages
         with a compact summary, starting from the oldest.
      4. If an assistant message references tool_calls whose results are
         dropped, drop that assistant message too.
    """
    current = _estimate_messages_tokens(messages, tools)
    if current <= max_tokens:
        return messages

    result = list(messages)

    # Find the last assistant message (pinned)
    last_assistant_idx = None
    for i in range(len(result) - 1, -1, -1):
        if result[i].get("role") == "assistant":
            last_assistant_idx = i
            break

    # Find shrinkable tool results
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

    # Progressively compress old tool results
    for i in range(first_shrinkable, (last_shrinkable or first_shrinkable) + 1):
        if _estimate_messages_tokens(result, tools) <= max_tokens:
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
    while _estimate_messages_tokens(result, tools) > max_tokens:
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


def _parse_tool_calls(response: Any) -> list[ToolCall]:
    """Extract ToolCall objects from an OpenAI ChatCompletion response."""
    return _parse_tool_calls_raw(response, ToolCall)


ProgressCallback = (
    Any  # async (progress: float, total: float|None, message: str) -> None
)


def _build_workflow_context_message(workflow_context: dict[str, Any]) -> str:
    """Build a structured context message from workflow_context.

    This gives the LLM the information it needs to know which outputs to
    look for without having to discover them by browsing.
    """
    lines = ["=== WORKFLOW CONTEXT ===", ""]

    wf_name = workflow_context.get("workflow_name", "unknown")
    wf_status = workflow_context.get("status", "unknown")
    wf_id = workflow_context.get("workflow_id", "")
    lines.append(f"Workflow: {wf_name} (ID: {wf_id})")
    lines.append(f"Status: {wf_status}")
    lines.append("")

    steps = workflow_context.get("steps", [])
    if steps:
        lines.append(f"Steps ({len(steps)}):")
        for i, step in enumerate(steps, 1):
            step_name = step.get("step_name", f"step_{i}")
            app_name = step.get("app_name", "unknown")
            step_status = step.get("status", "unknown")
            output_path = step.get("output_path", "")
            output_file = step.get("output_file", "")
            task_id = step.get("task_id", "")

            lines.append(f"  {i}. {step_name} ({app_name})")
            lines.append(f"     Status: {step_status}")
            if output_path:
                lines.append(f"     Output path: {output_path}")
            if output_file:
                lines.append(f"     Output file: {output_file}")
            if task_id:
                lines.append(f"     Task ID: {task_id}")
            lines.append("")

    output_paths = workflow_context.get("output_paths", [])
    if output_paths:
        lines.append("Output paths:")
        for p in output_paths:
            lines.append(f"  - {p}")

    lines.append("")
    lines.append(
        "Instructions: For each succeeded step, call get_expected_outputs "
        "with the app_name, resolve the output file paths using output_path "
        "and output_file, then browse and read the key output files to "
        "extract metrics."
    )

    return "\n".join(lines)


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

    The agent collects structured analysis data (output files, metrics,
    previews, report links, step summaries) from tool results. These are
    included in the AgentResult alongside the LLM's natural language summary.

    Args:
        query: Natural language analysis question or workflow completion prompt.
        config: Agent configuration.
        context: Optional additional context, may include workflow_context.
        progress_callback: Optional async callback for progress updates.
            Signature: async (progress, total, message) -> None.

    Returns:
        AgentResult with answer, structured data, and execution trace.
    """
    from analysis_agent.tools import execute_tool, truncate_result

    cfg = config or AgentConfig()
    state = AgentState(query=query, context=context or {})
    client = create_client(cfg)

    # Build initial messages
    system_content = SYSTEM_PROMPT

    # Inject workflow_context if provided
    workflow_context = None
    if context:
        workflow_context = context.get("workflow_context")
        # Add any non-workflow context
        ctx_for_prompt = {k: v for k, v in context.items() if k != "workflow_context"}
        if ctx_for_prompt:
            system_content += (
                f"\n\n=== ADDITIONAL CONTEXT ===\n{json.dumps(ctx_for_prompt)}"
            )

    state.add_system_message(system_content)

    # If workflow_context is present, inject it as a separate context message
    # before the user query so the LLM has structured metadata
    if workflow_context:
        wf_message = _build_workflow_context_message(workflow_context)
        state.add_user_message(wf_message)

    state.add_user_message(query)

    await emit_progress(progress_callback, 0, None, "Analyzing job outputs...")

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
            progress_callback,
            iteration,
            cfg.max_iterations,
            "Planning next analysis step...",
        )

        # 1. PLAN -- Ask the LLM what to do next
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
            await emit_progress(
                progress_callback,
                iteration + 1,
                cfg.max_iterations,
                "Composing analysis summary...",
            )
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
                _path = _tc_args.get("path", "")
                _tool_msg = f"Browsing output directory '{_path}'..."
            elif tc.name == "get_file_metadata":
                _tool_msg = "Retrieving file metadata..."
            elif tc.name == "read_file_preview":
                _tool_msg = "Reading output file..."
            elif tc.name == "get_expected_outputs":
                _svc = _tc_args.get("service_name", "")
                _tool_msg = f"Looking up expected outputs for {_svc}..."
            elif tc.name == "get_job_details":
                _ids = _tc_args.get("task_ids", [])
                _tool_msg = f"Retrieving job details for task {', '.join(str(i) for i in _ids)}..."
            await emit_progress(
                progress_callback, iteration, cfg.max_iterations, _tool_msg
            )

            fp = call_fingerprint(tc)

            # --- Duplicate detection ---
            if fp in executed_fingerprints:
                duplicate_count += 1
                state.add_tool_result(
                    tc.id,
                    json.dumps(
                        {"_duplicate": True, "_message": DUPLICATE_CALL_WARNING}
                    ),
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
                    _result_msg = f"Found {len(_items)} output files."
                elif result.get("error"):
                    _result_msg = "Query error, adjusting approach..."
                elif tc.name == "get_expected_outputs":
                    _known = result.get("known", False)
                    _svc = result.get("service_name", "")
                    if _known:
                        _n = len(result.get("output_patterns", {}))
                        _result_msg = f"Found {_n} expected output patterns for {_svc}."
                    else:
                        _result_msg = f"No known output patterns for {_svc}, will browse dynamically."
            await emit_progress(
                progress_callback, iteration, cfg.max_iterations, _result_msg
            )

            # Serialize and truncate for the LLM context
            result_str = truncate_result(result, max_chars=cfg.max_tool_result_chars)
            state.add_tool_result(tc.id, result_str)

    else:
        # Hit max iterations without a final text response.
        # Force synthesis with tool_choice="none".
        state.status = "max_iterations"
        await emit_progress(
            progress_callback,
            cfg.max_iterations,
            cfg.max_iterations,
            f"Synthesizing analysis from {len(state.tool_calls_executed)} queries...",
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
            state.final_answer = MAX_ITERATIONS_FALLBACK.format(
                n=len(state.tool_calls_executed)
            )

    await emit_progress(
        progress_callback, cfg.max_iterations, cfg.max_iterations, "Analysis complete."
    )
    return state.to_result()
