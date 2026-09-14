"""
Core agent loop for the BV-BRC Analysis Agent.

Uses the shared agent loop with message trimming enabled.
When workflow_context is provided, it is injected as a structured context
message before the user query so the LLM knows what outputs to look for.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ensure repo root and shared/config dirs are on sys.path
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
_SHARED_DIR = str(Path(__file__).resolve().parent.parent.parent / "shared")
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent.parent / "config")
for _p in (_REPO_ROOT, _SHARED_DIR, _CONFIG_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared.agent_loop import run_agent_loop
from shared.agent_utils import (
    build_user_content,
    format_attached_documents,
    format_recent_messages,
    format_session_workspace,
)
from shared.models import ToolCall

from analysis_agent.llm_client import (
    chat_completion,
    chat_completion_stream,
    create_client,
)
from analysis_agent.models import AgentConfig, AgentResult, AgentState
from analysis_agent.prompts import SYSTEM_PROMPT


ProgressCallback = Any


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


def _analysis_progress_message(tc: ToolCall) -> str:
    """Build a human-readable progress message for analysis tool calls."""
    args = tc.arguments
    if tc.name == "workspace_browse":
        path = args.get("path", "")
        return f"Browsing output directory '{path}'..."
    elif tc.name == "get_file_metadata":
        return "Retrieving file metadata..."
    elif tc.name == "read_file_preview":
        return "Reading output file..."
    elif tc.name == "get_expected_outputs":
        svc = args.get("service_name", "")
        return f"Looking up expected outputs for {svc}..."
    elif tc.name == "get_job_details":
        ids = args.get("task_ids", [])
        return f"Retrieving job details for task {', '.join(str(i) for i in ids)}..."
    return f"Calling {tc.name}..."


def _analysis_result_message(tc: ToolCall, result: Any) -> str:
    """Build a progress message after an analysis tool completes."""
    msg = f"Tool {tc.name} completed."
    if isinstance(result, dict):
        items = result.get("items")
        if isinstance(items, list):
            msg = f"Found {len(items)} output files."
        elif result.get("error"):
            msg = "Query error, adjusting approach..."
        elif tc.name == "get_expected_outputs":
            known = result.get("known", False)
            svc = result.get("service_name", "")
            if known:
                n = len(result.get("output_patterns", {}))
                msg = f"Found {n} expected output patterns for {svc}."
            else:
                msg = f"No known output patterns for {svc}, will browse dynamically."
    return msg


async def run_agent(
    query: str,
    config: AgentConfig | None = None,
    context: dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AgentResult:
    """
    Full agent loop: plan, execute, evaluate, repeat.

    The agent collects structured analysis data (output files, metrics,
    previews, report links, step summaries) from tool results.

    Args:
        query: Natural language analysis question or workflow completion prompt.
        config: Agent configuration.
        context: Optional additional context, may include workflow_context.
        progress_callback: Optional async callback for progress updates.

    Returns:
        AgentResult with answer, structured data, and execution trace.
    """
    cfg = config or AgentConfig()
    state = AgentState(query=query, context=context or {})
    client = create_client(cfg)

    # Pre-populate messages: system prompt + optional workflow context + query.
    # We do this manually because the analysis agent has a special
    # workflow_context user message that goes before the query.
    system_content = SYSTEM_PROMPT
    if context:
        page_context = context.get("page_context", "")
        if page_context:
            system_content += (
                f"\n\n=== PAGE CONTEXT ===\n"
                f"The user is currently viewing the following page:\n"
                f"{page_context}"
            )
        # Inject bounded conversation context from recent_messages
        recent_msgs = context.get("recent_messages")
        if recent_msgs:
            formatted = format_recent_messages(recent_msgs)
            if formatted:
                system_content += (
                    f"\n\n=== CONVERSATION CONTEXT ===\n{formatted}"
                )

        # Inject attached document excerpts (PDFs + text uploads)
        docs_section = format_attached_documents(
            context.get("parsed_documents")
        )
        if docs_section:
            system_content += f"\n\n{docs_section}"

        # Inject session workspace path for chat file discovery
        session_ws = format_session_workspace(context)
        if session_ws:
            system_content += f"\n\n{session_ws}"

        ctx_for_prompt = {
            k: v for k, v in context.items()
            if k not in (
                "workflow_context", "page_context", "images",
                "conversation_summary", "recent_messages",
                "parsed_documents", "attached_files",
            )
        }
        if ctx_for_prompt:
            system_content += (
                f"\n\n=== ADDITIONAL CONTEXT ===\n"
                f"{json.dumps(ctx_for_prompt, default=str)}"
            )

    state.add_system_message(system_content)

    # Inject workflow context as a separate user message before the query
    workflow_context = context.get("workflow_context") if context else None
    if workflow_context:
        wf_message = _build_workflow_context_message(workflow_context)
        state.add_user_message(wf_message)

    images = context.get("images", []) if context else []
    state.add_user_message(build_user_content(query, images))

    # The shared loop will detect that messages are already populated
    # and skip its own system prompt / user message setup.
    await run_agent_loop(
        state=state,
        config=cfg,
        client=client,
        system_prompt=SYSTEM_PROMPT,  # used only if messages were empty
        chat_completion_fn=chat_completion,
        chat_completion_stream_fn=chat_completion_stream,
        progress_callback=progress_callback,
        progress_message_fn=_analysis_progress_message,
        result_message_fn=_analysis_result_message,
        trim_messages=True,
        max_tool_result_chars=cfg.max_tool_result_chars,
        excluded_context_keys={"workflow_context", "page_context", "images", "conversation_summary", "recent_messages", "parsed_documents", "attached_files"},
        start_message="Analyzing job outputs...",
        done_message="Analysis complete.",
    )

    return state.to_result()
