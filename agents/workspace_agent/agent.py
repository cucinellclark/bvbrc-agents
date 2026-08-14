"""
Core agent loop for the BV-BRC Workspace Exploration Agent.

Uses the shared agent loop with message trimming enabled
(workspace queries can produce large results that exceed the context window).
"""

from __future__ import annotations

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
from shared.models import ToolCall

from workspace_agent.llm_client import (
    chat_completion,
    chat_completion_stream,
    create_client,
)
from workspace_agent.models import AgentConfig, AgentResult, AgentState
from workspace_agent.prompts.system import SYSTEM_PROMPT


ProgressCallback = Any


def _workspace_progress_message(tc: ToolCall) -> str:
    """Build a human-readable progress message for workspace tool calls."""
    args = tc.arguments
    if tc.name == "workspace_browse":
        path = args.get("path", "home")
        return f"Browsing workspace '{path}'..."
    elif tc.name == "get_file_metadata":
        return "Retrieving file metadata..."
    elif tc.name == "read_file_preview":
        return "Reading file preview..."
    return f"Calling {tc.name}..."


def _workspace_result_message(tc: ToolCall, result: Any) -> str:
    """Build a progress message after a workspace tool completes."""
    msg = f"Tool {tc.name} completed."
    if isinstance(result, dict):
        items = result.get("items")
        if isinstance(items, list):
            msg = f"Found {len(items)} items."
        elif result.get("error"):
            msg = "Workspace query error, adjusting approach..."
    return msg


async def run_agent(
    query: str,
    config: AgentConfig | None = None,
    context: dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AgentResult:
    """
    Full agent loop: plan, execute, evaluate, repeat.

    The agent collects structured workspace data (file listings, metadata,
    file previews) from tool results. These are included in the AgentResult
    alongside the LLM's natural language summary.

    Args:
        query: Natural language workspace exploration question.
        config: Agent configuration.
        context: Optional additional context.
        progress_callback: Optional async callback for progress updates.

    Returns:
        AgentResult with answer, structured data, and execution trace.
    """
    cfg = config or AgentConfig()
    state = AgentState(query=query, context=context or {})
    client = create_client(cfg)

    await run_agent_loop(
        state=state,
        config=cfg,
        client=client,
        system_prompt=SYSTEM_PROMPT,
        chat_completion_fn=chat_completion,
        chat_completion_stream_fn=chat_completion_stream,
        progress_callback=progress_callback,
        progress_message_fn=_workspace_progress_message,
        result_message_fn=_workspace_result_message,
        trim_messages=True,
        max_tool_result_chars=cfg.max_tool_result_chars,
        start_message="Analyzing your workspace question...",
        done_message="Workspace exploration complete.",
    )

    return state.to_result()
