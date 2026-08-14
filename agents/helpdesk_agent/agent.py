"""
Core agent loop for the BV-BRC Helpdesk Agent.

Uses the shared agent loop for the plan-execute-evaluate cycle.
The helpdesk agent customizes progress messages for its tools
(query_helpdesk, list_services, get_service_schema).
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

from helpdesk_agent.llm_client import (
    chat_completion,
    chat_completion_stream,
    create_client,
)
from helpdesk_agent.models import AgentConfig, AgentResult, AgentState
from helpdesk_agent.prompts.system import SYSTEM_PROMPT


ProgressCallback = Any


def _helpdesk_progress_message(tc: ToolCall) -> str:
    """Build a human-readable progress message for helpdesk tool calls."""
    if tc.name == "query_helpdesk":
        q = tc.arguments.get("query", "")
        return f"Searching helpdesk: {q[:60]}..."
    elif tc.name == "list_services":
        return "Listing available BV-BRC services..."
    elif tc.name == "get_service_schema":
        svc = tc.arguments.get("service_name", "service")
        return f"Looking up {svc} parameters..."
    return f"Calling {tc.name}..."


def _helpdesk_result_message(tc: ToolCall, result: Any) -> str:
    """Build a progress message after a helpdesk tool completes."""
    msg = f"Tool {tc.name} completed."
    if isinstance(result, dict):
        count = result.get("count")
        if count is not None:
            msg = f"Found {count} relevant documents."
        elif result.get("error"):
            msg = "Query returned an error, adjusting approach..."
    return msg


async def run_agent(
    query: str,
    config: AgentConfig | None = None,
    context: dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AgentResult:
    """
    Full agent loop: plan, execute, evaluate, repeat.

    Args:
        query: Natural language question about using BV-BRC.
        config: Agent configuration.
        context: Optional additional context.
        progress_callback: Optional async callback for progress updates.

    Returns:
        AgentResult with answer, execution trace, and sources.
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
        progress_message_fn=_helpdesk_progress_message,
        result_message_fn=_helpdesk_result_message,
        start_message="Analyzing your question...",
        done_message="Helpdesk response complete.",
    )

    return state.to_result()
