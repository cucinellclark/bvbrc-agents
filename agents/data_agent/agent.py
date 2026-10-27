"""
Core agent loop for the BV-BRC Data Retrieval Agent.

run_agent(): the plan-execute-evaluate loop, using the shared agent loop.
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

from data_agent.llm_client import chat_completion, chat_completion_stream, create_client
from data_agent.models import AgentConfig, AgentResult, AgentState
from data_agent.prompts.system import SYSTEM_PROMPT


ProgressCallback = Any


async def run_agent(
    query: str,
    config: AgentConfig | None = None,
    context: dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AgentResult:
    """
    Full agent loop: plan, execute, evaluate, repeat.

    Args:
        query: Natural language data retrieval question.
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
        start_message="Analyzing your question...",
        done_message="Data retrieval complete.",
    )

    return state.to_result()
