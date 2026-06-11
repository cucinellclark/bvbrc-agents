"""
Tool implementation for helpdesk RAG queries.

Thin async wrapper that translates agent tool-call arguments into the
MCP server's rag_database_functions.query_rag_helpdesk_func().
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from helpdesk_agent.mcp_tools.agent_helpdesk_tools import query_helpdesk_impl
from helpdesk_agent.models import AgentConfig


async def query_helpdesk(
    query: str,
    top_k: int = 5,
    config: Optional[AgentConfig] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Search the BV-BRC helpdesk knowledge base.

    Returns raw documents (no LLM summarization) so the agent's own LLM
    can reason over them.

    Args:
        query: Natural language search query about BV-BRC usage.
        top_k: Number of top results to return.
        config: Agent configuration.

    Returns:
        Dict with keys:
          - results: List of {content, score, metadata} dicts
          - count: Number of results
          - query: Original query
          - index: Database name queried
    """
    return await query_helpdesk_impl(
        query=query,
        top_k=top_k,
        config=config,
    )
