"""
Shared data-retrieval tools for BV-BRC Solr queries.

Thin async wrappers that translate agent tool-call arguments into the
``agent_data_tools`` module (which adds query validation, auto-quoting,
normalize_select/normalize_sort) before hitting ``data_functions``.

Any agent can import these tools to query BV-BRC Solr collections.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# The agent_data_tools module lives inside data_agent/mcp_tools/.
# We add that path once so the import works from any agent.
_AGENT_DATA_TOOLS_PATH = str(
    Path(__file__).resolve().parent.parent.parent / "agents" / "data_agent"
)

_data_tools_imported = False


def _ensure_data_tools_path() -> None:
    global _data_tools_imported
    if _data_tools_imported:
        return
    if _AGENT_DATA_TOOLS_PATH not in sys.path:
        sys.path.insert(0, _AGENT_DATA_TOOLS_PATH)
    _data_tools_imported = True


def _get_solr_functions():
    """Lazy import of agent_data_tools to avoid circular / early imports."""
    _ensure_data_tools_path()
    from mcp_tools.agent_data_tools import (  # type: ignore[import-untyped]
        solr_query,
        solr_facet_query,
        solr_probe_query,
    )

    return solr_query, solr_facet_query, solr_probe_query


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------


async def search_data(
    collection: str,
    query: str,
    select: Optional[List[str]] = None,
    sort: Optional[str] = None,
    limit: int = 25,
    count_only: bool = False,
    base_url: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Search a BV-BRC Solr collection.

    Args:
        collection: Solr collection name (e.g. ``"genome"``, ``"genome_amr"``).
        query: Solr query string (e.g. ``"genus:Salmonella AND host_name:Human"``).
        select: Fields to return.  ``None`` returns all fields.
        sort: Sort expression (e.g. ``"genome_name asc"``).
        limit: Maximum records to return (default 25, max 10 000).
        count_only: If ``True``, return only the count of matching records.
        base_url: Override the default BV-BRC API URL.
        headers: HTTP headers dict; auth token extracted automatically.

    Returns:
        Dict with ``numFound`` (count_only) or ``results`` + ``count`` + ``numFound``.
    """
    solr_query_fn, _, _ = _get_solr_functions()

    token = _extract_token(headers)
    return await solr_query_fn(
        collection=collection,
        query=query,
        select=select,
        sort=sort,
        limit=limit,
        count_only=count_only,
        token=token,
        base_url=base_url,
    )


async def facet_query(
    collection: str,
    query: str,
    facet_fields: List[str],
    facet_limit: int = 20,
    facet_mincount: int = 1,
    base_url: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Execute a facet query to get value distributions for fields.

    Args:
        collection: Solr collection name.
        query: Solr query string to filter before faceting.
        facet_fields: Fields to get distributions for.
        facet_limit: Max facet values per field (default 20).
        facet_mincount: Minimum count to include a value (default 1).
        base_url: Override the default BV-BRC API URL.
        headers: HTTP headers dict.

    Returns:
        Dict with ``numFound`` and ``facets``.
    """
    _, solr_facet_query_fn, _ = _get_solr_functions()

    token = _extract_token(headers)
    return await solr_facet_query_fn(
        collection=collection,
        query=query,
        facet_fields=facet_fields,
        facet_limit=facet_limit,
        facet_mincount=facet_mincount,
        token=token,
        base_url=base_url,
    )


async def probe_data(
    collection: str,
    keywords: str,
    facet_fields: Optional[List[str]] = None,
    facet_limit: int = 20,
    base_url: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Keyword-based reconnaissance search with faceted field summaries.

    Args:
        collection: Solr collection name.
        keywords: Search terms for full-text matching.
        facet_fields: Fields to get value distributions for.
        facet_limit: Max values per facet field (default 20).
        base_url: Override the default BV-BRC API URL.
        headers: HTTP headers dict.

    Returns:
        Dict with ``numFound`` and ``facets``.
    """
    _, _, solr_probe_query_fn = _get_solr_functions()

    token = _extract_token(headers)
    return await solr_probe_query_fn(
        collection=collection,
        keywords=keywords,
        facet_fields=facet_fields,
        facet_limit=facet_limit,
        token=token,
        base_url=base_url,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_token(headers: Optional[Dict[str, str]]) -> Optional[str]:
    """Pull the auth token out of a headers dict."""
    if headers and "Authorization" in headers:
        return headers["Authorization"]
    return None
