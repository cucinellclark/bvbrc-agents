"""
Tool implementations for BV-BRC Solr data queries.

Thin async wrappers that translate agent tool-call arguments into the
structured MCP tool functions in data_agent.mcp_tools.agent_data_tools.

The agent's LLM produces tool calls matching the schemas in tool_registry.py
(search_data, facet_query). These wrappers map those arguments to
solr_query / solr_facet_query, which add normalize_select/normalize_sort
and proper error handling before calling data_functions.query_direct /
query_faceted.

When the MCP migration is complete, these wrappers will be replaced with
MCP HTTP client calls.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from data_agent.mcp_tools.agent_data_tools import solr_query, solr_facet_query, solr_probe_query


async def search_data(
    collection: str,
    query: str,
    select: Optional[List[str]] = None,
    sort: Optional[str] = None,
    limit: int = 25,
    count_only: bool = False,
    base_url: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Search a BV-BRC Solr collection.

    Wraps solr_query() from agent_data_tools.

    Args:
        collection: Solr collection name (e.g., "genome", "genome_amr").
        query: Solr query string (e.g., "genus:Salmonella AND host_name:Human").
        select: Fields to return. If None, returns all fields.
        sort: Sort expression (e.g., "genome_name asc").
        limit: Maximum records to return (default 25, max 10000).
        count_only: If True, return only the count of matching records.
        base_url: Override the default BV-BRC API URL.
        headers: Additional HTTP headers (e.g., auth token).

    Returns:
        Dict with keys:
          - count_only=True:  {"numFound": int}
          - count_only=False: {"results": [...], "count": int, "numFound": int}
    """
    # Extract token from headers if present (agent passes headers dict,
    # but solr_query expects a token string)
    token = None
    if headers and "Authorization" in headers:
        token = headers["Authorization"]

    return await solr_query(
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
) -> Dict[str, Any]:
    """
    Execute a facet query to get value distributions for fields.

    Wraps solr_facet_query() from agent_data_tools.

    Args:
        collection: Solr collection name.
        query: Solr query string to filter before faceting.
        facet_fields: Fields to get distributions for.
        facet_limit: Max facet values per field (default 20).
        facet_mincount: Minimum count to include a value (default 1).
        base_url: Override the default BV-BRC API URL.
        headers: Additional HTTP headers.

    Returns:
        Dict with keys:
          - "numFound": total matching records
          - "facets": {field_name: [{"value": str, "count": int}, ...], ...}
    """
    # Extract token from headers if present
    token = None
    if headers and "Authorization" in headers:
        token = headers["Authorization"]

    return await solr_facet_query(
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
) -> Dict[str, Any]:
    """
    Keyword-based reconnaissance search with faceted field summaries.

    Performs a full-text keyword search against a BV-BRC collection and
    returns the total match count plus value distributions for the
    requested facet fields. Use this to discover correct field values
    before building a structured Solr query.

    Wraps solr_probe_query() from agent_data_tools.

    Args:
        collection: Solr collection name (e.g., "genome", "genome_amr").
        keywords: Search terms for full-text matching.
        facet_fields: Fields to get value distributions for.
        facet_limit: Max values per facet field (default 20).
        base_url: Override the default BV-BRC API URL.
        headers: Additional HTTP headers (e.g., auth token).

    Returns:
        Dict with keys:
          - "numFound": total matching records
          - "facets": {field: [{"value": str, "count": int}, ...], ...}
    """
    # Extract token from headers if present
    token = None
    if headers and "Authorization" in headers:
        token = headers["Authorization"]

    return await solr_probe_query(
        collection=collection,
        keywords=keywords,
        facet_fields=facet_fields,
        facet_limit=facet_limit,
        token=token,
        base_url=base_url,
    )
