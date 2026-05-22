"""
Agent-compatible MCP tool functions for BV-BRC data access.

This module contains tool functions that will be migrated into the MCP server's
tools/data_tools.py. They are developed and tested here first.

To migrate: paste the tool functions inside register_data_tools() in
bvbrc-mcp-server/tools/data_tools.py, add @mcp.tool() decorators, and replace
_base_url / _build_auth_headers references with the MCP server's versions.

Local development imports from the MCP server's functions/ via sys.path
(through data_agent.tools._mcp_imports). When migrated to the MCP server,
these become standard relative imports.
"""

from __future__ import annotations

import re
import sys
from types import ModuleType
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Default BV-BRC API endpoint (matches AgentConfig.bvbrc_api_url)
_DEFAULT_BASE_URL = "https://www.bv-brc.org/api-bulk"

# Lazy import cache -- avoids circular import through data_agent.tools
_data_functions: Optional[ModuleType] = None
_path_added: bool = False


def _get_data_functions() -> ModuleType:
    """
    Import and return the MCP server's data_functions module.

    Uses the same sys.path approach as data_agent.tools._mcp_imports but
    without importing through data_agent.tools (which would trigger a
    circular import since search.py imports from this module).
    """
    global _data_functions, _path_added
    if _data_functions is not None:
        return _data_functions

    if not _path_added:
        from data_agent.models import AgentConfig
        mcp_path = AgentConfig().mcp_server_path
        if mcp_path and mcp_path not in sys.path:
            sys.path.insert(0, mcp_path)
        _path_added = True

    from functions import data_functions
    _data_functions = data_functions
    return _data_functions


def _build_auth_headers(token: Optional[str]) -> Optional[Dict[str, str]]:
    """
    Build auth headers from a token string.

    In the MCP server, replace this with the closure version that uses
    token_provider.
    """
    if not token:
        return None
    return {"Authorization": token}


# Solr boolean operators and keywords that should NOT be quoted
_SOLR_KEYWORDS = {"AND", "OR", "NOT", "TO"}

# Regex: match  field_name:UnquotedWord1 Word2 ...WordN
# where the words are NOT already quoted, not inside brackets/parens,
# and are followed by AND/OR/NOT, another field:value, or end of string.
#
# Breakdown:
#   ([a-z_]\w*)           capture group 1: field name (lowercase + underscores)
#   :                     colon separator
#   (?!")                 NOT already followed by a quote (skip already-quoted)
#   (?!\[)                NOT followed by [ (skip range queries)
#   (?!\{)                NOT followed by { (skip exclusive ranges)
#   (?!\()                NOT followed by ( (skip grouped values)
#   (?!\*)                NOT followed by * (skip wildcard)
#   ([A-Z][a-z]+          capture group 2: first word (capitalized)
#   (?:\s+[A-Z][a-z]+)*   followed by one or more additional capitalized words
#   (?:\s+(?:str|subsp|var|sv|serovar|pv)\.\s*\S+)*  optional taxonomic qualifiers
#   )
#   (?=\s+(?:AND|OR|NOT)\s|\s*$)  lookahead: followed by operator or end of string
_UNQUOTED_MULTIWORD_RE = re.compile(
    r'([a-z_]\w*)'                          # field name
    r':'                                    # colon
    r'(?!["\[\{(*])'                        # not already quoted/range/group/wildcard
    r'('                                    # start capture: value
    r'[A-Z][a-z]+'                          # first capitalized word
    r'(?:\s+(?!AND\b|OR\b|NOT\b|TO\b)[A-Za-z][A-Za-z.]+)+'  # 1+ additional words (not operators)
    r')'                                    # end capture
    r'(?=\s+(?:AND|OR|NOT)\b|\s*$)'         # lookahead: operator or end
)


def _auto_quote_query(query: str) -> str:
    """
    Auto-quote unquoted multi-word values in a Solr query string.

    LLMs sometimes produce queries like:
        organism:Escherichia coli AND property:"Virulence Factor"

    This function detects the pattern `field:Word1 Word2` (capitalized words
    that are not Solr operators) and wraps them in quotes:
        organism:"Escherichia coli" AND property:"Virulence Factor"

    Only applies to values that:
      - Start with a capitalized word
      - Contain at least two words
      - Are NOT already quoted, in brackets (ranges), or in parentheses (groups)
      - Are followed by a Solr operator (AND/OR/NOT) or end of string

    This is intentionally conservative -- it only fixes the most common
    pattern (taxonomic names, organism names, product descriptions) and
    avoids modifying range queries, wildcards, or already-correct syntax.
    """
    if not query or query == "*:*":
        return query

    def _replacer(m: re.Match) -> str:
        field = m.group(1)
        value = m.group(2)
        return f'{field}:"{value}"'

    return _UNQUOTED_MULTIWORD_RE.sub(_replacer, query)


# ---------------------------------------------------------------------------
# Field validation guardrail
# ---------------------------------------------------------------------------

# Regex to extract field names from a Solr query string.
# Matches word:  patterns where the word is a plausible field name
# (lowercase letters, digits, underscores).  Skips the *:* match-all.
_QUERY_FIELD_RE = re.compile(r'\b([a-z_][a-z0-9_]*)\s*:')


def _extract_query_fields(query: str) -> List[str]:
    """Extract field names referenced in a Solr query string.

    Returns a deduplicated list of field names found in patterns like
    ``field:value``, ``field:"quoted"``, ``field:[range TO range]``, etc.
    Ignores the ``*:*`` match-all pattern.
    """
    if not query or query.strip() == "*:*":
        return []
    return list(dict.fromkeys(_QUERY_FIELD_RE.findall(query)))


def _validate_query_fields(
    collection: str,
    query: str,
    select: Optional[List[str]] = None,
    sort_expr: Optional[str] = None,
    facet_fields: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Pre-flight validation of field names against the collection schema.

    Checks that all field names referenced in the query string, select
    list, sort expression, and facet fields actually exist on the target
    collection.

    Returns ``None`` if everything is valid.  Returns an informative error
    dict if invalid fields are found (this dict can be returned directly
    to the LLM instead of sending a doomed request to Solr).
    """
    from data_agent.prompts.field_registry import (
        validate_fields,
        suggest_collections,
        COLLECTION_FIELDS,
    )

    # If the collection isn't in our registry, skip validation
    if collection not in COLLECTION_FIELDS:
        return None

    # Gather all field names from every source
    all_fields: List[str] = _extract_query_fields(query)

    if select:
        all_fields.extend(select)

    if sort_expr:
        # Sort expression is like "field_name asc" or "field_name desc"
        sort_field = sort_expr.strip().split()[0] if sort_expr.strip() else None
        if sort_field:
            all_fields.append(sort_field)

    if facet_fields:
        all_fields.extend(facet_fields)

    # Deduplicate
    unique_fields = list(dict.fromkeys(all_fields))
    if not unique_fields:
        return None

    invalid = validate_fields(collection, unique_fields)
    if not invalid:
        return None

    # Build an informative error for the LLM
    suggestions: Dict[str, List[str]] = {}
    for f in invalid:
        collections_with = suggest_collections(f)
        if collections_with:
            suggestions[f] = collections_with

    # Build the error message
    field_list = ", ".join(invalid)
    msg_parts = [
        f"Invalid field(s) for collection '{collection}': {field_list}.",
    ]
    for f in invalid:
        if f in suggestions:
            cols = ", ".join(suggestions[f])
            msg_parts.append(
                f"Field '{f}' does not exist on '{collection}' but IS available "
                f"on: {cols}."
            )
        else:
            msg_parts.append(
                f"Field '{f}' does not exist on '{collection}' or any known collection."
            )
    msg_parts.append(
        "Check the COLLECTION FIELD REFERENCE to find the correct field names, "
        "or use a different collection that has the field you need."
    )

    valid_fields = sorted(COLLECTION_FIELDS.get(collection, set()))

    return {
        "error": " ".join(msg_parts),
        "invalid_fields": invalid,
        "valid_fields_on_collection": valid_fields,
        "suggestions": suggestions,
        "collection": collection,
        "query": query,
        "source": "bvbrc-mcp-data",
    }


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

async def solr_query(
    collection: str,
    query: str = "*:*",
    select: Optional[List[str]] = None,
    sort: Optional[str] = None,
    limit: int = 25,
    count_only: bool = False,
    cursor_id: Optional[str] = None,
    token: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute a direct Solr query against a BV-BRC collection.

    Accepts raw Solr query syntax and executes it directly without
    LLM-based query planning. Intended for use by agent systems that
    produce their own structured queries.

    Args:
        collection: Solr collection name (e.g., "genome", "genome_amr", "sp_gene").
        query: Solr query string using Lucene syntax.
            Examples:
              - "genus:Salmonella AND host_name:Human"
              - "resistant_phenotype:Resistant AND antibiotic:ciprofloxacin"
              - 'property:"Virulence Factor" AND organism:"Escherichia coli"'
              - "*:*" (match all)
        select: Fields to return. If None, returns default fields for the collection.
        sort: Solr sort expression (e.g., "genome_name asc"). If None, uses default.
        limit: Maximum records to return (1-10000, default 25).
        count_only: If True, return only the total count (no documents).
        cursor_id: Cursor ID for pagination. Use the nextCursorId from a previous
            response to fetch the next page.
        token: Authentication token (optional).
        base_url: Override the default BV-BRC API URL.

    Returns:
        If count_only: {"numFound": int, "source": "bvbrc-mcp-data"}
        Otherwise: {"results": [...], "count": int, "numFound": int,
                    "nextCursorId": str|null, "source": "bvbrc-mcp-data"}
    """
    data_fn = _get_data_functions()
    headers = _build_auth_headers(token)
    effective_url = base_url or _DEFAULT_BASE_URL

    # Auto-quote unquoted multi-word values (e.g., organism:Escherichia coli)
    query = _auto_quote_query(query or "*:*")

    # Normalize select and sort using MCP server's utility functions
    options: Dict[str, Any] = {}
    select_list = data_fn.normalize_select(select)
    if select_list:
        options["select"] = select_list
    sort_expr = data_fn.normalize_sort(sort)
    if sort_expr:
        options["sort"] = sort_expr

    # Validate field names before sending to Solr
    validation_error = _validate_query_fields(
        collection, query, select=select_list, sort_expr=sort_expr,
    )
    if validation_error:
        return validation_error

    try:
        result = await data_fn.query_direct(
            core=collection,
            filter_str=query or "*:*",
            options=options if options else None,
            base_url=effective_url,
            headers=headers,
            cursorId=cursor_id or "*",
            countOnly=count_only,
            batch_size=min(limit, 10000) if not count_only else None,
        )
        result["source"] = "bvbrc-mcp-data"
        return result

    except Exception as e:
        return {
            "error": f"solr_query failed: {type(e).__name__}: {str(e)}",
            "collection": collection,
            "query": query,
            "source": "bvbrc-mcp-data",
        }


async def solr_facet_query(
    collection: str,
    query: str = "*:*",
    facet_fields: Optional[List[str]] = None,
    facet_limit: int = 20,
    facet_mincount: int = 1,
    token: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute a Solr facet query to get value distributions.

    Returns grouped counts for the specified facet fields, with no
    document bodies. Useful for "how many X per Y" or "top N by field"
    questions.

    Args:
        collection: Solr collection name.
        query: Solr query string to filter before faceting.
        facet_fields: Fields to compute distributions for.
        facet_limit: Maximum values per facet field (default 20).
        facet_mincount: Minimum count to include (default 1).
        token: Authentication token (optional).
        base_url: Override the default BV-BRC API URL.

    Returns:
        {"numFound": int, "facets": {field: [{"value": str, "count": int}, ...]},
         "source": "bvbrc-mcp-data"}
    """
    data_fn = _get_data_functions()
    headers = _build_auth_headers(token)
    effective_url = base_url or _DEFAULT_BASE_URL

    # Auto-quote unquoted multi-word values
    query = _auto_quote_query(query or "*:*")

    if not facet_fields:
        return {
            "error": "solr_facet_query requires at least one facet_field",
            "collection": collection,
            "query": query,
            "source": "bvbrc-mcp-data",
        }

    # Validate field names before sending to Solr
    validation_error = _validate_query_fields(
        collection, query, facet_fields=facet_fields,
    )
    if validation_error:
        return validation_error

    try:
        result = await data_fn.query_faceted(
            core=collection,
            filter_str=query or "*:*",
            facet_fields=facet_fields,
            base_url=effective_url,
            headers=headers,
            facet_limit=facet_limit,
            facet_mincount=facet_mincount,
        )
        result["source"] = "bvbrc-mcp-data"
        return result

    except Exception as e:
        return {
            "error": f"solr_facet_query failed: {type(e).__name__}: {str(e)}",
            "collection": collection,
            "query": query,
            "source": "bvbrc-mcp-data",
        }


# ---------------------------------------------------------------------------
# Probe (keyword reconnaissance) tool
# ---------------------------------------------------------------------------

# Primary-key field for each collection, used in select() to minimise payload.
_COLLECTION_PK: Dict[str, str] = {
    "genome": "genome_id",
    "genome_feature": "feature_id",
    "genome_amr": "id",
    "genome_sequence": "sequence_id",
    "taxonomy": "taxon_id",
    "sp_gene": "id",
    "pathway": "id",
    "subsystem": "id",
    "epitope": "epitope_id",
    "epitope_assay": "assay_id",
    "experiment": "eid",
    "protein_structure": "pdb_id",
    "surveillance": "id",
    "serology": "id",
    "strain": "id",
    "spike_lineage": "id",
    "spike_variant": "id",
    "antibiotics": "antibiotic_name",
}


def _build_rql_facet_clause(
    facet_fields: List[str],
    facet_limit: int = 20,
    facet_mincount: int = 1,
) -> str:
    """Build an RQL facet() clause matching the BV-BRC API format.

    The BV-BRC API expects the RQL facet syntax used by the website:
        facet((field,genus),(field,species),(mincount,1),(limit,20))
    """
    parts = [f"(field,{f})" for f in facet_fields]
    parts.append(f"(mincount,{facet_mincount})")
    parts.append(f"(limit,{facet_limit})")
    return "facet(" + ",".join(parts) + ")"


def _parse_solr_facet_fields(
    raw: Dict[str, List],
) -> Dict[str, List[Dict[str, Any]]]:
    """Parse Solr's alternating [value, count, value, count, ...] arrays
    into a list of {value, count} dicts per field."""
    parsed: Dict[str, List[Dict[str, Any]]] = {}
    for field, pairs in raw.items():
        entries: List[Dict[str, Any]] = []
        i = 0
        while i < len(pairs) - 1:
            entries.append({"value": pairs[i], "count": pairs[i + 1]})
            i += 2
        parsed[field] = entries
    return parsed


async def solr_probe_query(
    collection: str,
    keywords: str,
    facet_fields: Optional[List[str]] = None,
    facet_limit: int = 20,
    token: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Keyword-based reconnaissance query using BV-BRC's RQL keyword() + facet().

    Sends a single request that:
      - Full-text searches via keyword(X) across all indexed fields
      - Returns server-side faceted counts for the requested fields
      - Returns only 1 document (minimal payload; we care about facets + numFound)

    This lets the agent discover what field values actually exist in the data
    before committing to a structured Solr query.

    Args:
        collection: Solr collection name (e.g., "genome", "genome_amr").
        keywords: Search terms for full-text matching (e.g., "Deltacoronavirus").
        facet_fields: Fields to get value distributions for.
        facet_limit: Max values per facet field (default 20).
        token: Authentication token (optional).
        base_url: Override the default BV-BRC API URL.

    Returns:
        {
            "numFound": int,
            "facets": { field: [{"value": str, "count": int}, ...], ... },
            "source": "bvbrc-mcp-data"
        }
    """
    import httpx as _httpx
    from urllib.parse import quote as _url_quote

    effective_url = (base_url or _DEFAULT_BASE_URL).rstrip("/")

    # Build the RQL request body
    pk_field = _COLLECTION_PK.get(collection, "id")
    encoded_kw = _url_quote(keywords, safe="")
    body_parts = [
        f"keyword({encoded_kw})",
        "limit(1)",
        f"select({pk_field})",
    ]
    if facet_fields:
        body_parts.append(
            _build_rql_facet_clause(facet_fields, facet_limit=facet_limit)
        )
    body = "&".join(body_parts)

    url = f"{effective_url}/{collection}/"
    headers: Dict[str, str] = {
        "Content-Type": "application/rqlquery+x-www-form-urlencoded",
        "Accept": "application/solr+json",
    }
    if token:
        headers["Authorization"] = token

    print(f"[probe_data] RQL request: POST {url}")
    print(f"[probe_data] Body: {body}")

    try:
        async with _httpx.AsyncClient() as client:
            response = await client.post(
                url,
                content=body,
                headers=headers,
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()

        # Parse the Solr-format response
        solr_response = data.get("response", {})
        num_found = solr_response.get("numFound", 0)

        facets: Dict[str, List[Dict[str, Any]]] = {}
        facet_counts = data.get("facet_counts", {})
        raw_facet_fields = facet_counts.get("facet_fields", {})
        if raw_facet_fields:
            facets = _parse_solr_facet_fields(raw_facet_fields)

        print(f"[probe_data] numFound={num_found}, facet_fields={list(facets.keys())}")

        result: Dict[str, Any] = {
            "numFound": num_found,
            "facets": facets,
            "source": "bvbrc-mcp-data",
        }
        return result

    except Exception as e:
        return {
            "error": f"solr_probe_query failed: {type(e).__name__}: {str(e)}",
            "collection": collection,
            "keywords": keywords,
            "source": "bvbrc-mcp-data",
        }
