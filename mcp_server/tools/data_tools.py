#!/usr/bin/env python3
"""
BV-BRC MVP Tools

This module contains MCP tools for querying MVP (Minimum Viable Product) data from BV-BRC.
"""

import json
import subprocess
import os
import re
import tempfile
from urllib.parse import quote, unquote_plus
from typing import Optional, Dict, Any, List

from fastmcp import FastMCP, Context

from functions.data_functions import (
    query_direct,
    query_faceted,
    lookup_parameters,
    list_solr_collections,
    normalize_select,
    normalize_sort,
    build_filter,
    get_collection_fields,
    validate_filter_fields,
    create_query_plan_internal,
    select_collection_for_query,
    get_feature_sequence_by_id,
    get_genome_sequence_by_id,
    CURSOR_BATCH_SIZE
)
from common.llm_client import create_llm_client_from_config

# Global variables to store configuration
_base_url = None
_token_provider = None
_llm_client = None
_download_cancel_tokens: Dict[str, bool] = {}
_default_select_by_collection: Optional[Dict[str, List[str]]] = None
_default_facet_by_collection: Optional[Dict[str, List[str]]] = None
_default_snapshot_limit: Optional[int] = None

# Default snapshot limit used when config is not available
_FALLBACK_SNAPSHOT_LIMIT = 100

# Primary key field per Solr collection, used for deterministic sort in downloads
_COLLECTION_KEY_FIELD = {
    "genome": "genome_id",
    "genome_feature": "feature_id",
    "taxonomy": "taxon_id",
    "genome_amr": "id",
    "genome_sequence": "sequence_id",
    "sp_gene": "patric_id",
    "pathway": "pathway_id",
    "subsystem": "id",
    "protein_family_ref": "family_id",
    "specialty_gene_ref": "id",
    "antibiotics": "antibiotic_name",
    "enzyme_class_ref": "ec_number",
    "gene_ontology_ref": "go_id",
}


def _load_default_snapshot_limit() -> int:
    """Load data.default_snapshot_limit from config.json. Cached on first call."""
    global _default_snapshot_limit
    if _default_snapshot_limit is not None:
        return _default_snapshot_limit
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "config.json"
    )
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        data_config = config.get("data") or {}
        limit = data_config.get("default_snapshot_limit")
        if isinstance(limit, int) and limit > 0:
            _default_snapshot_limit = limit
        else:
            _default_snapshot_limit = _FALLBACK_SNAPSHOT_LIMIT
    except Exception:
        _default_snapshot_limit = _FALLBACK_SNAPSHOT_LIMIT
    return _default_snapshot_limit


def _load_default_select_by_collection() -> Dict[str, List[str]]:
    """Load data.default_select_by_collection from config.json. Cached on first call."""
    global _default_select_by_collection
    if _default_select_by_collection is not None:
        return _default_select_by_collection
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "config.json"
    )
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        data_config = config.get("data") or {}
        _default_select_by_collection = dict(data_config.get("default_select_by_collection") or {})
    except Exception:
        _default_select_by_collection = {}
    return _default_select_by_collection


def _get_select_fields_for_collection(collection: str) -> Optional[List[str]]:
    """
    Return the configured default fields for a collection, or None (all fields).
    Ignores LLM/user select; config is the sole source of truth.
    """
    defaults = _load_default_select_by_collection()
    fields = defaults.get(collection)
    return list(fields) if fields else None


def _load_default_facet_by_collection() -> Dict[str, List[str]]:
    """Load data.default_facet_by_collection from config.json. Cached on first call."""
    global _default_facet_by_collection
    if _default_facet_by_collection is not None:
        return _default_facet_by_collection
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "config.json"
    )
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        data_config = config.get("data") or {}
        _default_facet_by_collection = dict(data_config.get("default_facet_by_collection") or {})
    except Exception:
        _default_facet_by_collection = {}
    return _default_facet_by_collection


def _get_facet_fields_for_collection(collection: str) -> Optional[List[str]]:
    """
    Return the configured default facet fields for a collection, or None if
    faceting is not configured for this collection.
    """
    defaults = _load_default_facet_by_collection()
    fields = defaults.get(collection)
    return list(fields) if fields else None


def _clip_log_text(text: Any, max_len: int = 180) -> str:
    """Keep debug log fields concise and single-line."""
    s = str(text or "").replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return f"{s[:max_len]}..."


def _normalize_cancel_token(cancel_token: Optional[str]) -> str:
    return str(cancel_token or "").strip()


def _mark_download_cancelled(cancel_token: Optional[str]) -> bool:
    token = _normalize_cancel_token(cancel_token)
    if not token:
        return False
    _download_cancel_tokens[token] = True
    return True


def request_download_cancel(cancel_token: Optional[str]) -> bool:
    """Public helper for HTTP routes to request cooperative download cancellation."""
    return _mark_download_cancelled(cancel_token)


def _is_download_cancelled(cancel_token: Optional[str]) -> bool:
    token = _normalize_cancel_token(cancel_token)
    if not token:
        return False
    return bool(_download_cancel_tokens.get(token, False))


def _clear_download_cancel_token(cancel_token: Optional[str]) -> None:
    token = _normalize_cancel_token(cancel_token)
    if token:
        _download_cancel_tokens.pop(token, None)


def _get_llm_client():
    """Create and cache the internal LLM client used for query planning."""
    global _llm_client
    if _llm_client is not None:
        return _llm_client

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "config.json"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    _llm_client = create_llm_client_from_config(config)
    return _llm_client


def _contains_solr_syntax(query: str) -> bool:
    """Detect common raw Solr syntax patterns to enforce natural-language input."""
    if not query:
        return False

    # Fielded search (e.g., species:"E. coli"), boolean operators, range/grouping operators.
    solr_patterns = [
        r"\b[A-Za-z_][A-Za-z0-9_]*\s*:",  # field:
        r"\b(AND|OR|NOT)\b",
        r"[\(\)\[\]\{\}]",
        r"\bTO\b",
        r"[*?~^]",
    ]
    return any(re.search(pattern, query) for pattern in solr_patterns)


# Stopwords to exclude from keyword queries
STOPWORDS = {
    # articles & determiners
    "a", "an", "the", "this", "that", "these", "those", "each", "every", "either", "neither",

    # prepositions
    "from", "to", "in", "on", "at", "by", "for", "with", "of", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below",
    "over", "under", "within", "without", "across", "per",

    # conjunctions
    "and", "or", "but", "nor", "yet", "so", "while", "although", "because", "if",

    # verbs (auxiliary / common)
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
    "will", "would", "shall", "should",
    "can", "could", "may", "might", "must",

    # pronouns
    "i", "me", "my", "mine",
    "you", "your", "yours",
    "he", "him", "his",
    "she", "her", "hers",
    "it", "its",
    "we", "us", "our", "ours",
    "they", "them", "their", "theirs",

    # interrogatives
    "what", "which", "who", "whom", "whose",
    "when", "where", "why", "how",

    # quantifiers & comparatives
    "some", "any", "all", "none", "many", "much", "few", "less", "more", "most",
    "several", "such", "same", "other", "another", 

    # numbers & counting terms
    "count", "counts", "total", "number", "numbers", "amount", "average", "sum",

    # generic descriptors
    "type", "types", "kind", "kinds", "example", "examples", "case", "cases", "found", "find"
}

# Custom domain-specific stopwords to exclude
CUSTOM_STOPWORDS = {
    # genome & taxonomy boilerplate
    "genome", "genomes", "genomic", "taxa", "taxon", "species", "strain", "strains",
    "subtype", "clade", "lineage", "serotype", "pathways", "pathway", "subsystems", "subsystem",
    "genus", "taxonomy", "taxonomic", "taxonomies", "taxonomy_id", "taxonomy_ids", "taxonomy_name", "taxonomy_names",
    "protein", "proteins", "protein_id", "protein_ids", "protein_name", "protein_names",
    "structures", "structure", "structure_id", "structure_ids", "structure_name", "structure_names",
    "domains", "domain", "domain_id", "domain_ids", "domain_name", "domain_names",
    "families", "family", "family_id", "family_ids", "family_name", "family_names",
    "clusters", "cluster", "cluster_id", "cluster_ids", "cluster_name", "cluster_names",
    "clusters", "cluster", "cluster_id", "cluster_ids", "cluster_name", "cluster_names",
    "specialty", "sp_gene"

    # identifiers & metadata
    "id", "ids", "accession", "accessions", "identifier", "identifiers",
    "version", "versions", "release", "date", "year",

    # database / platform terms
    "bv-brc", "bvbrc", "database", "platform", "resource", "portal",
    "dataset", "datasets", "collection", "collections", 

    # feature & annotation noise
    "feature", "features", "annotation", "annotations", "annotated",
    "gene", "genes", "proteins", "product", "products",
    "function", "functions",

    # search / UI / reporting terms
    "find", "search", "query", "results", "result",
    "record", "records", "entry", "entries",
    "summary", "description", "describe", "details", "summarize"

    # geography boilerplate
    "country", "countries", "location", "locations", "region", "regions",

    # relationship fluff
    "related", "associated", "including", "includes", "based", "using",

    # overly generic science words
    "data", "information", "analysis", "study", "studies", "sample", "samples"
    "sequences", "summarize", "summary", "description", "describe", "details",
    "files", "reads", "in", "sra", "archive"
}

# Normalize plural/variant forms to canonical terms for consistent search
REPLACE_WORDS = {
    "proteins": "protein",
    "flu": "influenza"
}


# ---------------------------------------------------------------------------
# Auto-quoting for agent-generated Solr queries
# ---------------------------------------------------------------------------

# Regex: match  field_name:UnquotedWord1 Word2 ...WordN
# where the words are NOT already quoted, not inside brackets/parens,
# and are followed by AND/OR/NOT, another field:value, or end of string.
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
    """
    if not query or query == "*:*":
        return query

    def _replacer(m: re.Match) -> str:
        field = m.group(1)
        value = m.group(2)
        return f'{field}:"{value}"'

    return _UNQUOTED_MULTIWORD_RE.sub(_replacer, query)


def _tokenize_keywords(text: str) -> List[str]:
    """
    Convert a natural-language query into search terms for keyword mode.
    Supports comma-separated groups and fallback token splitting.
    Filters out common stopwords and custom stopwords.
    """
    if not text:
        return []
    # Decode URL-encoded query text so stopword filtering works for inputs like
    # "genome%20208964.12" and "genome+208964.12".
    text = unquote_plus(str(text))

    # Apply word replacements (e.g. plural → singular) for consistent search
    for old_word, new_word in REPLACE_WORDS.items():
        text = re.sub(r"\b" + re.escape(old_word) + r"\b", new_word, text, flags=re.IGNORECASE)

    # Combine all stopwords
    all_stopwords = STOPWORDS | CUSTOM_STOPWORDS

    def _clean_non_stopword_terms(raw_text: str) -> List[str]:
        parts = re.split(r"\s+", raw_text.strip())
        terms: List[str] = []
        for part in parts:
            cleaned = re.sub(r"^[^\w]+|[^\w]+$", "", part)
            if cleaned and cleaned.lower() not in all_stopwords:
                terms.append(cleaned)
        return terms

    # Prefer comma-separated groups if present; still apply stopword filtering.
    if "," in text:
        groups = [group.strip() for group in text.split(",")]
        terms: List[str] = []
        for group in groups:
            if not group:
                continue
            filtered_group_terms = _clean_non_stopword_terms(group)
            if filtered_group_terms:
                terms.append(" ".join(filtered_group_terms))
        return terms

    # Otherwise split on whitespace and strip punctuation.
    return _clean_non_stopword_terms(text)


def _quote_solr_term(term: str) -> str:
    """Quote and minimally escape a term for safe Solr q= usage."""
    escaped = str(term).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _is_count_only_query(user_query: str) -> bool:
    """
    Detect explicit count-intent phrasing in natural-language global search queries.
    """
    text = str(user_query or "").strip().lower()
    if not text:
        return False
    count_patterns = [
        r"^\s*how\s+many\b",
        r"\bnumber\s+of\b",
        r"\btotal\s+number\s+of\b",
        r"\bcount\s+of\b",
        r"^\s*count\b",
    ]
    return any(re.search(pattern, text) for pattern in count_patterns)


def _strip_count_only_intent_text(user_query: str) -> str:
    """
    Remove common count-intent phrasing so keyword extraction focuses on entities.
    """
    text = str(user_query or "").strip()
    if not text:
        return text
    normalized = re.sub(r"^\s*how\s+many\s+", "", text, flags=re.IGNORECASE)
    normalized = re.sub(r"\b(total\s+)?number\s+of\s+", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bcount\s+of\s+", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"^\s*count\s+", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+\b(are|is|there)\b\s*\??\s*$", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip(" ?.")
    return normalized


def _build_global_search_q_expr(user_query: str) -> Dict[str, Any]:
    """
    Build a Solr q expression from natural language for AND keyword mode.
    Returns dict containing q_expr and parsed keywords for transparency.
    """
    count_only = _is_count_only_query(user_query)
    keyword_source = _strip_count_only_intent_text(user_query) if count_only else user_query
    keywords = _tokenize_keywords(keyword_source)
    if not keywords and keyword_source != user_query:
        keywords = _tokenize_keywords(user_query)
    if not keywords:
        raise ValueError("Could not parse keywords from user_query")

    # Single keyword still uses quoted term matching.
    if len(keywords) == 1:
        return {
            "q_expr": _quote_solr_term(keywords[0]),
            "keywords": keywords,
            "searchMode": "and",
            "countOnly": count_only
        }

    q_expr = "(" + " AND ".join(_quote_solr_term(term) for term in keywords) + ")"
    return {
        "q_expr": q_expr,
        "keywords": keywords,
        "searchMode": "and",
        "countOnly": count_only
    }


def _build_rql_keyword_query(keywords: List[str]) -> str:
    """
    Build an RQL keyword query that mirrors the sanitized keyword intent.
    Do not include paging controls; replay clients can apply them separately.
    """
    # Defensive sanitization: callers may accidentally pass raw user text or
    # mixed keyword fragments. Normalize back through tokenizer to enforce
    # stopword stripping before constructing replayable RQL.
    if isinstance(keywords, str):
        normalized_keywords = _tokenize_keywords(keywords)
    elif isinstance(keywords, (list, tuple)):
        combined = ", ".join(str(term).strip() for term in keywords if str(term).strip())
        normalized_keywords = _tokenize_keywords(combined)
    else:
        normalized_keywords = []
    keyword_text = " ".join(str(term).strip() for term in normalized_keywords if str(term).strip())
    safe_keyword_text = keyword_text.replace(")", "\\)")
    return_parts = []

    for word in safe_keyword_text.split():
        if any(char.isdigit() for char in word):
            word = f'"{word}"'
        return_parts.append(f"keyword({word})")

    return_str = "and(" + ",".join(return_parts) + ")"
    return return_str


def _looks_like_patric_feature_id(value: str) -> bool:
    """
    Detect canonical PATRIC genome_feature IDs:
    fig|<genome_id>.peg.<feature_number>
    """
    return bool(re.match(r"^fig\|\d+\.\d+\.peg\.\d+$", str(value or "").strip(), re.IGNORECASE))


def _escape_rql_value(value: str) -> str:
    """Escape RQL value delimiters used inside function argument lists."""
    text = str(value or "")
    return text.replace("\\", "\\\\").replace(",", "\\,").replace(")", "\\)")


def _apply_collection_solr_additions(collection: str, q_expr: str, **ctx) -> str:
    """Apply collection-specific additions to the Solr q expression."""
    if collection == "genome_feature":
        return f"({q_expr}) AND patric_id:*"
    return q_expr


def _build_solr_select_options(collection: str) -> Dict[str, Any]:
    """
    Build options dict with select fields from config for the collection.
    Returns {} or {"select": [field1, field2, ...]} with only allowed fields.
    """
    options: Dict[str, Any] = {}
    select_fields = _get_select_fields_for_collection(collection)
    if not select_fields:
        return options
    allowed = set(get_collection_fields(collection))
    if allowed:
        select_fields = [f for f in select_fields if f in allowed]
    if select_fields:
        options["select"] = select_fields
    return options


def _apply_collection_rql_additions(collection: str, rql_query: str, **ctx) -> str:
    """Apply collection-specific additions to the RQL query."""
    keywords = ctx.get("sanitized_keywords", [])
    if collection == "genome_feature":
        if len(keywords) == 1 and _looks_like_patric_feature_id(keywords[0]):
            escaped_id = _escape_rql_value(keywords[0])
            rql_query = f"and(eq(annotation,PATRIC),eq(patric_id,{escaped_id}))"
        else:
            rql_query = f"and(eq(annotation,PATRIC),{rql_query})"
    return rql_query


def _build_rql_replay_query(rql_query: str, collection: Optional[str] = None, limit: int = 100) -> str:
    """
    Build a URL-safe RQL replay string for BV-BRC links.
    Format: ?{query} — select fields are not added to RQL (they remain in Solr only).
    """
    encoded_rql = quote(str(rql_query or ""), safe="(),:*")
    return f"?{encoded_rql}"


def convert_json_to_tsv(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert a list of JSON objects to TSV format using jq.
    
    Args:
        results: List of dictionaries to convert
        
    Returns:
        Dict with either:
        - {"tsv": <tsv_string>} on success
        - {"error": <error_message>} on failure
    """
    if not results:
        # Empty results - return empty TSV
        print("  TSV conversion: Empty results, returning empty TSV")
        return {"tsv": ""}
    
    print(f"  TSV conversion: Starting conversion of {len(results)} results to TSV format...")
    
    try:
        # Prepare jq command to convert JSON array to TSV
        # This extracts all unique keys from the first object as headers,
        # then converts each object to a TSV row
        jq_command = [
            "jq",
            "-r",
            "(.[0] | keys_unsorted) as $keys | $keys, (.[] | [.[$keys[]] | tostring]) | @tsv"
        ]
        
        # Convert results to JSON string
        json_input = json.dumps(results)
        print(f"  TSV conversion: Running jq command to convert JSON to TSV...")
        
        # Run jq command
        result = subprocess.run(
            jq_command,
            input=json_input,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print(f"  TSV conversion: FAILED - jq returned error code {result.returncode}")
            print(f"  TSV conversion error: {result.stderr}")
            return {
                "error": f"jq conversion failed: {result.stderr}",
                "hint": "Ensure jq is installed on your system"
            }
        
        tsv_lines = result.stdout.count('\n')
        print(f"  TSV conversion: SUCCESS - Converted to TSV with {tsv_lines} lines (including header)")
        return {"tsv": result.stdout}
        
    except FileNotFoundError:
        print("  TSV conversion: FAILED - jq command not found")
        return {
            "error": "jq command not found",
            "hint": "Please install jq to use TSV format. See https://jqlang.github.io/jq/download/"
        }
    except subprocess.TimeoutExpired:
        print("  TSV conversion: FAILED - Conversion timed out after 30 seconds")
        return {
            "error": "TSV conversion timed out (>30s)",
            "hint": "Try reducing the batch size"
        }
    except Exception as e:
        print(f"  TSV conversion: FAILED - Unexpected error: {str(e)}")
        return {
            "error": f"TSV conversion error: {str(e)}"
        }


def register_data_tools(mcp: FastMCP, base_url: str, token_provider=None):
    """
    Register all MVP-related MCP tools with the FastMCP server.
    
    Args:
        mcp: FastMCP server instance
        base_url: Base URL for BV-BRC API
        token_provider: TokenProvider instance for handling authentication tokens (optional)
    """
    global _base_url, _token_provider
    _base_url = base_url
    _token_provider = token_provider

    def _build_auth_headers(token: Optional[str]) -> Optional[Dict[str, str]]:
        """Build auth headers using token provider when configured."""
        headers: Optional[Dict[str, str]] = None
        if _token_provider:
            auth_token = _token_provider.get_token(token)
            if auth_token:
                headers = {"Authorization": auth_token}
        elif token:
            headers = {"Authorization": token}
        return headers

    def _apply_sequence_response_mode_headers(
        collection: str,
        headers: Optional[Dict[str, str]],
        sequence_response_mode: str
    ) -> Dict[str, str]:
        """
        Apply planner-selected sequence response mode by setting API bulk header.
        Enum intentionally captures both collection and sequence type.
        """
        mode = str(sequence_response_mode or "none").strip().lower()
        out_headers = dict(headers or {})
        if mode in {"", "none"}:
            return out_headers
        if mode == "genome_feature_dna_fasta":
            if collection != "genome_feature":
                raise ValueError("genome_feature_dna_fasta is only valid for collection 'genome_feature'")
            out_headers["http_accept"] = "application/dna+fasta"
            return out_headers
        if mode == "genome_feature_protein_fasta":
            if collection != "genome_feature":
                raise ValueError("genome_feature_protein_fasta is only valid for collection 'genome_feature'")
            out_headers["http_accept"] = "application/protein+fasta"
            return out_headers
        raise ValueError(
            "sequence_response_mode must be one of: none, genome_feature_dna_fasta, genome_feature_protein_fasta"
        )

    async def _execute_structured_query(
        collection: str,
        filters: Optional[Dict[str, Any]],
        select: Optional[Any],
        sort: Optional[Any],
        cursorId: Optional[str],
        countOnly: bool,
        batchSize: Optional[int],
        num_results: Optional[int],
        result_format: str,
        token: Optional[str],
        sequence_response_mode: str = "none",
        cancel_token: Optional[str] = None,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """Execute a structured single-collection query."""
        options = _build_solr_select_options(collection)
        sort_expr = normalize_sort(sort)
        if sort_expr:
            options["sort"] = sort_expr

        allowed_fields = set(get_collection_fields(collection))
        invalid_fields = validate_filter_fields(filters, allowed_fields) if filters else []
        if invalid_fields:
            sample_fields = sorted(list(allowed_fields))[:25] if allowed_fields else []
            return {
                "error": f"Invalid field(s) for collection '{collection}': {', '.join(invalid_fields)}",
                "hint": "Call bvbrc_collection_fields_and_parameters to see valid fields.",
                "allowedFieldsSample": sample_fields,
                "source": "bvbrc-mcp-data"
            }

        filter_str = build_filter(filters)
        if collection == "genome_feature":
            auto = "patric_id:*"
            if filter_str and filter_str != "*:*":
                filter_str = f"({filter_str}) AND {auto}"
            else:
                filter_str = auto

        if batchSize is not None and (batchSize < 1 or batchSize > 10000):
            return {
                "error": f"Invalid batchSize: {batchSize}. Must be between 1 and 10000.",
                "source": "bvbrc-mcp-data"
            }

        if num_results is not None:
            if num_results < 1:
                return {
                    "error": f"Invalid num_results: {num_results}. Must be >= 1.",
                    "source": "bvbrc-mcp-data"
                }
            if batchSize is None:
                batchSize = min(CURSOR_BATCH_SIZE, num_results)
            elif batchSize > num_results:
                batchSize = num_results

        try:
            headers = _build_auth_headers(token)
            headers = _apply_sequence_response_mode_headers(
                collection=collection,
                headers=headers,
                sequence_response_mode=sequence_response_mode
            )

            # Server-side cursor pagination with disk spooling:
            # fetch pages in batches of 1000 and write each batch to a temp file to avoid
            # retaining all pages in memory during long-running queries.
            if not countOnly and (cursorId is None or cursorId == "*"):
                spool_batch_size = CURSOR_BATCH_SIZE
                target_results = num_results if num_results is not None else None
                total_fetched = 0
                total_num_found: Optional[int] = None
                next_cursor_id: Optional[str] = None
                total_batches = 0
                cursor = cursorId or "*"

                with tempfile.TemporaryDirectory(prefix="bvbrc_query_spool_") as spool_dir:
                    jsonl_path = os.path.join(spool_dir, "results.jsonl")
                    tsv_path = os.path.join(spool_dir, "results.tsv")
                    wrote_tsv_header = False

                    while cursor:
                        if _is_download_cancelled(cancel_token):
                            return {
                                "error": "Data download cancelled.",
                                "errorType": "CANCELLED",
                                "cancelled": True,
                                "count": total_fetched,
                                "numFound": total_num_found if total_num_found is not None else total_fetched,
                                "source": "bvbrc-mcp-data"
                            }
                        if target_results is not None and total_fetched >= target_results:
                            break

                        if target_results is not None:
                            remaining = target_results - total_fetched
                            page_batch_size = min(spool_batch_size, remaining)
                        else:
                            page_batch_size = spool_batch_size

                        page_result = await query_direct(
                            collection,
                            filter_str,
                            options,
                            _base_url,
                            headers=headers,
                            cursorId=cursor,
                            countOnly=False,
                            batch_size=page_batch_size
                        )
                        page_results = page_result.get("results", [])
                        if not page_results:
                            next_cursor_id = None
                            break

                        if target_results is not None and len(page_results) > (target_results - total_fetched):
                            page_results = page_results[: target_results - total_fetched]

                        if result_format == "tsv":
                            conversion_result = convert_json_to_tsv(page_results)
                            if "error" in conversion_result:
                                return {
                                    "error": conversion_result["error"],
                                    "hint": conversion_result.get("hint", ""),
                                    "count": total_fetched,
                                    "numFound": total_num_found if total_num_found is not None else total_fetched,
                                    "nextCursorId": page_result.get("nextCursorId"),
                                    "source": "bvbrc-mcp-data"
                                }
                            batch_tsv = conversion_result.get("tsv", "")
                            tsv_lines = [line for line in batch_tsv.splitlines() if line.strip()]
                            if tsv_lines:
                                with open(tsv_path, "a", encoding="utf-8") as tsv_file:
                                    if not wrote_tsv_header:
                                        tsv_file.write("\n".join(tsv_lines) + "\n")
                                        wrote_tsv_header = True
                                    else:
                                        data_lines = tsv_lines[1:] if len(tsv_lines) > 1 else []
                                        if data_lines:
                                            tsv_file.write("\n".join(data_lines) + "\n")
                        else:
                            with open(jsonl_path, "a", encoding="utf-8") as jsonl_file:
                                for row in page_results:
                                    jsonl_file.write(json.dumps(row, ensure_ascii=False))
                                    jsonl_file.write("\n")

                        total_batches += 1
                        total_fetched += len(page_results)
                        total_num_found = page_result.get("numFound", total_num_found)
                        next_cursor = page_result.get("nextCursorId")
                        next_cursor_id = next_cursor

                        print(
                            f"[search_data_progress] collection={collection} batch={total_batches} "
                            f"fetched={total_fetched} numFound={total_num_found} "
                            f"hasNextCursor={bool(next_cursor)}"
                        )
                        if ctx is not None:
                            progress_total = float(target_results) if target_results is not None else (
                                float(total_num_found) if total_num_found is not None else None
                            )
                            await ctx.report_progress(
                                progress=float(total_fetched),
                                total=progress_total,
                                message=(
                                    f"Fetched {total_fetched} records"
                                    f" (batch {total_batches}, collection={collection})"
                                )
                            )

                        if not next_cursor:
                            break
                        cursor = next_cursor

                    result = {
                        "count": total_fetched,
                        "numFound": total_num_found if total_num_found is not None else total_fetched,
                        "nextCursorId": next_cursor_id if (target_results is not None and total_fetched >= target_results) else None,
                        "_paginationInfo": {
                            "totalBatches": total_batches,
                            "batchSize": spool_batch_size,
                            "spooledToTempFiles": True
                        }
                    }
                    if target_results is not None:
                        result["limit"] = target_results
                        result["limitReached"] = total_fetched >= target_results

                    if result_format == "tsv":
                        if os.path.exists(tsv_path):
                            with open(tsv_path, "r", encoding="utf-8") as tsv_file:
                                result["tsv"] = tsv_file.read()
                        else:
                            result["tsv"] = ""
                    else:
                        merged_results: List[Dict[str, Any]] = []
                        if os.path.exists(jsonl_path):
                            with open(jsonl_path, "r", encoding="utf-8") as jsonl_file:
                                for line in jsonl_file:
                                    text = line.strip()
                                    if text:
                                        merged_results.append(json.loads(text))
                        result["results"] = merged_results
            else:
                result = await query_direct(
                    collection,
                    filter_str,
                    options,
                    _base_url,
                    headers=headers,
                    cursorId=cursorId,
                    countOnly=countOnly,
                    batch_size=batchSize
                )

            if result_format == "tsv" and not countOnly and "results" in result:
                conversion_result = convert_json_to_tsv(result.get("results", []))
                if "error" in conversion_result:
                    return {
                        "error": conversion_result["error"],
                        "hint": conversion_result.get("hint", ""),
                        "results": result.get("results", []),
                        "count": result.get("count"),
                        "numFound": result.get("numFound"),
                        "nextCursorId": result.get("nextCursorId"),
                        "source": "bvbrc-mcp-data"
                    }
                result["tsv"] = conversion_result["tsv"]
                del result["results"]

            result["source"] = "bvbrc-mcp-data"
            return result
        except Exception as e:
            return {
                "error": f"Error querying {collection}: {str(e)}",
                "source": "bvbrc-mcp-data"
            }

    # @mcp.tool(annotations={"readOnlyHint": True})
    def bvbrc_collection_fields_and_parameters(collection: str) -> str:
        """
        Get fields and query parameters for a given BV-BRC collection.
        
        Args:
            collection: The collection name (e.g., "genome")
        
        Returns:
            String with the parameters for the given collection
        """
        return lookup_parameters(collection)

    # @mcp.tool(annotations={"readOnlyHint": True})
    def bvbrc_list_collections() -> str:
        """
        List all available BV-BRC collections.
        
        Returns:
            String with the available collections
        """
        print("Fetching available collections.")
        return list_solr_collections()

    @mcp.tool(annotations={"readOnlyHint": True, "streamingHint": True})
    async def bvbrc_search_data(
        user_query: str,
        advanced: bool = False,
        count: bool = False,
        snapshot_limit: Optional[int] = None,
        cancel_token: Optional[str] = None,
        stream: bool = False,
        token: Optional[str] = None,
        ctx: Context = None
    ) -> Dict[str, Any]:
        """
        Unified natural-language data search tool for BV-BRC.

        Default behavior (`advanced=False`) is exploratory global search:
        - Select a likely collection with internal LLM routing
        - Execute a q= search using AND semantics
        - Returns a snapshot of results (default 100) plus the total count
        - If count=True, return only the matching total

        Advanced behavior (`advanced=True`) is targeted retrieval:
        - Plan a structured query from user_query
        - Execute the resulting single-collection query with validated fields
        - Pagination/result controls are determined internally by the planner

        Args:
            user_query: Natural language query text or keyword list.
            advanced: False (default) for global discovery; True for targeted structured query execution.
            count: If True, return only the count for the planner-selected query.
            snapshot_limit: Maximum number of rows to return in the snapshot (default from config, typically 100). Must be between 1 and 10000.
            cancel_token: Internal cancellation token (set by API, not user-facing).
            stream: Internal transport flag used by API/MCP streaming pipeline.
            token: Authentication token (optional, auto-detected if token_provider is configured).
        """
        normalized_cancel_token = _normalize_cancel_token(cancel_token)
        if not user_query or not str(user_query).strip():
            return {
                "error": "user_query parameter is required",
                "errorType": "INVALID_PARAMETERS",
                "hint": "Provide a natural-language query or keyword list",
                "source": "bvbrc-mcp-data"
            }

        # This is here intentionally to disable advanced mode. We will remove this once we have a proper advanced mode.
        advanced = False

        query_text = str(user_query).strip()
        if not advanced:
            if _contains_solr_syntax(query_text):
                return {
                    "error": "user_query appears to contain Solr syntax, which is not allowed for this tool.",
                    "errorType": "INVALID_PARAMETERS",
                    "hint": "Provide natural language or plain keywords only.",
                    "source": "bvbrc-mcp-data"
                }

            try:
                search_info = _build_global_search_q_expr(query_text)
                sanitized_keywords = _tokenize_keywords(", ".join(search_info.get("keywords", [])))
                rql_query = _build_rql_keyword_query(sanitized_keywords)
                print(
                    "[bvbrc_search_data:rql] "
                    f"query='{_clip_log_text(query_text)}' "
                    f"raw_keywords={search_info.get('keywords', [])} "
                    f"sanitized_keywords={sanitized_keywords} "
                    f"rql='{rql_query}'"
                )
                llm_client = _get_llm_client()
                selection = select_collection_for_query(query_text, llm_client)
                print(f"[bvbrc_search_data] Planning output (global mode): {json.dumps(selection, indent=2)}")
                collection = str(selection.get("collection", "")).strip()
                if not collection:
                    return {
                        "error": "Collection selection failed to produce a collection.",
                        "errorType": "PLANNING_FAILED",
                        "selection": selection,
                        "source": "bvbrc-mcp-data"
                    }

                q_expr = str(search_info.get("q_expr", "")).strip() or "*:*"
                q_expr = _apply_collection_solr_additions(collection, q_expr, sanitized_keywords=sanitized_keywords)
                rql_query = _apply_collection_rql_additions(collection, rql_query, sanitized_keywords=sanitized_keywords)
                rql_replay_query = _build_rql_replay_query(rql_query, collection=collection, limit=100)
                print(
                    "[bvbrc_search_data:rql_replay] "
                    f"collection='{collection}' "
                    f"rql='{rql_query}' "
                    f"replay='{rql_replay_query}'"
                )

                headers = _build_auth_headers(token)
                options = _build_solr_select_options(collection)

                # --- Count / faceting branch ---
                # Use the existing deterministic regex to detect count intent,
                # then return faceted counts using the configured fields for
                # the selected collection.
                count_query = _is_count_only_query(query_text)
                if count_query:
                    facet_fields = _get_facet_fields_for_collection(collection)
                    if facet_fields:
                        print(
                            f"[bvbrc_search_data:facet] Executing facet query: "
                            f"collection='{collection}', q='{q_expr}', "
                            f"facet_fields={facet_fields}"
                        )
                        facet_result = await query_faceted(
                            core=collection,
                            filter_str=q_expr,
                            facet_fields=facet_fields,
                            base_url=_base_url,
                            headers=headers,
                        )
                        # Convert facet buckets to a flat list of dicts for TSV
                        facet_rows: List[Dict[str, Any]] = []
                        for field, buckets in facet_result.get("facets", {}).items():
                            for bucket in buckets:
                                facet_rows.append({
                                    "field": field,
                                    "value": bucket["value"],
                                    "count": bucket["count"],
                                })
                        conversion_result = convert_json_to_tsv(facet_rows)
                        tsv = conversion_result.get("tsv", "") if "error" not in conversion_result else None
                        result_payload: Dict[str, Any] = {
                            "mode": "faceted",
                            "collection": collection,
                            "numFound": facet_result.get("numFound", 0),
                            "facets": facet_result.get("facets", {}),
                            "call": {
                                "tool": "bvbrc_search_data",
                                "backend_method": "data_functions.query_faceted",
                                "replayable": True,
                                "arguments_executed": {
                                    "mode": "faceted",
                                    "collection": collection,
                                    "selection": selection,
                                    "searchMode": search_info.get("searchMode"),
                                    "keywords": sanitized_keywords,
                                    "q": q_expr,
                                    "facet_fields": facet_fields,
                                    "data_api_base_url": _base_url,
                                },
                                "replay": {
                                    "rql_query": rql_query,
                                    "rql_replay_query": rql_replay_query,
                                },
                                "source": "bvbrc-mcp-data",
                            },
                        }
                        if tsv is not None:
                            result_payload["tsv"] = tsv
                        return result_payload
                    else:
                        print(
                            f"[bvbrc_search_data:count] Count query detected but no "
                            f"default facet fields configured for '{collection}', "
                            f"falling through to normal count"
                        )

                count_only = bool(count)
                if count_only:
                    count_result = await query_direct(
                        core=collection,
                        filter_str=q_expr,
                        options=options,
                        base_url=_base_url,
                        headers=headers,
                        cursorId="*",
                        countOnly=True,
                        batch_size=CURSOR_BATCH_SIZE,
                    )
                    count_value = int(count_result.get("numFound", 0) or 0)
                    return {
                        "countOnly": True,
                        "value": count_value,
                        "count": count_value,
                        "numFound": count_value,
                        "source": "bvbrc-mcp-data",
                        "call": {
                            "tool": "bvbrc_search_data",
                            "backend_method": "data_functions.query_direct",
                            "replayable": True,
                            "arguments_executed": {
                                "mode": "global",
                                "collection": collection,
                                "selection": selection,
                                "searchMode": search_info.get("searchMode"),
                                "keywords": sanitized_keywords,
                                "q": q_expr,
                                "data_api_base_url": _base_url,
                            },
                            "replay": {
                                "rql_query": rql_query,
                                "rql_replay_query": rql_replay_query,
                            },
                            "source": "bvbrc-mcp-data",
                        },
                    }

                # --- Snapshot mode ---
                # Fetch a bounded snapshot of results (default 100) instead of
                # exhausting all cursor pages.  The total count (numFound) is
                # returned so the LLM and client can communicate the full scope
                # of the result set, along with a download URL for the complete
                # dataset.
                effective_snapshot_limit = snapshot_limit if snapshot_limit is not None else _load_default_snapshot_limit()
                if effective_snapshot_limit < 1:
                    effective_snapshot_limit = 1
                elif effective_snapshot_limit > 10000:
                    effective_snapshot_limit = 10000

                print(
                    f"[bvbrc_search_data:snapshot] Fetching snapshot: "
                    f"collection='{collection}', limit={effective_snapshot_limit}"
                )

                snapshot_result = await query_direct(
                    core=collection,
                    filter_str=q_expr,
                    options=options,
                    base_url=_base_url,
                    headers=headers,
                    cursorId="*",
                    countOnly=False,
                    batch_size=effective_snapshot_limit
                )
                snapshot_rows = snapshot_result.get("results", [])
                total_num_found = snapshot_result.get("numFound", len(snapshot_rows))
                snapshot_count = len(snapshot_rows)
                is_snapshot = snapshot_count < total_num_found

                if ctx is not None:
                    await ctx.report_progress(
                        progress=float(snapshot_count),
                        total=float(total_num_found),
                        message=(
                            f"Fetched {snapshot_count} of {total_num_found} records"
                            f" (collection={collection})"
                        )
                    )

                # Build the full-dataset download URL for TSV export
                base_api = str(_base_url or "https://www.bv-brc.org/api-bulk").rstrip("/")
                download_rql = rql_replay_query
                if download_rql and download_rql.startswith("?"):
                    download_rql = download_rql[1:]
                sort_field = _COLLECTION_KEY_FIELD.get(collection, "id")
                download_url = (
                    f"{base_api}/{collection}/"
                    f"?{download_rql}"
                    f"&http_accept={quote('text/tsv')}"
                    f"&http_download=true"
                    f"&limit({total_num_found})"
                    f"&sort(+{sort_field})"
                )

                # Convert snapshot rows to TSV
                conversion_result = convert_json_to_tsv(snapshot_rows)

                call_metadata = {
                    "tool": "bvbrc_search_data",
                    "backend_method": "data_functions.query_direct",
                    "replayable": True,
                    "arguments_executed": {
                        "mode": "global",
                        "collection": collection,
                        "selection": selection,
                        "searchMode": search_info.get("searchMode"),
                        "keywords": sanitized_keywords,
                        "q": q_expr,
                        "data_api_base_url": _base_url,
                    },
                    "replay": {
                        "rql_query": rql_query,
                        "rql_replay_query": rql_replay_query,
                        "download_url": download_url,
                    },
                    "source": "bvbrc-mcp-data",
                }

                if "error" in conversion_result:
                    return {
                        "error": conversion_result["error"],
                        "hint": conversion_result.get("hint", ""),
                        "results": snapshot_rows,
                        "count": snapshot_count,
                        "numFound": total_num_found,
                        "is_snapshot": is_snapshot,
                        "snapshot_limit": effective_snapshot_limit,
                        "collection": collection,
                        "source": "bvbrc-mcp-data",
                        "call": call_metadata,
                    }

                print(
                    f"[bvbrc_search_data:snapshot] Returning snapshot: "
                    f"count={snapshot_count}, numFound={total_num_found}, "
                    f"is_snapshot={is_snapshot}, collection='{collection}'"
                )

                return {
                    "tsv": conversion_result["tsv"],
                    "count": snapshot_count,
                    "numFound": total_num_found,
                    "is_snapshot": is_snapshot,
                    "snapshot_limit": effective_snapshot_limit,
                    "collection": collection,
                    "nextCursorId": None,
                    "source": "bvbrc-mcp-data",
                    "call": call_metadata,
                }
            except Exception as e:
                return {
                    "error": f"Global data search failed: {str(e)}",
                    "errorType": "SEARCH_FAILED",
                    "source": "bvbrc-mcp-data"
                }

        # advanced=True: plan + execute targeted query
        try:
            llm_client = _get_llm_client()
            planning_result = create_query_plan_internal(query_text, llm_client)
            print(f"[bvbrc_search_data] Planning output (advanced mode): {json.dumps(planning_result, indent=2)}")
            if "error" in planning_result:
                return {
                    "error": planning_result.get("error"),
                    "errorType": "PLANNING_FAILED",
                    "selection": planning_result.get("selection"),
                    "validationError": planning_result.get("validationError"),
                    "rawPlan": planning_result.get("rawPlan"),
                    "source": "bvbrc-mcp-data"
                }

            plan = planning_result.get("plan", {})
            collection = plan.get("collection")
            if not collection:
                return {
                    "error": "Planner did not return a target collection.",
                    "errorType": "PLANNING_FAILED",
                    "selection": planning_result.get("selection"),
                    "rawPlan": plan,
                    "source": "bvbrc-mcp-data"
                }

            result = await _execute_structured_query(
                collection=collection,
                filters=plan.get("filters"),
                select=plan.get("select"),
                sort=plan.get("sort"),
                cursorId=plan.get("cursorId"),
                countOnly=bool(plan.get("countOnly", False)),
                batchSize=plan.get("batchSize"),
                num_results=plan.get("num_results"),
                result_format=plan.get("format", "tsv"),
                token=token,
                sequence_response_mode=plan.get("sequence_response_mode", "none"),
                cancel_token=normalized_cancel_token,
                ctx=ctx,
            )

            if "error" in result:
                return result

            result["mode"] = "advanced"
            result["data_api_base_url"] = _base_url
            result["selection"] = planning_result.get("selection", {})
            # Keep planner internals private from tool consumers.
            plan_for_response = dict(plan)
            plan_for_response.pop("sequence_response_mode", None)
            result["plan"] = plan_for_response
            return result
        except FileNotFoundError as e:
            return {
                "error": f"Configuration or prompt file not found: {str(e)}",
                "errorType": "CONFIGURATION_ERROR",
                "hint": "Ensure config/config.json and planning prompt files exist",
                "source": "bvbrc-mcp-data"
            }
        except Exception as e:
            return {
                "error": f"Advanced data query failed: {str(e)}",
                "errorType": "SEARCH_FAILED",
                "source": "bvbrc-mcp-data"
            }
        finally:
            # Prevent unbounded growth of token registry.
            _clear_download_cancel_token(normalized_cancel_token)

    # @mcp.tool(annotations={"readOnlyHint": True})
    async def bvbrc_get_feature_sequence_by_id(patric_ids: List[str], 
                                              type: str,
                                              token: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the nucleotide or amino acid sequences for genomic features by their PATRIC IDs.
        
        This tool performs a two-step batch query for efficient retrieval of multiple sequences:
        1. Queries genome_feature collection to get the sequence MD5 hashes for all IDs
        2. Queries feature_sequence collection with those MD5s to retrieve the actual sequences
        
        Args:
            patric_ids: List of PATRIC feature IDs (e.g., ["fig|91750.131.peg.1283", "fig|91750.131.peg.1284"])
                       Can be a single ID in a list or multiple IDs for batch retrieval
            type: Type of sequence to retrieve - "na" for nucleotide or "aa" for amino acid
            token: Authentication token (optional, auto-detected if token_provider is configured)
            
        Returns:
            Dict with either:
            - Success: {
                "results": [                      # Array of sequence results
                  {
                    "patric_id": <str>,          # PATRIC feature ID
                    "sequence": <str>,           # The actual DNA/RNA or protein sequence
                    "md5": <str>,                # MD5 hash of the sequence
                    "sequence_type": <str>,      # "na" or "aa"
                    "length": <int>              # Length of sequence
                  },
                  ...
                ],
                "count": <int>,                  # Number of sequences successfully retrieved
                "requested": <int>,              # Number of IDs requested
                "not_found": [<str>, ...],       # Optional: IDs that weren't found
                "warnings": [<str>, ...],        # Optional: Warning messages
                "source": "bvbrc-mcp-data"
              }
            - Error: {
                "error": <error_message>,
                "source": "bvbrc-mcp-data"
              }
        
        Example:
            # Get nucleotide sequence for a single feature
            result = bvbrc_get_feature_sequence_by_id(["fig|91750.131.peg.1283"], "na")
            
            # Get amino acid sequences for multiple features (batch query)
            result = bvbrc_get_feature_sequence_by_id([
                "fig|91750.131.peg.1283",
                "fig|91750.131.peg.1284",
                "fig|91750.131.peg.1285"
            ], "aa")
        """
        print(f"Getting {type.upper()} sequence(s) for {len(patric_ids)} feature(s)")
        
        # Authentication headers
        headers: Optional[Dict[str, str]] = None
        if _token_provider:
            auth_token = _token_provider.get_token(token)
            if auth_token:
                headers = {"Authorization": auth_token}
        elif token:
            headers = {"Authorization": token}
        
        try:
            result = await get_feature_sequence_by_id(
                patric_ids=patric_ids,
                sequence_type=type,
                base_url=_base_url,
                headers=headers
            )
            return result
        except Exception as e:
            return {
                "error": f"Error retrieving sequence: {str(e)}",
                "source": "bvbrc-mcp-data"
            }

    # @mcp.tool(annotations={"readOnlyHint": True})
    async def bvbrc_get_genome_sequence_by_id(genome_ids: List[str],
                                             token: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the nucleotide sequences for complete genomes by their genome IDs.
        
        This tool queries the genome_sequence collection to retrieve full genomic sequences
        including chromosomes and plasmids. Each genome may have multiple sequences.
        
        Args:
            genome_ids: List of genome IDs (e.g., ["208964.12", "511145.12"])
                       Can be a single ID in a list or multiple IDs for batch retrieval
            token: Authentication token (optional, auto-detected if token_provider is configured)
            
        Returns:
            Dict with either:
            - Success: {
                "fasta": <str>,                   # FASTA formatted sequences with headers
                "count": <int>,                   # Number of sequences successfully retrieved
                "requested": <int>,               # Number of IDs requested
                "not_found": [<str>, ...],        # Optional: IDs that weren't found
                "warnings": [<str>, ...],         # Optional: Warning messages
                "source": "bvbrc-mcp-data"
              }
            - Error: {
                "error": <error_message>,
                "source": "bvbrc-mcp-data"
              }
        
        Example FASTA format:
            >NC_002516
            tttaaagagaccggcgattctagtgaaatcgaacgggcaggtc...
            >plasmid_01
            atcgatcgatcgatcg...
        
        Example:
            # Get genome sequence for a single genome
            result = bvbrc_get_genome_sequence_by_id(["208964.12"])
            
            # Get genome sequences for multiple genomes (batch query)
            result = bvbrc_get_genome_sequence_by_id([
                "208964.12",
                "511145.12",
                "83332.12"
            ])
        """
        print(f"Getting genome sequence(s) for {len(genome_ids)} genome(s)")
        
        # Authentication headers
        headers: Optional[Dict[str, str]] = None
        if _token_provider:
            auth_token = _token_provider.get_token(token)
            if auth_token:
                headers = {"Authorization": auth_token}
        elif token:
            headers = {"Authorization": token}
        
        try:
            result = await get_genome_sequence_by_id(
                genome_ids=genome_ids,
                base_url=_base_url,
                headers=headers
            )
            return result
        except Exception as e:
            return {
                "error": f"Error retrieving genome sequence: {str(e)}",
                "source": "bvbrc-mcp-data"
            }

    # -----------------------------------------------------------------
    # Direct Solr query tools (no internal LLM planning)
    #
    # These tools accept pre-structured Solr queries and execute them
    # directly against the BV-BRC Solr API.  They are intended for use
    # by agent systems that produce their own query plans.
    # -----------------------------------------------------------------

    @mcp.tool(annotations={"readOnlyHint": True})
    async def solr_query(
        collection: str,
        query: str = "*:*",
        select: Optional[List[str]] = None,
        sort: Optional[str] = None,
        limit: int = 25,
        count_only: bool = False,
        cursor_id: Optional[str] = None,
        token: Optional[str] = None,
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
            token: Authentication token (optional, auto-detected if token_provider is configured).

        Returns:
            If count_only: {"numFound": int, "source": "bvbrc-mcp-data"}
            Otherwise: {"results": [...], "count": int, "numFound": int,
                        "nextCursorId": str|null, "source": "bvbrc-mcp-data"}
        """
        headers = _build_auth_headers(token)

        # Auto-quote unquoted multi-word values
        safe_query = _auto_quote_query(query or "*:*")

        # Build options dict
        options: Dict[str, Any] = {}
        select_list = normalize_select(select)
        if select_list:
            options["select"] = select_list
        sort_expr = normalize_sort(sort)
        if sort_expr:
            options["sort"] = sort_expr

        # Clamp limit
        if limit < 1:
            limit = 1
        elif limit > 10000:
            limit = 10000

        try:
            result = await query_direct(
                core=collection,
                filter_str=safe_query,
                options=options if options else None,
                base_url=_base_url,
                headers=headers,
                cursorId=cursor_id or "*",
                countOnly=count_only,
                batch_size=limit if not count_only else None,
            )
            result["source"] = "bvbrc-mcp-data"
            return result

        except Exception as e:
            return {
                "error": f"solr_query failed: {type(e).__name__}: {str(e)}",
                "collection": collection,
                "query": safe_query,
                "source": "bvbrc-mcp-data",
            }

    @mcp.tool(annotations={"readOnlyHint": True})
    async def solr_facet_query(
        collection: str,
        query: str = "*:*",
        facet_fields: Optional[List[str]] = None,
        facet_limit: int = 20,
        facet_mincount: int = 1,
        token: Optional[str] = None,
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
            token: Authentication token (optional, auto-detected if token_provider is configured).

        Returns:
            {"numFound": int, "facets": {field: [{"value": str, "count": int}, ...]},
             "source": "bvbrc-mcp-data"}
        """
        headers = _build_auth_headers(token)

        # Auto-quote unquoted multi-word values
        safe_query = _auto_quote_query(query or "*:*")

        if not facet_fields:
            return {
                "error": "solr_facet_query requires at least one facet_field",
                "collection": collection,
                "query": safe_query,
                "source": "bvbrc-mcp-data",
            }

        try:
            result = await query_faceted(
                core=collection,
                filter_str=safe_query,
                facet_fields=facet_fields,
                base_url=_base_url,
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
                "query": safe_query,
                "source": "bvbrc-mcp-data",
            }
