"""
URL construction utilities for BV-BRC web links and downloads.

Generates viewer URLs, download URLs, and entity page URLs from query
results so agents can embed actionable markdown links in their responses.

URLs contain raw (unencoded) parentheses because the BV-BRC RQL parser
does not URL-decode before parsing.  Markdown-safety is handled at the
prompt layer: agents are instructed to use angle-bracket link syntax
``[text](<url>)`` which allows raw parentheses inside URLs (CommonMark).
"""

from __future__ import annotations

import re
from typing import Optional
from urllib.parse import quote

# ── BV-BRC base URLs ──────────────────────────────────────────────────

BVBRC_BASE_URL = "https://www.bv-brc.org"
BVBRC_API_URL = "https://www.bv-brc.org/api"
BVBRC_API_BULK_URL = "https://www.bv-brc.org/api-bulk"

# ── Collection → list viewer mapping ─────────────────────────────────
# Matches the COLLECTION_VIEWER_MAP in the gateway's orchestratorClient.js.

COLLECTION_VIEWER_MAP = {
    "genome": "GenomeList",
    "taxonomy": "TaxonList",
    "genome_feature": "FeatureList",
    "pathway": "PathwayList",
    "protein_structure": "ProteinStructureList",
    "strain": "StrainList",
    "surveillance": "SurveillanceList",
    "subsystem": "SubsystemList",
    "serology": "SerologyList",
    "epitope": "EpitopeList",
    "sp_gene": "SpecialtyGeneList",
    "sp_gene_ref": "SpecialtyVFGeneList",
    "protein_feature": "DomainsAndMotifsList",
}

# ── Collection → unique ID field ─────────────────────────────────────

COLLECTION_ID_FIELD = {
    "genome": "genome_id",
    "genome_feature": "feature_id",
    "taxonomy": "taxon_id",
    "epitope": "epitope_id",
    "protein_structure": "pdb_id",
    "genome_amr": "id",
    "genome_sequence": "sequence_id",
    "sp_gene": "id",
    "pathway": "id",
    "subsystem": "id",
}

# ── Entity → single-item viewer ─────────────────────────────────────

ENTITY_VIEWER_MAP = {
    "genome": "Genome",
    "genome_feature": "Feature",
    "taxonomy": "Taxonomy",
    "epitope": "Epitope",
}

# Collections that support direct FASTA downloads (query fields match the
# collection being downloaded from).  The ``genome`` collection is NOT
# included because its query fields (genus, species, etc.) don't exist on
# ``genome_sequence`` — a FASTA download would require a two-step ID lookup.
FASTA_COLLECTIONS = {"genome_feature", "genome_sequence"}


# ── Solr-to-RQL converter ────────────────────────────────────────────
# Pure string transformation — converts Solr/Lucene query syntax to
# BV-BRC RQL format used by viewer URLs and the data API.
# This lived in data_agent/models.py but is needed by search_data()
# in shared/tools/data.py, so it lives here to avoid circular imports.


def _escape_rql_value(value: str) -> str:
    """Escape characters that are meaningful in RQL function argument lists."""
    return value.replace("\\", "\\\\").replace(",", "\\,").replace(")", "\\)")


def _solr_term_to_rql(term: str) -> str:
    """Convert a single Solr ``field:value`` term to an RQL expression."""
    if term in ("*:*", "*"):
        return "eq(*,*)"

    colon_idx = term.find(":")
    if colon_idx <= 0:
        return f"keyword({_escape_rql_value(term)})"

    field = term[:colon_idx].strip()
    raw_value = term[colon_idx + 1 :].strip()

    if not raw_value:
        return f"keyword({_escape_rql_value(field)})"

    # Range query: field:[min TO max]
    range_match = re.match(r"^[\[\{]\s*(.+?)\s+TO\s+(.+?)\s*[\]\}]$", raw_value)
    if range_match:
        low, high = range_match.group(1), range_match.group(2)
        return f"between({field},{_escape_rql_value(low.strip(chr(34)))},{_escape_rql_value(high.strip(chr(34)))})"

    # Grouped OR values: field:(val1 OR val2)
    if raw_value.startswith("(") and raw_value.endswith(")"):
        inner = raw_value[1:-1].strip()
        parts = re.split(r"\s+OR\s+", inner)
        if len(parts) > 1:
            eq_parts = [f"eq({field},{_escape_rql_value(p.strip().strip(chr(34)))})" for p in parts]
            return "or(" + ",".join(eq_parts) + ")"
        val = inner.strip('"')
        return f"eq({field},{_escape_rql_value(val)})"

    # Quoted value
    if raw_value.startswith('"') and raw_value.endswith('"'):
        val = raw_value[1:-1]
        return f"eq({field},{_escape_rql_value(val)})"

    return f"eq({field},{_escape_rql_value(raw_value)})"


def _tokenize_solr_query(query: str) -> list[str]:
    """Split a Solr query into tokens preserving quoted strings and brackets."""
    tokens: list[str] = []
    i = 0
    n = len(query)

    while i < n:
        if query[i].isspace():
            i += 1
            continue

        if query[i] == "(" and (i == 0 or query[i - 1] != ":"):
            tokens.append("(")
            i += 1
            continue
        if query[i] == ")":
            tokens.append(")")
            i += 1
            continue

        for kw in ("AND", "OR", "NOT"):
            if (
                query[i : i + len(kw)] == kw
                and (i + len(kw) >= n or not query[i + len(kw)].isalnum())
                and (i == 0 or not query[i - 1].isalnum())
            ):
                tokens.append(kw)
                i += len(kw)
                break
        else:
            term_start = i
            while i < n and not query[i].isspace():
                if query[i] == '"':
                    i += 1
                    while i < n and query[i] != '"':
                        i += 1
                    if i < n:
                        i += 1
                elif query[i] == "(" and i > term_start and query[i - 1] == ":":
                    depth = 1
                    i += 1
                    while i < n and depth > 0:
                        if query[i] == "(":
                            depth += 1
                        elif query[i] == ")":
                            depth -= 1
                        elif query[i] == '"':
                            i += 1
                            while i < n and query[i] != '"':
                                i += 1
                        i += 1
                elif query[i] in ("[", "{"):
                    close_char = "]" if query[i] == "[" else "}"
                    i += 1
                    while i < n and query[i] != close_char:
                        i += 1
                    if i < n:
                        i += 1
                elif query[i] == ")":
                    break
                else:
                    i += 1

            token = query[term_start:i].strip()
            if token:
                tokens.append(token)

    return tokens


def _parse_solr_expr(tokens: list[str], pos: int = 0) -> tuple[str, int]:
    """Recursive-descent parser for Solr boolean expressions."""
    left, pos = _parse_solr_unary(tokens, pos)

    while pos < len(tokens) and tokens[pos] in ("AND", "OR"):
        op = tokens[pos]
        pos += 1
        parts = [left]
        rql_op = "and" if op == "AND" else "or"
        right, pos = _parse_solr_unary(tokens, pos)
        parts.append(right)
        while pos < len(tokens) and tokens[pos] == op:
            pos += 1
            next_part, pos = _parse_solr_unary(tokens, pos)
            parts.append(next_part)
        left = f"{rql_op}(" + ",".join(parts) + ")"

    return left, pos


def _parse_solr_unary(tokens: list[str], pos: int) -> tuple[str, int]:
    """Parse NOT prefix and parenthesized groups."""
    if pos >= len(tokens):
        return "eq(*,*)", pos
    if tokens[pos] == "NOT":
        pos += 1
        inner, pos = _parse_solr_unary(tokens, pos)
        return f"not({inner})", pos
    if tokens[pos] == "(":
        pos += 1
        inner, pos = _parse_solr_expr(tokens, pos)
        if pos < len(tokens) and tokens[pos] == ")":
            pos += 1
        return inner, pos
    term = tokens[pos]
    pos += 1
    return _solr_term_to_rql(term), pos


def solr_to_rql(solr_query: str) -> str:
    """Convert a Solr/Lucene query string to BV-BRC RQL format.

    Handles the subset of Solr syntax produced by the data agent LLM:
      - ``field:value``, ``field:"multi word"``, ``field:(v1 OR v2)``
      - ``AND``, ``OR``, ``NOT`` boolean operators
      - Parenthesized grouping
      - Range queries ``field:[min TO max]``
      - Wildcards ``field:val*``
    """
    if not solr_query or not solr_query.strip():
        return "eq(*,*)"
    solr_query = solr_query.strip()
    if solr_query in ("*:*", "*"):
        return "eq(*,*)"
    tokens = _tokenize_solr_query(solr_query)
    if not tokens:
        return "eq(*,*)"
    rql, _ = _parse_solr_expr(tokens, 0)
    return rql


# ── Markdown-safe URL encoding ───────────────────────────────────────


def _encode_rql_for_markdown(rql_query: str) -> str:
    """Return the RQL query as-is (no encoding).

    Previously this percent-encoded parentheses to prevent LLMs from
    confusing RQL closing parens with markdown link closing parens.
    However, the BV-BRC frontend sends the URL query string as a raw
    POST body to the data API (content-type: application/rqlquery+
    x-www-form-urlencoded). The API's RQL parser treats the body as
    literal text — it does NOT URL-decode first — so percent-encoded
    parentheses produce parse failures and empty grids. The BV-BRC
    site itself uses unencoded parentheses in all viewer URLs.

    The markdown-safety problem is now solved at the prompt layer:
    ``response_format_skill.py`` instructs agents to use angle-bracket
    link syntax ``[text](<url>)`` which allows raw parentheses inside
    the URL without confusing the markdown parser (CommonMark spec).
    """
    return rql_query


# ── List viewer URL ──────────────────────────────────────────────────


def build_viewer_url(
    collection: str,
    rql_query: str,
    *,
    base_url: str = BVBRC_BASE_URL,
) -> Optional[str]:
    """Build a BV-BRC list viewer URL for a collection + RQL query.

    Returns ``None`` if *collection* or *rql_query* is empty.

    Examples::

        >>> build_viewer_url("genome", "eq(genus,Mycobacterium)")
        'https://www.bv-brc.org/view/GenomeList/?eq%28genus,Mycobacterium%29'
    """
    if not collection or not rql_query:
        return None

    encoded_rql = _encode_rql_for_markdown(rql_query)
    viewer = COLLECTION_VIEWER_MAP.get(collection.lower())

    if viewer:
        return f"{base_url}/view/{viewer}/?{encoded_rql}"
    # Fallback for unknown collections
    return f"{base_url}/search/?{encoded_rql}"


# ── Entity page URL ──────────────────────────────────────────────────


def build_entity_url(
    collection: str,
    entity_id: str,
    *,
    base_url: str = BVBRC_BASE_URL,
) -> Optional[str]:
    """Build a BV-BRC entity page URL (single genome, feature, etc.).

    Examples::

        >>> build_entity_url("genome", "83332.12")
        'https://www.bv-brc.org/view/Genome/83332.12'
    """
    if not collection or not entity_id:
        return None

    viewer = ENTITY_VIEWER_MAP.get(collection.lower())
    if viewer:
        safe_id = quote(str(entity_id), safe=".")
        return f"{base_url}/view/{viewer}/{safe_id}"
    return None


# ── Download URLs ────────────────────────────────────────────────────


def build_tsv_download_url(
    collection: str,
    rql_query: str,
    *,
    limit: int = 25000,
    api_url: str = BVBRC_API_BULK_URL,
) -> Optional[str]:
    """Build a direct TSV download URL for a collection + RQL query.

    Uses the ``api-bulk`` endpoint which supports large downloads.
    The URL triggers a browser download when opened (``http_download=true``).
    A ``sort()`` clause on the collection's primary key is required.

    Examples::

        >>> build_tsv_download_url("genome", "eq(genus,Salmonella)")
        'https://www.bv-brc.org/api-bulk/genome/?eq%28genus,Salmonella%29&sort%28+genome_id%29&limit%2825000%29&http_accept=text/tsv&http_download=true'
    """
    if not collection or not rql_query:
        return None

    col_lower = collection.lower()
    sort_field = COLLECTION_ID_FIELD.get(col_lower, "id")
    rql = _encode_rql_for_markdown(rql_query)
    return (
        f"{api_url}/{col_lower}/?"
        f"{rql}&sort%28+{sort_field}%29&limit%28{limit}%29"
        f"&http_accept=text/tsv&http_download=true"
    )


def build_fasta_download_url(
    collection: str,
    rql_query: str,
    *,
    fasta_type: str = "dna",
    limit: int = 100000,
    api_url: str = BVBRC_API_BULK_URL,
) -> Optional[str]:
    """Build a FASTA download URL.

    Uses the ``api-bulk`` endpoint which supports large downloads.
    For ``genome_feature``: downloads feature sequences (DNA or protein).
    For ``genome``: downloads whole genome sequences via ``genome_sequence``.

    Args:
        collection: Source collection name.
        rql_query: RQL query string.
        fasta_type: ``"dna"`` or ``"protein"``.
        limit: Max records (default 100,000).
        api_url: BV-BRC API bulk endpoint URL.

    Returns:
        Download URL string, or ``None`` if the collection doesn't support FASTA.
    """
    col_lower = collection.lower() if collection else ""

    if col_lower == "genome_feature":
        accept = (
            "application/protein+fasta" if fasta_type == "protein"
            else "application/dna+fasta"
        )
        rql = _encode_rql_for_markdown(rql_query)
        return (
            f"{api_url}/genome_feature/?"
            f"{rql}&sort%28+feature_id%29&limit%28{limit}%29"
            f"&http_accept={accept}&http_download=true"
        )

    if col_lower == "genome_sequence":
        rql = _encode_rql_for_markdown(rql_query)
        return (
            f"{api_url}/genome_sequence/?"
            f"{rql}&sort%28+sequence_id%29&limit%28{limit}%29"
            f"&http_accept=application/dna+fasta&http_download=true"
        )

    # The ``genome`` collection is NOT supported here because its query
    # fields (genus, species, host_name, etc.) don't exist on the
    # ``genome_sequence`` collection.  A FASTA download for genomes would
    # require first fetching genome_ids, then querying genome_sequence —
    # that can't be expressed as a single download URL.

    return None


# ── Workspace URLs ───────────────────────────────────────────────────


def build_workspace_url(
    workspace_path: str,
    *,
    base_url: str = BVBRC_BASE_URL,
) -> Optional[str]:
    """Build a BV-BRC workspace browser URL for a workspace path.

    Examples::

        >>> build_workspace_url("/user@bv-brc.org/home/MyData")
        'https://www.bv-brc.org/workspace/user@bv-brc.org/home/MyData'
    """
    if not workspace_path:
        return None

    # Strip leading slash for URL construction (workspace route adds it)
    clean_path = workspace_path.lstrip("/")
    return f"{base_url}/workspace/{clean_path}"


# ── Service / App page URLs ──────────────────────────────────────────


def build_service_url(
    app_name: str,
    *,
    base_url: str = BVBRC_BASE_URL,
) -> Optional[str]:
    """Build a URL to a BV-BRC service/application page.

    Examples::

        >>> build_service_url("Assembly2")
        'https://www.bv-brc.org/app/Assembly2'
    """
    if not app_name:
        return None
    return f"{base_url}/app/{app_name}"


# ── Convenience: attach URLs to a search_data result ─────────────────


def enrich_search_result(
    result: dict,
    collection: str,
    rql_query: Optional[str],
) -> dict:
    """Add actionable URL fields to a ``search_data`` tool result dict.

    Mutates *result* in place and returns it. Adds:
    - ``viewer_url``: Link to the BV-BRC list viewer for this query
    - ``download_tsv_url``: Direct TSV download link
    - ``download_fasta_dna_url``: DNA FASTA download (if applicable)
    - ``download_fasta_protein_url``: Protein FASTA download (if applicable)
    """
    if not rql_query or result.get("error"):
        return result

    viewer_url = build_viewer_url(collection, rql_query)
    if viewer_url:
        result["viewer_url"] = viewer_url

    tsv_url = build_tsv_download_url(collection, rql_query)
    if tsv_url:
        result["download_tsv_url"] = tsv_url

    col_lower = collection.lower() if collection else ""

    # FASTA downloads (only for collections whose query fields match the
    # download collection — genome_feature and genome_sequence, NOT genome)
    if col_lower in FASTA_COLLECTIONS:
        dna_url = build_fasta_download_url(collection, rql_query, fasta_type="dna")
        if dna_url:
            result["download_fasta_dna_url"] = dna_url

    if col_lower == "genome_feature":
        protein_url = build_fasta_download_url(
            collection, rql_query, fasta_type="protein"
        )
        if protein_url:
            result["download_fasta_protein_url"] = protein_url

    return result
