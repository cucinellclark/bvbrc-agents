"""
Shared literature RAG retrieval tool.

Queries the RAGStack API's retrieval endpoint to find relevant
scientific publication passages for a natural-language query.

The RAGStack service is hosted at www.bv-brc.org/ragstack/asm-next/api
and supports hybrid (vector + BM25) retrieval with server-side
reranking and optional knowledge-graph augmentation.

RAGStack returns *chunks*. This tool turns them into *papers*:

- Several chunks of one paper (and papers ingested more than once under
  different ``doc_id``s) are grouped by DOI / title, keeping the best
  passages.
- Supplementary files, reviewer comments and similar non-article chunks
  are dropped.
- ASM PDFs with no ``doi`` field get one from their filename
  (``spectrum.00097-22.pdf`` -> ``10.1128/spectrum.00097-22``).
- Only citation fields are kept (no scratch paths or chunk ids), and the
  result bounds itself to ``MAX_RESULT_CHARS`` so the agent loop never
  hard-cuts it.

The tool makes a direct HTTP call — it does not import MCP server
functions.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from typing import Any, Dict, List, Optional

import httpx


_DEFAULT_LITERATURE_RAG_URL = "https://www.bv-brc.org/ragstack/asm-next/api"
_DEFAULT_TIMEOUT_SECONDS = 45
# ASM journals (asm-tok256) plus PubMed Central open-access. Measured
# 2026-10-08 on 16 questions: asm-tok256 alone gave ~35% on-topic papers,
# open-access ~61%; searching both lets the reranker pick across them.
_DEFAULT_COLLECTIONS = ["asm-tok256", "open-access"]

MAX_PAPERS = 20
MAX_PASSAGES_PER_PAPER = 2
MAX_PASSAGE_CHARS = 900
MAX_AUTHORS = 3
# Must stay under TOOL_RESULT_CHAR_LIMITS["search_literature"].
MAX_RESULT_CHARS = 18_000

_ASM_DOI_FILENAME = re.compile(
    r"^((?:aac|aem|asmcr|cmr|ecosalplus|genomea|iai|jb|jcm|jmbe|jvi|mbio|mcb|"
    r"microbiolspec|mmbr|mra|msphere|msystems|spectrum)\.[0-9a-z][0-9a-z.\-]*)\.pdf$",
    re.IGNORECASE,
)
# Supplementary material, reviewer reports, tables of contents.
_NON_ARTICLE_FILENAME = re.compile(
    r"(-s\d+\.pdf$|_supp|supplement|reviewer|table-of-contents|^z[a-z]{2}\d+s[o0-9]*\.pdf$)",
    re.IGNORECASE,
)
_NON_ARTICLE_TITLE = re.compile(r"^(SM-[A-Z0-9]+ |reviewer|table of contents)", re.IGNORECASE)

USAGE_NOTE = (
    "Cite only papers listed here, by title and DOI (or PMID). List authors only "
    "where 'authors' is given; never add authors, years or journals that are not in "
    "this result. If these passages do not answer the question, say so and suggest a "
    "narrower or different search instead of filling the gap from memory."
)


def _parse_list(value: Any) -> List[str]:
    """Authors arrive as a list or as the string repr of one."""
    if isinstance(value, list):
        return [str(v) for v in value if v]
    if isinstance(value, str) and value.startswith("["):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if v]
        except (ValueError, SyntaxError):
            return []
    if isinstance(value, str) and value and value != "None":
        return [value]
    return []


def _clean(value: Any) -> Any:
    """RAGStack stringifies missing values as 'None' on some collections."""
    if value in (None, "", "None", "null"):
        return None
    return value


def _derive_doi(meta: Dict[str, Any]) -> Optional[str]:
    doi = _clean(meta.get("doi"))
    if doi:
        return str(doi).lower()
    m = _ASM_DOI_FILENAME.match(str(meta.get("filename") or ""))
    if m:
        return f"10.1128/{m.group(1).lower()}"
    return None


def _usable_title(title: Any) -> Optional[str]:
    title = _clean(title)
    if not title:
        return None
    title = " ".join(str(title).split())
    # Filenames and accession-like strings are not titles.
    if " " not in title or title.lower().endswith(".pdf"):
        return None
    return title


def _is_non_article(meta: Dict[str, Any]) -> bool:
    filename = str(meta.get("filename") or "")
    title = str(meta.get("title") or "")
    return bool(_NON_ARTICLE_FILENAME.search(filename) or _NON_ARTICLE_TITLE.match(title))


def _format_authors(authors: List[str]) -> Optional[str]:
    if not authors:
        return None
    shown = "; ".join(authors[:MAX_AUTHORS])
    return shown + ("; et al." if len(authors) > MAX_AUTHORS else "")


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + " …"


def group_sources(sources: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    """Group RAGStack chunks into at most *top_k* distinct papers.

    Returns ``{"papers": [...], "duplicate_chunks": n, "dropped_non_article": n}``.
    Papers keep the order of their best-scoring chunk.
    """
    papers: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    duplicates = 0
    dropped = 0

    for src in sources:
        meta = src.get("metadata") or {}
        if _is_non_article(meta):
            dropped += 1
            continue
        doi = _derive_doi(meta)
        title = _usable_title(meta.get("title"))
        key = doi or (title.lower() if title else None) or src.get("doc_id")
        if not key:
            dropped += 1
            continue

        passage = _clip(src.get("content") or "", MAX_PASSAGE_CHARS)
        if key in papers:
            paper = papers[key]
            if passage and passage not in paper["passages"] and len(paper["passages"]) < MAX_PASSAGES_PER_PAPER:
                paper["passages"].append(passage)
            else:
                duplicates += 1
            continue

        pmid = _clean(meta.get("pmid"))
        paper = {
            "title": title,
            "authors": _format_authors(_parse_list(meta.get("authors"))),
            "year": _clean(meta.get("year")),
            "journal": _clean(meta.get("journal")),
            "doi": doi,
            "url": f"https://doi.org/{doi}" if doi else (
                f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None),
            "pmid": pmid,
            "pmcid": _clean(meta.get("pmcid")),
            # open-access chunks always carry 0 (field not populated), which
            # would read as "uncited"; keep only real counts.
            "n_citations": _clean(meta.get("n_citations")) if str(meta.get("n_citations")) not in ("0", "0.0") else None,
            "collection": _clean(src.get("collection")),
            "score": round(float(src["score"]), 3) if isinstance(src.get("score"), (int, float)) else None,
            "passages": [passage] if passage else [],
        }
        papers[key] = paper
        order.append(key)

    selected = [papers[k] for k in order[:top_k]]
    for i, paper in enumerate(selected, 1):
        paper["ref"] = i
    # Drop empty fields so the LLM sees only what exists.
    selected = [{k: v for k, v in p.items() if v not in (None, [], "")} for p in selected]
    return {"papers": selected, "duplicate_chunks": duplicates, "dropped_non_article": dropped}


def _fit(result: Dict[str, Any]) -> Dict[str, Any]:
    """Shrink *result* until its JSON fits MAX_RESULT_CHARS.

    Second passages go first, then the lowest-ranked papers.
    """
    def size() -> int:
        return len(json.dumps(result, indent=2, default=str))

    papers = result["papers"]
    if size() <= MAX_RESULT_CHARS:
        return result
    for paper in reversed(papers):
        if len(paper.get("passages", [])) > 1:
            paper["passages"] = paper["passages"][:1]
            if size() <= MAX_RESULT_CHARS:
                return result
    while len(papers) > 1 and size() > MAX_RESULT_CHARS:
        papers.pop()
        result["truncated_papers"] = result.get("truncated_papers", 0) + 1
    result["count"] = len(papers)
    return result


async def search_literature(
    query: str,
    top_k: int = 10,
    use_graph: bool = False,
    config: Any = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Search scientific literature via the literature RAG service.

    Args:
        query: Natural-language search query (organism, gene, topic, etc.).
        top_k: Maximum number of distinct papers to return (1-20).
        use_graph: Enable knowledge-graph-augmented retrieval.
        config: Agent configuration.  Reads ``literature_rag_url``,
                ``literature_rag_timeout_seconds`` and
                ``literature_rag_collections`` if available.
        headers: HTTP headers (expects ``Authorization`` for BV-BRC auth).

    Returns:
        Dict with ``papers`` (title, authors, year, journal, doi, url,
        passages, ...), ``count``, ``query``, ``note`` and ``_summary``.
        On error, returns an ``error`` key with a description.
    """
    base_url = _DEFAULT_LITERATURE_RAG_URL
    timeout = _DEFAULT_TIMEOUT_SECONDS
    collections = list(_DEFAULT_COLLECTIONS)

    if config is not None:
        base_url = getattr(config, "literature_rag_url", None) or base_url
        timeout = getattr(config, "literature_rag_timeout_seconds", None) or timeout
        collections = getattr(config, "literature_rag_collections", None) or collections

    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        top_k = 10
    top_k = max(1, min(top_k, MAX_PAPERS))

    base_url = base_url.rstrip("/")
    retrieve_url = f"{base_url}/v1/retrieve"

    req_headers: Dict[str, str] = {"Content-Type": "application/json"}

    auth_value: str | None = None
    if headers:
        # HTTP header names are case-insensitive; support both common variants.
        auth_value = headers.get("Authorization") or headers.get("authorization")

    if not auth_value and config is not None:
        # Some call paths inject the token via config (e.g. orchestrator default token),
        # not via headers.
        auth_value = getattr(config, "bvbrc_auth_token", None)

    if auth_value:
        # RAGStack accepts Authorization (BearerIdentity) or X-API-Key.
        # Send the BV-BRC auth token in the Authorization header.  If it
        # already has a "Bearer " prefix, keep it; otherwise add one.
        auth_value = auth_value.strip()
        if not auth_value.lower().startswith("bearer "):
            auth_value = f"Bearer {auth_value}"
        req_headers["Authorization"] = auth_value
    else:
        print(
            "search_literature: AUTH_MISSING "
            f"url={retrieve_url} query={query[:80]!r}",
            file=sys.stderr,
        )
        return {
            "papers": [],
            "count": 0,
            "query": query,
            "error": (
                "Literature RAG authentication missing: no Authorization header "
                "or config.bvbrc_auth_token was provided to search_literature."
            ),
            "errorType": "LITERATURE_RAG_AUTH_MISSING",
            "source": "literature-rag",
        }

    # Over-fetch chunks: several chunks per paper and duplicate ingests
    # collapse into one paper each (~2.4 of every 10 chunks, measured).
    payload: Dict[str, Any] = {
        "query": query,
        "top_k": min(top_k * 3, 50),
        "use_graph": use_graph,
    }
    if collections:
        payload["collections"] = collections

    print(
        "search_literature: POST "
        f"url={retrieve_url} top_k={top_k} chunks={payload['top_k']} "
        f"collections={collections} timeout={timeout} query={query[:80]!r}",
        file=sys.stderr,
    )

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(retrieve_url, json=payload, headers=req_headers)
            if response.status_code == 404 and "collections" in payload:
                # A collection was removed or is not readable by this user;
                # RAGStack refuses the whole request. Fall back to the
                # caller's default collection rather than failing.
                print(
                    "search_literature: collections refused (404), retrying with default",
                    file=sys.stderr,
                )
                payload.pop("collections")
                response = await client.post(retrieve_url, json=payload, headers=req_headers)
            response.raise_for_status()
            data = response.json()

        sources = data.get("sources", []) or []
        grouped = group_sources(sources, top_k)
        papers = grouped["papers"]

        print(
            f"search_literature: OK status={response.status_code} "
            f"chunks={len(sources)} papers={len(papers)} "
            f"duplicate_chunks={grouped['duplicate_chunks']} "
            f"dropped_non_article={grouped['dropped_non_article']}",
            file=sys.stderr,
        )
        result: Dict[str, Any] = {
            "query": query,
            "count": len(papers),
            "papers": papers,
            "note": USAGE_NOTE if papers else (
                "No papers matched. Tell the user nothing was found in the indexed "
                "literature; do not cite papers from memory as search results."
            ),
            "source": "literature-rag",
        }
        result = _fit(result)
        # Kept by trim_messages_to_fit when older tool results are compressed,
        # so a trimmed search can still be cited correctly.
        result["_summary"] = {
            "query": query,
            "papers": [
                {"ref": p.get("ref"), "title": p.get("title"), "doi": p.get("doi"), "year": p.get("year")}
                for p in result["papers"]
            ],
        }
        return result

    except httpx.HTTPStatusError as e:
        status_code = e.response.status_code
        detail = e.response.text[:500]
        print(
            "search_literature: REQUEST_ERROR "
            f"status={status_code} detail={detail[:200]!r}",
            file=sys.stderr,
        )
        return {
            "papers": [],
            "count": 0,
            "query": query,
            "error": f"Literature RAG request failed: HTTP {status_code}",
            "errorType": "LITERATURE_RAG_REQUEST_ERROR",
            "status_code": status_code,
            "detail": detail,
            "source": "literature-rag",
        }
    except httpx.HTTPError as e:
        print(
            f"search_literature: REQUEST_ERROR {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return {
            "papers": [],
            "count": 0,
            "query": query,
            "error": f"Literature RAG request failed: {type(e).__name__}: {e}",
            "errorType": "LITERATURE_RAG_REQUEST_ERROR",
            "source": "literature-rag",
        }
    except Exception as e:
        print(
            f"search_literature: UNEXPECTED {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return {
            "papers": [],
            "count": 0,
            "query": query,
            "error": f"Literature RAG unexpected error: {type(e).__name__}: {str(e)}",
            "errorType": "LITERATURE_RAG_ERROR",
            "source": "literature-rag",
        }
