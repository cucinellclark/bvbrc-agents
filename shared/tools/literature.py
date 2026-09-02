"""
Shared literature RAG retrieval tool.

Queries the RAGStack API's retrieval endpoint to find relevant
scientific publication passages for a natural-language query.

The RAGStack service is hosted at www.bv-brc.org/ragstack/asm-next/api
and supports hybrid (vector + BM25) retrieval with optional
knowledge-graph augmentation.

The tool makes a direct HTTP call — it does not import MCP server
functions.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional

import requests


_DEFAULT_LITERATURE_RAG_URL = "https://www.bv-brc.org/ragstack/asm-next/api"
_DEFAULT_TIMEOUT_SECONDS = 45


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
        top_k: Maximum number of source passages to return.
        use_graph: Enable knowledge-graph-augmented retrieval.
        config: Agent configuration.  Reads ``literature_rag_url`` and
                ``literature_rag_timeout_seconds`` if available.
        headers: HTTP headers (expects ``Authorization`` for BV-BRC auth).

    Returns:
        Dict with ``sources`` (list of document chunks with content, score,
        and metadata), ``count``, and ``query``.  On error, returns an
        ``error`` key with a description.
    """
    # Resolve config values
    base_url = _DEFAULT_LITERATURE_RAG_URL
    timeout = _DEFAULT_TIMEOUT_SECONDS

    if config is not None:
        base_url = getattr(config, "literature_rag_url", None) or base_url
        timeout = getattr(config, "literature_rag_timeout_seconds", None) or timeout

    base_url = base_url.rstrip("/")
    retrieve_url = f"{base_url}/v1/retrieve"

    # Build request headers
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
            "sources": [],
            "count": 0,
            "query": query,
            "error": (
                "Literature RAG authentication missing: no Authorization header "
                "or config.bvbrc_auth_token was provided to search_literature."
            ),
            "errorType": "LITERATURE_RAG_AUTH_MISSING",
            "debug": {
                "had_auth_header": False,
                "gateway_url": base_url,
            },
            "source": "literature-rag",
        }

    payload: Dict[str, Any] = {
        "query": query,
        "top_k": top_k,
        "use_graph": use_graph,
    }

    print(
        "search_literature: POST "
        f"url={retrieve_url} auth_len={len(auth_value)} "
        f"top_k={top_k} timeout={timeout} query={query[:80]!r}",
        file=sys.stderr,
    )

    try:
        response = requests.post(
            retrieve_url,
            json=payload,
            headers=req_headers,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()

        sources = data.get("sources", [])

        print(
            f"search_literature: OK status={response.status_code} "
            f"sources={len(sources)}",
            file=sys.stderr,
        )
        return {
            "sources": sources,
            "count": len(sources),
            "query": query,
            "source": "literature-rag",
        }

    except requests.RequestException as e:
        status_code = getattr(getattr(e, "response", None), "status_code", None)
        detail = ""
        if hasattr(e, "response") and e.response is not None:
            try:
                detail = e.response.text[:500]
            except Exception:
                pass
        print(
            "search_literature: REQUEST_ERROR "
            f"status={status_code} err={e!s} detail={detail[:200]!r}",
            file=sys.stderr,
        )
        return {
            "sources": [],
            "count": 0,
            "query": query,
            "error": f"Literature RAG request failed: {str(e)}",
            "errorType": "LITERATURE_RAG_REQUEST_ERROR",
            "status_code": status_code,
            "detail": detail,
            "debug": {
                "had_auth_header": bool(auth_value),
                "gateway_url": base_url,
            },
            "source": "literature-rag",
        }
    except Exception as e:
        print(
            f"search_literature: UNEXPECTED {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return {
            "sources": [],
            "count": 0,
            "query": query,
            "error": f"Literature RAG unexpected error: {type(e).__name__}: {str(e)}",
            "errorType": "LITERATURE_RAG_ERROR",
            "source": "literature-rag",
        }
