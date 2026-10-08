"""Tests for shared/tools/literature.py (search_literature).

RAGStack is replaced by an httpx.MockTransport, so no network is needed.
Chunk shapes mirror real /v1/retrieve responses captured 2026-10-08
(stringified author lists, "None" strings, ASM filenames without a doi).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from shared.tools import literature  # noqa: E402
from shared.tools import result_char_limit, truncate_result  # noqa: E402
from shared.tools.literature import group_sources, search_literature  # noqa: E402


def _chunk(doc_id, content, score=0.9, collection="asm-tok256", **meta):
    return {
        "doc_id": doc_id,
        "chunk_id": f"{doc_id}-c",
        "content": content,
        "score": score,
        "collection": collection,
        "metadata": {
            "source_path": f"/local/scratch/{doc_id}/x.pdf",
            "tenant_id": "public",
            "prev_chunk_id": "None",
            **meta,
        },
    }


KPC = dict(filename="jcm.00015-07.pdf", doi="10.1128/jcm.00015-07",
           title="Evaluation of Methods To Identify the Klebsiella pneumoniae Carbapenemase", year="2007")


@pytest.fixture
def mock_ragstack(monkeypatch):
    """Route search_literature's httpx client to a handler; record requests."""
    calls = []
    state = {"handler": None}

    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        def handler(request):
            body = json.loads(request.content)
            calls.append({"url": str(request.url), "body": body, "headers": dict(request.headers)})
            return state["handler"](body)
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(literature.httpx, "AsyncClient", factory)
    return state, calls


CFG = SimpleNamespace(bvbrc_auth_token="un=alice@patricbrc.org|sig=x")


# ---------------------------------------------------------------- grouping


def test_duplicate_ingests_collapse_into_one_paper():
    # The same paper ingested three times under different doc_ids (seen live).
    src = [_chunk(f"d{i}", "rug-resistant Enterobacteriaceae ... KPC", **KPC) for i in range(3)]
    src.append(_chunk("d9", "another passage of the same paper", **KPC))
    out = group_sources(src, top_k=10)
    assert len(out["papers"]) == 1
    paper = out["papers"][0]
    assert paper["passages"] == ["rug-resistant Enterobacteriaceae ... KPC", "another passage of the same paper"]
    assert out["duplicate_chunks"] == 2
    assert paper["ref"] == 1


def test_passages_capped_per_paper():
    src = [_chunk(f"d{i}", f"passage {i}", **KPC) for i in range(5)]
    paper = group_sources(src, 10)["papers"][0]
    assert len(paper["passages"]) == literature.MAX_PASSAGES_PER_PAPER


def test_doi_derived_from_asm_filename():
    src = [
        _chunk("a", "x", filename="spectrum.00097-22.pdf", title="Potential Cross-Transmission of M. abscessus"),
        _chunk("b", "y", filename="aac.42.7.1853.pdf", title="aac.42.7.1853.pdf"),
        _chunk("c", "z", filename="PMC5860732", title="Some open access paper", doi="None"),
    ]
    papers = group_sources(src, 10)["papers"]
    assert papers[0]["doi"] == "10.1128/spectrum.00097-22"
    assert papers[0]["url"] == "https://doi.org/10.1128/spectrum.00097-22"
    assert papers[1]["doi"] == "10.1128/aac.42.7.1853"
    assert "title" not in papers[1]  # a filename is not a title
    assert "doi" not in papers[2]  # "None" string is missing, not a DOI


def test_non_article_chunks_dropped():
    src = [
        _chunk("s1", "supp", filename="spectrum.02918-23-s0002.pdf", title="spectrum.02918-23-s0002.pdf"),
        _chunk("s2", "supp", filename="x.pdf", title="SM-MSYS210341 1..17"),
        _chunk("s3", "rev", filename="reviewer-comments.pdf", title="reviewer-comments.pdf"),
        _chunk("s4", "toc", filename="table-of-contents.pdf"),
        _chunk("ok", "real", **KPC),
    ]
    out = group_sources(src, 10)
    assert [p["doi"] for p in out["papers"]] == ["10.1128/jcm.00015-07"]
    assert out["dropped_non_article"] == 4


def test_authors_parsed_and_shortened():
    src = [
        _chunk("a", "x", title="Paper A title here", doi="10.1/a",
               authors="['Ding, Li', 'Shen, Siquan', 'Chen, Jing', 'Tian, Zhen']"),
        _chunk("b", "y", title="Paper B title here", doi="10.1/b", authors=["Pitout, J", "Nordmann, P"]),
        _chunk("c", "z", title="Paper C title here", doi="10.1/c", authors=None),
    ]
    papers = group_sources(src, 10)["papers"]
    assert papers[0]["authors"] == "Ding, Li; Shen, Siquan; Chen, Jing; et al."
    assert papers[1]["authors"] == "Pitout, J; Nordmann, P"
    assert "authors" not in papers[2]


def test_zero_citation_count_omitted():
    src = [_chunk("a", "x", title="Open access paper", doi="10.1/a", n_citations="0", collection="open-access"),
           _chunk("b", "y", title="ASM paper title", doi="10.1/b", n_citations="137")]
    papers = group_sources(src, 10)["papers"]
    assert "n_citations" not in papers[0]
    assert papers[1]["n_citations"] == "137"


def test_internal_fields_never_reach_the_llm():
    paper = group_sources([_chunk("a", "x", **KPC)], 10)["papers"][0]
    blob = json.dumps(paper)
    for leaked in ("source_path", "/local/scratch", "chunk_id", "tenant_id", "prev_chunk_id"):
        assert leaked not in blob


def test_top_k_counts_papers_not_chunks():
    src = []
    for i in range(12):
        src += [_chunk(f"p{i}a", f"a{i}", title=f"Paper number {i}", doi=f"10.1/{i}"),
                _chunk(f"p{i}b", f"b{i}", title=f"Paper number {i}", doi=f"10.1/{i}")]
    papers = group_sources(src, top_k=5)["papers"]
    assert [p["doi"] for p in papers] == [f"10.1/{i}" for i in range(5)]


# ---------------------------------------------------------------- request


async def test_request_searches_both_collections_and_overfetches(mock_ragstack):
    state, calls = mock_ragstack
    state["handler"] = lambda body: httpx.Response(200, json={"sources": [_chunk("a", "x", **KPC)]})
    out = await search_literature("KPC", top_k=10, config=CFG)
    body = calls[0]["body"]
    assert body["collections"] == ["asm-tok256", "open-access"]
    assert body["top_k"] == 30
    assert calls[0]["headers"]["authorization"].startswith("Bearer ")
    assert calls[0]["url"].endswith("/v1/retrieve")
    assert out["count"] == 1 and out["papers"][0]["doi"] == "10.1128/jcm.00015-07"
    assert out["_summary"]["papers"][0]["doi"] == "10.1128/jcm.00015-07"
    assert "Cite only papers listed here" in out["note"]


async def test_top_k_clamped(mock_ragstack):
    state, calls = mock_ragstack
    state["handler"] = lambda body: httpx.Response(200, json={"sources": []})
    await search_literature("q", top_k=500, config=CFG)
    await search_literature("q", top_k="bogus", config=CFG)
    assert calls[0]["body"]["top_k"] == 50  # 20 papers * 3, capped
    assert calls[1]["body"]["top_k"] == 30  # default 10


async def test_config_collections_override(mock_ragstack):
    state, calls = mock_ragstack
    state["handler"] = lambda body: httpx.Response(200, json={"sources": []})
    cfg = SimpleNamespace(bvbrc_auth_token="tok", literature_rag_collections=["open-access"])
    await search_literature("q", config=cfg)
    assert calls[0]["body"]["collections"] == ["open-access"]


async def test_unreadable_collection_falls_back_to_default(mock_ragstack):
    state, calls = mock_ragstack

    def handler(body):
        if "collections" in body:
            return httpx.Response(404, json={"detail": "not found"})
        return httpx.Response(200, json={"sources": [_chunk("a", "x", **KPC)]})

    state["handler"] = handler
    out = await search_literature("KPC", config=CFG)
    assert len(calls) == 2 and "collections" not in calls[1]["body"]
    assert out["count"] == 1


async def test_empty_result_tells_llm_not_to_cite_from_memory(mock_ragstack):
    state, _ = mock_ragstack
    state["handler"] = lambda body: httpx.Response(200, json={"sources": []})
    out = await search_literature("nothing", config=CFG)
    assert out["count"] == 0 and "from memory" in out["note"]


async def test_http_error_is_structured(mock_ragstack):
    state, _ = mock_ragstack
    state["handler"] = lambda body: httpx.Response(500, text="boom")
    out = await search_literature("q", config=CFG)
    assert out["errorType"] == "LITERATURE_RAG_REQUEST_ERROR"
    assert out["status_code"] == 500 and out["papers"] == []


async def test_missing_auth():
    out = await search_literature("q", config=SimpleNamespace())
    assert out["errorType"] == "LITERATURE_RAG_AUTH_MISSING"


# ---------------------------------------------------------------- size


async def test_large_result_fits_loop_cap_without_hard_cut(mock_ragstack):
    """20 papers with long passages must survive truncate_result intact
    (before this change, results were hard-cut at 8000 chars and the LLM
    saw ~4 of 10 sources)."""
    state, _ = mock_ragstack
    long = "word " * 2000
    sources = []
    for i in range(20):
        for j in range(3):
            sources.append(_chunk(f"p{i}-{j}", f"{j} {long}", title=f"A long paper title number {i}",
                                  doi=f"10.1/{i}", authors="['A, B', 'C, D', 'E, F', 'G, H']",
                                  journal="Journal of Tests", year="2020"))
    state["handler"] = lambda body: httpx.Response(200, json={"sources": sources})
    out = await search_literature("q", top_k=20, config=CFG)
    serialized = truncate_result(out, max_chars=result_char_limit("search_literature"))
    assert "TRUNCATED" not in serialized
    parsed = json.loads(serialized)
    assert parsed["count"] == len(parsed["papers"]) >= 5
    assert all(len(p["passages"][0]) <= literature.MAX_PASSAGE_CHARS + 2 for p in parsed["papers"])


async def test_default_ten_papers_fit_without_dropping(mock_ragstack):
    state, _ = mock_ragstack
    sources = [_chunk(f"p{i}", "text " * 400, title=f"Paper title number {i}", doi=f"10.1/{i}")
               for i in range(10)]
    state["handler"] = lambda body: httpx.Response(200, json={"sources": sources})
    out = await search_literature("q", config=CFG)
    assert out["count"] == 10 and "truncated_papers" not in out
