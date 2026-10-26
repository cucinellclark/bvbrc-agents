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

from bvbrc_mcp.tools import literature  # noqa: E402
from shared.tools import result_char_limit, truncate_result  # noqa: E402
from bvbrc_mcp.tools.literature import group_sources, search_literature  # noqa: E402


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


















# ---------------------------------------------------------------- request
















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




def test_literature_collections_default_and_env_override(monkeypatch):
    from shared.agent_dispatch import _build_config_kwargs
    from shared.models import BaseAgentConfig

    cfg = BaseAgentConfig(llm_base_url="x", llm_api_key="x", llm_model="x")
    assert cfg.literature_rag_collections == ["open-access"]
    monkeypatch.delenv("LITERATURE_RAG_COLLECTIONS", raising=False)
    assert "literature_rag_collections" not in _build_config_kwargs("t", {})
    monkeypatch.setenv("LITERATURE_RAG_COLLECTIONS", " open-access , asm-tok256 ,")
    assert _build_config_kwargs("t", {})["literature_rag_collections"] == ["open-access", "asm-tok256"]
