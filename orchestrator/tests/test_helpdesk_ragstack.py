"""query_helpdesk on the RAGStack backend (current_plans/ragstack_helpdesk_reingest.md, step 5).

``helpdesk_backend: "ragstack"`` in the MCP config's ``rag_database`` section sends
the query to a RAGStack collection with the caller's token; anything else keeps
the legacy RAG API. No network: ``requests.post`` is patched.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "mcp_server") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "mcp_server"))

import shared.agent_dispatch  # noqa: E402, F401
from functions import rag_database_functions as rdf  # noqa: E402

RAGSTACK_CFG = {
    "helpdesk_backend": "ragstack",
    "ragstack_base_url": "https://example.org/ragstack/clark/api/",
    "ragstack_collection": "bvbrc_helpdesk",
}


def _response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


RAGSTACK_HIT = {
    "sources": [
        {
            "content": "Use the Genome Assembly service...",
            "score": 0.93,
            "metadata": {"filename": "quick_references__services__genome_assembly_service.md",
                         "chunk_index": 2, "tenant_id": "bvbrc:someone"},
        }
    ]
}


def test_ragstack_request_shape():
    with patch.object(rdf.requests, "post", return_value=_response(RAGSTACK_HIT)) as post:
        rdf.query_rag_helpdesk_func("assemble reads", top_k=3, config=RAGSTACK_CFG,
                                    summarize=False, auth_token="un=u|tokenid=x")
    url = post.call_args.args[0]
    kwargs = post.call_args.kwargs
    assert url == "https://example.org/ragstack/clark/api/v1/retrieve"
    # Defaults chosen in step 4: vector retrieval, reranker off.
    assert kwargs["json"] == {"query": "assemble reads", "collection": "bvbrc_helpdesk",
                              "top_k": 3, "retrieval_mode": "vector", "rerank": False}
    assert kwargs["headers"]["Authorization"] == "Bearer un=u|tokenid=x"
    # Cloudflare 403s the default Python-urllib agent; we always name ourselves.
    assert kwargs["headers"]["User-Agent"].startswith("bvbrc-copilot-helpdesk")


def test_existing_bearer_prefix_not_doubled():
    with patch.object(rdf.requests, "post", return_value=_response({"sources": []})) as post:
        rdf.query_rag_helpdesk_func("q", config=RAGSTACK_CFG, summarize=False,
                                    auth_token="Bearer abc")
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer abc"


def test_ragstack_results_carry_doc_path_and_url():
    with patch.object(rdf.requests, "post", return_value=_response(RAGSTACK_HIT)):
        out = rdf.query_rag_helpdesk_func("q", config=RAGSTACK_CFG, summarize=False, auth_token="t")
    assert out["count"] == 1
    assert out["source"] == "bvbrc-ragstack"
    assert out["index"] == "bvbrc_helpdesk"
    md = out["results"][0]["metadata"]
    assert md["doc_path"] == "quick_references/services/genome_assembly_service"
    assert md["docs_url"] == ("https://www.bv-brc.org/docs/quick_references/services/"
                              "genome_assembly_service.html")
    assert "tenant_id" not in md  # internal RAGStack fields are not passed to the LLM


def test_ragstack_without_token_is_an_error_not_a_request():
    with patch.object(rdf.requests, "post") as post:
        out = rdf.query_rag_helpdesk_func("q", config=RAGSTACK_CFG, summarize=False, auth_token=None)
    post.assert_not_called()
    assert out["count"] == 0
    assert out["errorType"] == "API_ERROR"
    assert "token" in out["error"]


def test_ragstack_http_error_shape():
    resp = MagicMock()
    resp.raise_for_status.side_effect = rdf.requests.HTTPError("403 Forbidden")
    with patch.object(rdf.requests, "post", return_value=resp):
        out = rdf.query_rag_helpdesk_func("q", config=RAGSTACK_CFG, summarize=False, auth_token="t")
    assert out["errorType"] == "RAG_API_REQUEST_ERROR"
    assert out["results"] == []


def test_legacy_backend_unchanged_by_default():
    legacy = {"rag_api_base_url": "http://old:8001", "database_name": "bvbrc_helpdesk"}
    payload = {"documents": [{"content": "c", "score": 0.5, "metadata": {"doc_id": "d"}}]}
    with patch.object(rdf.requests, "post", return_value=_response(payload)) as post:
        out = rdf.query_rag_helpdesk_func("q", top_k=4, config=legacy, summarize=False,
                                          auth_token="ignored")
    assert post.call_args.args[0] == "http://old:8001/query/bvbrc_helpdesk"
    assert post.call_args.kwargs["json"] == {"query": "q", "top_k": 4, "score_threshold": 0.0}
    assert out["source"] == "bvbrc-rag-api"
    assert out["results"] == [{"content": "c", "score": 0.5, "metadata": {"doc_id": "d"}}]


def test_doc_path_mapping():
    assert rdf._ragstack_doc_path("about.md") == "about"
    assert rdf._ragstack_doc_path("cli_tutorial__cli_installation.md") == "cli_tutorial/cli_installation"
    assert rdf._ragstack_doc_path(None) is None


async def test_agent_impl_passes_user_token(tmp_path):
    """The Copilot tool path searches as the user: their token reaches the function."""
    from helpdesk_agent.mcp_tools import agent_helpdesk_tools as aht
    from helpdesk_agent.models import AgentConfig

    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.json").write_text('{"rag_database": {"helpdesk_backend": "ragstack"}}')
    cfg = AgentConfig(bvbrc_auth_token="un=u|tokenid=x", mcp_server_path=str(tmp_path),
                      llm_base_url="http://llm", llm_api_key="k", llm_model="m")
    fake = MagicMock()
    fake.query_rag_helpdesk_func.return_value = {"results": [], "count": 0}
    with patch.object(aht, "_get_rag_functions", return_value=fake):
        await aht.query_helpdesk_impl("q", top_k=5, config=cfg)
    kwargs = fake.query_rag_helpdesk_func.call_args.kwargs
    assert kwargs["auth_token"] == "un=u|tokenid=x"
    assert kwargs["summarize"] is False
    assert kwargs["config"] == {"helpdesk_backend": "ragstack"}
