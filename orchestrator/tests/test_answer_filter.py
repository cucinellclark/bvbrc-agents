"""Links in an answer must come from this turn's tool results (or url_utils patterns).

Seen 2026-10-09 (helpdesk, Qwen3.8, "who is the PI of BV-BRC"):
docs.bv-brc.org/website/personnel.html (domain does not resolve) and
www.bv-brc.org/contact (404), next to real docs_url links.  Sep: an invented
/view/SimilarGenomeList/ page and /job/ links.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.answer_filter import REMOVED, collect_known_links, filter_answer  # noqa: E402

RQL_VIEW = "https://www.bv-brc.org/view/GenomeList/?and(eq(genus,Salmonella),eq(genome_status,Complete))"
TSV = "https://www.bv-brc.org/api/genome/?and(eq(genus,Salmonella),eq(genome_status,Complete))&http_accept=text/tsv&http_download=true"
DOCS = "https://www.bv-brc.org/docs/quick_references/services/blast.html"
TOOL_RESULTS = [{"viewer_url": RQL_VIEW, "download_tsv_url": TSV,
                 "results": [{"docs_url": DOCS, "doi": "10.1093/nar/gkac1003"}]}]
KNOWN = collect_known_links(TOOL_RESULTS)


def f(text):
    return filter_answer(text, KNOWN)


def test_links_from_tool_results_are_kept_with_rql_parentheses():
    text = f"[genomes](<{RQL_VIEW}>) and [TSV]({TSV}). Docs: {DOCS}."
    out, removed = f(text)
    assert out == text and removed == []


def test_invented_links_are_unwrapped_to_their_text():
    out, removed = f("See [the team](https://docs.bv-brc.org/website/personnel.html) or "
                     "[contact us](https://www.bv-brc.org/contact).")
    assert out == "See the team or contact us."
    assert removed == ["https://docs.bv-brc.org/website/personnel.html", "https://www.bv-brc.org/contact"]


def test_bare_invented_url_becomes_link_removed_and_keeps_punctuation():
    out, _ = f("Write to https://www.bv-brc.org/contact.")
    assert out == f"Write to {REMOVED}."


def test_link_whose_text_is_the_url_is_replaced():
    out, _ = f("[https://foo.example/x](https://foo.example/x)")
    assert out == REMOVED


def test_url_utils_page_patterns_are_kept_without_a_tool_result():
    for url in ("https://www.bv-brc.org/", "https://www.bv-brc.org/view/Genome/83332.12",
                "https://www.bv-brc.org/workspace/u@patricbrc.org/home/x",
                "https://www.bv-brc.org/app/ComprehensiveGenomeAnalysis"):
        assert f(url)[1] == [], url


def test_unknown_viewer_name_is_removed():
    out, removed = f("[list](<https://www.bv-brc.org/view/SimilarGenomeList/?x>)")
    assert out == "list" and removed


def test_dois_must_come_from_the_tools():
    out, removed = f("https://doi.org/10.1093/nar/gkac1003 and [x](https://doi.org/10.1000/made.up)")
    assert out == "https://doi.org/10.1093/nar/gkac1003 and x"
    assert removed == ["https://doi.org/10.1000/made.up"]


def test_images_are_unwrapped_too():
    out, _ = f("![chart](https://evil.example/c.png)")
    assert out == "chart"


def test_relative_and_non_http_links_are_untouched():
    text = "[here](#section) and [mail](mailto:help@bv-brc.org)"
    assert f(text) == (text, [])


def test_no_links_is_a_no_op():
    assert f("plain text") == ("plain text", [])


# --- dispatch_agent filters against the agent's own tool results ------------

@pytest.mark.asyncio
async def test_dispatch_filters_using_the_agents_tool_results():
    from shared import agent_dispatch

    tc = SimpleNamespace(name="query_helpdesk", arguments={})
    result = SimpleNamespace(tool_trace=[SimpleNamespace(tool_call=tc, result={"docs_url": DOCS},
                                                         error=None, duration_ms=1)])

    async def runner(query, config_kwargs, ctx, progress_callback):
        return {"answer": f"See {DOCS} and https://docs.bv-brc.org/website/personnel.html",
                "status": "completed", "tool_trace": [], "known_links": agent_dispatch._known_links(result)}

    with patch.dict(agent_dispatch._AGENT_RUNNERS, {"helpdesk": runner}):
        resp = await agent_dispatch.dispatch_agent("helpdesk", "who is the PI", token="t", context={})
    assert resp["answer"] == f"See {DOCS} and {REMOVED}"
    assert resp["links_removed"] == ["https://docs.bv-brc.org/website/personnel.html"]


# --- executor keeps known_links out of result_for_ui -------------------------

@pytest.mark.asyncio
async def test_executor_moves_known_links_to_the_event():
    from orchestrator.events.events import EventType
    from orchestrator.events.stream import collect_events
    from orchestrator.executor.agent_executor import execute_agent_step
    from orchestrator.models import OrchestratorRequest
    from orchestrator.router.models import Step

    class _Agent:
        key = "helpdesk"
        config = SimpleNamespace(chat_tool="agent_chat", chat_tool_params={}, mcp_server_name=None, timeout_seconds=300)
        tool_names = ["agent_chat"]

        async def call_tool(self, name, arguments, progress_handler=None):
            return {"answer": "ok", "status": "completed", "known_links": [DOCS]}

    events = await collect_events(execute_agent_step(Step(agent_key="helpdesk", task="q"), _Agent(),
                                                     OrchestratorRequest(query="q")))
    ev = next(e for e in events if e.type == EventType.AGENT_RESULT)
    assert ev.data["known_links"] == [DOCS]
    assert "known_links" not in ev.data["result_for_ui"]


# --- orchestrator filters a direct answer -----------------------------------

@pytest.mark.asyncio
async def test_direct_answer_is_filtered():
    from orchestrator import orchestrate as orch
    from orchestrator.events.events import EventType
    from orchestrator.models import OrchestratorRequest
    from orchestrator.router.models import RoutingDecision

    decision = RoutingDecision(decision="direct",
                               direct_response="Hi! More at https://www.bv-brc.org/contact or https://www.bv-brc.org/")

    async def fake_route(*a, **k):
        return decision

    with patch.object(orch, "route", fake_route), \
         patch.object(orch, "prepare_attached_documents", lambda *a, **k: _noop()):
        events = [e async for e in orch.orchestrate(OrchestratorRequest(query="hello"),
                                                    registry=SimpleNamespace(), llm=None)]
    done = next(e for e in events if e.type == EventType.ORCHESTRATOR_DONE)
    assert done.data["response_text"] == f"Hi! More at {REMOVED} or https://www.bv-brc.org/"


async def _noop():
    return None


def test_tool_urls_with_raw_spaces_are_kept():
    """url_utils leaves raw spaces in RQL values; the real TSV link must survive."""
    from shared.tools.url_utils import enrich_search_result, solr_to_rql

    r = {"results": [], "numFound": 5}
    enrich_search_result(r, "genome", solr_to_rql('species:"Escherichia coli" AND genome_status:Complete'))
    known = collect_known_links([r])
    text = f"[view](<{r['viewer_url']}>) and [TSV](<{r['download_tsv_url']}>)"
    assert filter_answer(text, known) == (text, [])
