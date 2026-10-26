"""genome_feature / pathway queries cover PATRIC annotations only.

BV-BRC stores each gene twice (PATRIC and RefSeq); the website shows PATRIC.
2026-10-20: "influenza A proteins" counted both (5,530,264 = 2,951,181 PATRIC
+ 2,579,083 RefSeq), the answer's link opened a page showing ~2.95 M, and the
FASTA download held every protein twice.  Clark: force PATRIC in code and do
not support RefSeq queries.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import shared.agent_dispatch  # noqa: E402, F401  -- sets up agent package paths
from shared.tools import data  # noqa: E402

from data_agent.mcp_tools import agent_data_tools as adt  # noqa: E402


@pytest.mark.parametrize("query, expected", [
    ('genome_name:"Influenza A virus" AND feature_type:CDS',
     '(genome_name:"Influenza A virus" AND feature_type:CDS) AND annotation:PATRIC'),
    ("*:*", "annotation:PATRIC"),
    ("", "annotation:PATRIC"),
    ("annotation:PATRIC", "annotation:PATRIC"),
    ("annotation:PATRIC AND gene:gyrA", "(annotation:PATRIC AND gene:gyrA) AND annotation:PATRIC"),
    # an annotation:PATRIC inside an OR restricts nothing, so it is still ANDed
    ("gene:x OR annotation:PATRIC", "(gene:x OR annotation:PATRIC) AND annotation:PATRIC"),
])
def test_feature_queries_are_limited_to_patric(query, expected):
    out, error = adt.patric_only("genome_feature", query)
    assert error is None and out == expected
    assert adt.patric_only("genome_feature", out) == (out, None)    # applying twice is harmless


@pytest.mark.parametrize("query", [
    "annotation:RefSeq", "annotation:(PATRIC OR RefSeq)", "annotation:*",
    "NOT annotation:PATRIC", "-annotation:PATRIC", 'gene:x AND annotation:"RefSeq"',
])
def test_any_other_annotation_is_refused(query):
    out, error = adt.patric_only("pathway", query)
    assert out == query
    assert error["errorType"] == "ANNOTATION_NOT_SUPPORTED"
    assert "PATRIC annotations only" in error["error"]


def test_other_collections_pass_through():
    assert adt.patric_only("genome", "genus:Salmonella") == ("genus:Salmonella", None)
    assert adt.patric_only("genome_amr", "annotation:RefSeq") == ("annotation:RefSeq", None)


async def test_solr_query_applies_it_for_every_caller():
    fake = SimpleNamespace(normalize_select=lambda s: s, normalize_sort=lambda s: s,
                           query_direct=AsyncMock(return_value={"numFound": 2951181}))
    with patch.object(adt, "_get_data_functions", return_value=fake):
        result = await adt.solr_query("genome_feature", "feature_type:CDS", count_only=True)
        assert fake.query_direct.await_args.kwargs["filter_str"] == "(feature_type:CDS) AND annotation:PATRIC"
        assert result["annotation"] == adt.PATRIC_ONLY_NOTE
        refused = await adt.solr_query("genome_feature", "annotation:RefSeq", count_only=True)
    assert refused["errorType"] == "ANNOTATION_NOT_SUPPORTED"
    assert fake.query_direct.await_count == 1                          # the refusal sent nothing


async def test_search_data_links_match_the_count():
    solr = AsyncMock(return_value={"numFound": 2951181})
    with patch.object(data, "_get_solr_functions", return_value=(solr, None, None)):
        result = await data.search_data(
            "genome_feature", 'genome_name:"Influenza A virus" AND feature_type:CDS', count_only=True)
    assert solr.await_args.kwargs["query"].endswith(") AND annotation:PATRIC")
    assert "eq(annotation,PATRIC)" in result["viewer_url"]
    assert "eq(annotation,PATRIC)" in result["download_tsv_url"]
    assert "eq(annotation,PATRIC)" in result["download_fasta_protein_url"]


async def test_search_data_refuses_refseq_before_querying():
    solr = AsyncMock()
    with patch.object(data, "_get_solr_functions", return_value=(solr, None, None)):
        result = await data.search_data("genome_feature", "annotation:RefSeq AND gene:gyrA")
    solr.assert_not_awaited()
    assert result["errorType"] == "ANNOTATION_NOT_SUPPORTED"


async def test_facet_query_applies_it():
    fake = SimpleNamespace(query_faceted=AsyncMock(return_value={"numFound": 1, "facets": {}}))
    with patch.object(adt, "_get_data_functions", return_value=fake):
        result = await adt.solr_facet_query("pathway", "genome_id:1.1", facet_fields=["pathway_name"])
    sent = str(fake.query_faceted.await_args)
    assert "(genome_id:1.1) AND annotation:PATRIC" in sent
    assert result["annotation"] == adt.PATRIC_ONLY_NOTE


async def test_probe_data_filters_patric():
    response = MagicMock(status_code=200, json=lambda: {"response": {"numFound": 3}, "facet_counts": {}})
    response.raise_for_status = lambda: None
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    with patch("httpx.AsyncClient", return_value=client):
        await adt.solr_probe_query("genome_feature", "gyrA")
        body = client.post.await_args.kwargs["content"]
        assert "eq(annotation,PATRIC)" in body
        await adt.solr_probe_query("genome", "Salmonella")
        assert "annotation" not in client.post.await_args.kwargs["content"]
