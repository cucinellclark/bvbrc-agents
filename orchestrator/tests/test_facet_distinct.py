"""Tests for facet_query count_distinct and facet-aware result truncation.

Background: "How many S. pneumoniae genomes are resistant to penicillin?"
needs a DISTINCT genome_id count over genome_amr.  Without one, the data
agent requested 10 000 buckets, got a hard-cut JSON blob with no total, and
the thinking model reasoned for >180 s trying to count it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import shared.agent_dispatch  # noqa: E402, F401  -- sets up agent package paths
from shared.tools import truncate_result  # noqa: E402
from shared.tools.schemas import TOOL_SCHEMA_MAP  # noqa: E402


def _buckets(n: int, count: int = 3) -> list[dict]:
    return [{"value": f"1313.{20000 + i}", "count": count} for i in range(n)]


# -----------------------------------------------------------------------
# truncate_result: facets
# -----------------------------------------------------------------------


class TestFacetTruncation:
    def test_large_facet_is_sampled_not_hard_cut(self):
        r = {"numFound": 4949, "facets": {"genome_id": _buckets(4949)}, "source": "x"}
        out = truncate_result(r, max_chars=8000)
        assert len(out) <= 8000
        assert "[TRUNCATED" not in out
        d = json.loads(out)  # valid JSON
        assert d["numFound"] == 4949
        assert d["bucket_counts"] == {"genome_id": 4949}
        assert d["_truncated"]["buckets_total"] == {"genome_id": 4949}
        shown = d["_truncated"]["buckets_shown"]["genome_id"]
        assert 0 < shown < 4949 and len(d["facets"]["genome_id"]) == shown
        assert "do not count the buckets" in d["_truncated"]["note"]

    def test_existing_counts_preserved(self):
        r = {
            "numFound": 4949,
            "facets": {"genome_id": _buckets(3000)},
            "bucket_counts": {"genome_id": 4612},
            "distinct_counts": {"genome_id": 4612},
        }
        d = json.loads(truncate_result(r, max_chars=4000))
        assert d["distinct_counts"] == {"genome_id": 4612}
        assert d["bucket_counts"] == {"genome_id": 4612}

    def test_small_facet_untouched(self):
        r = {"numFound": 10, "facets": {"host": [{"value": "Human", "count": 9}]}}
        assert truncate_result(r, max_chars=8000) == json.dumps(r, indent=2)

    def test_multi_field(self):
        r = {"numFound": 1, "facets": {"a": _buckets(2000), "b": _buckets(5)}}
        d = json.loads(truncate_result(r, max_chars=3000))
        assert d["_truncated"]["buckets_total"] == {"a": 2000, "b": 5}
        assert len(d["facets"]["b"]) <= 5

    def test_non_facet_results_unaffected(self):
        r = {"items": [{"id": i, "type": "x"} for i in range(500)], "numFound": 500}
        d = json.loads(truncate_result(r, max_chars=3000))
        assert "_summary" in d and "_truncated" in d


# -----------------------------------------------------------------------
# solr_facet_query count_distinct
# -----------------------------------------------------------------------


class TestCountDistinct:
    async def _call(self, fake_query_faceted, **kwargs):
        from data_agent.mcp_tools import agent_data_tools as adt

        fake_fn = type("F", (), {"query_faceted": AsyncMock(side_effect=fake_query_faceted)})()
        with patch.object(adt, "_get_data_functions", lambda: fake_fn), patch.object(
            adt, "_validate_query_fields", lambda *a, **k: None
        ):
            r = await adt.solr_facet_query(
                collection="genome_amr", query="antibiotic:penicillin", facet_fields=["genome_id"], **kwargs
            )
        return r, fake_fn.query_faceted

    async def test_count_distinct_returns_exact_number_and_trims_buckets(self):
        async def fake(**kw):
            assert kw["facet_limit"] == 200_000  # every bucket fetched
            return {"numFound": 4949, "facets": {"genome_id": _buckets(4612)}, "bucket_counts": {"genome_id": 4612}}

        r, _ = await self._call(fake, count_distinct=True)
        assert r["distinct_counts"] == {"genome_id": 4612}
        assert len(r["facets"]["genome_id"]) == 20
        assert "distinct_counts_capped" not in r
        assert "do not count them" in r["_note"]
        assert len(json.dumps(r)) < 3000

    async def test_capped_flag(self):
        async def fake(**kw):
            return {"numFound": 10, "facets": {"genome_id": _buckets(200_000)}}

        r, _ = await self._call(fake, count_distinct=True)
        assert r["distinct_counts_capped"] == ["genome_id"]

    async def test_bucket_counts_derived_when_missing(self):
        async def fake(**kw):
            return {"numFound": 10, "facets": {"genome_id": _buckets(7)}}

        r, _ = await self._call(fake, count_distinct=True)
        assert r["bucket_counts"] == {"genome_id": 7}
        assert r["distinct_counts"] == {"genome_id": 7}

    async def test_plain_mode_unchanged(self):
        async def fake(**kw):
            assert kw["facet_limit"] == 5
            return {"numFound": 10, "facets": {"genome_id": _buckets(5)}, "bucket_counts": {"genome_id": 5}}

        r, _ = await self._call(fake, facet_limit=5)
        assert "distinct_counts" not in r
        assert len(r["facets"]["genome_id"]) == 5

    async def test_shared_tool_forwards_flag(self):
        from shared.tools import data as shared_data

        spy = AsyncMock(return_value={"numFound": 1, "facets": {}})
        with patch.object(shared_data, "_get_solr_functions", lambda: (None, spy, None)):
            await shared_data.facet_query("genome_amr", "q", ["genome_id"], count_distinct=True)
        assert spy.await_args.kwargs["count_distinct"] is True

    def test_schema_documents_count_distinct(self):
        fn = TOOL_SCHEMA_MAP["facet_query"]["function"]
        schema = fn["parameters"]
        assert "count_distinct" in schema["properties"]
        assert schema["properties"]["count_distinct"]["type"] == "boolean"
        assert "count_distinct" in fn["description"]

    def test_mcp_query_faceted_returns_bucket_counts(self):
        src = (Path(_REPO_ROOT) / "mcp_server/functions/data_functions.py").read_text()
        assert '"bucket_counts": bucket_counts' in src
