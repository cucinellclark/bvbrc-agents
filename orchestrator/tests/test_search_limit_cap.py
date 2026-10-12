"""search_data rows are capped at 200 for agents (allow_large lifts it).

Results are cut to ~8000 chars before the model sees them, so big pulls only
burn time: marcnguy's Oct 9 turn spent its 300 s on two cursor-paged 1000-row
pulls and timed out before reaching the tree service.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.tools import data  # noqa: E402


async def _search(**kw):
    solr = AsyncMock(return_value={"results": [], "count": 0, "numFound": 9000})
    with patch.object(data, "_get_solr_functions", lambda: (solr, None, None)):
        r = await data.search_data("genome", "genus:Streptococcus", **kw)
    return r, solr.await_args.kwargs


@pytest.mark.asyncio
async def test_large_limit_is_capped_with_a_note():
    r, sent = await _search(limit=1000)
    assert sent["limit"] == data.AGENT_SEARCH_LIMIT == 200
    assert "you asked for 1000" in r["limit_note"]
    assert "count_only" in r["limit_note"]


@pytest.mark.asyncio
async def test_small_limit_untouched_and_no_note():
    r, sent = await _search(limit=50)
    assert sent["limit"] == 50 and "limit_note" not in r


@pytest.mark.asyncio
async def test_allow_large_lifts_the_cap():
    r, sent = await _search(limit=1000, allow_large=True)
    assert sent["limit"] == 1000 and "limit_note" not in r


@pytest.mark.asyncio
async def test_count_only_is_not_capped():
    _, sent = await _search(limit=5000, count_only=True)
    assert sent["limit"] == 5000


def test_schema_documents_the_cap_and_allow_large():
    from shared.tools.schemas import TOOL_SCHEMA_MAP
    props = TOOL_SCHEMA_MAP["search_data"]["function"]["parameters"]["properties"]
    assert "allow_large" in props and "200" in props["limit"]["description"]
