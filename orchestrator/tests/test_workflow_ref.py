"""Workflow references: accept a name, and never dead-end on a bad id.

GoWe's resolveWorkflow already accepts an id OR a name, on both
GET /workflows/:id/inputs and POST /submissions. The failure mode was never the
server -- it was the LLM inventing a UUID that matches nothing and getting back
a bare "GoWeError: No ..." with no way to recover.

Observed 2026-10-08: HA Subtype never submitted because the agent tried
  wf_893a5074-258b-4682-b9fa-6cb79a52943e   (HA's prefix + MetaCATS's suffix)
  wf_893a5074-2c3c-488a-a044-2ed684ec7f50   (one digit off the real id)
and then gave up. Both of those exact strings are asserted below.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.tools import gowe  # noqa: E402

CATALOG = [
    {"id": "wf_893a5074-5c3c-4838-a044-1ed684ec7f50", "name": "HA Subtype Numbering Conversion"},
    {"id": "wf_3f556526-258b-4682-b9fa-6cb79a52943e", "name": "MetaCATS"},
    {"id": "wf_4965af9b-8715-4cb0-a189-44546214c9ee", "name": "Taxonomic Classification"},
]
HA = CATALOG[0]["id"]


class _Client:
    def __init__(self, catalog=CATALOG):
        self._catalog = catalog

    async def list_workflows(self, **_kw):
        return self._catalog, None


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(gowe, "_get_client", lambda config=None: _Client())
    monkeypatch.setattr(gowe, "_get_auth", lambda config=None: "tok")


@pytest.mark.asyncio
@pytest.mark.parametrize("ref", [
    "HA Subtype Numbering Conversion",
    "ha subtype numbering conversion",          # case-insensitive
    "  HA Subtype Numbering Conversion  ",      # stray whitespace
    HA,                                         # the real id still works
])
async def test_resolves(patched, ref):
    wid, err = await gowe._resolve_workflow_ref(ref, None)
    assert err is None
    assert wid == HA


@pytest.mark.asyncio
@pytest.mark.parametrize("bad,expect_hint", [
    # The two the agent actually invented.
    ("wf_893a5074-258b-4682-b9fa-6cb79a52943e", "MetaCATS"),
    ("wf_893a5074-2c3c-488a-a044-2ed684ec7f50", "HA Subtype Numbering Conversion"),
    # A partial name and a typo.
    ("HA Subtype", "HA Subtype Numbering Conversion"),
    ("Taxonomic Clasification", "Taxonomic Classification"),
])
async def test_bad_ref_names_the_alternative(patched, bad, expect_hint):
    wid, err = await gowe._resolve_workflow_ref(bad, None)
    assert wid is None
    # The message must name a real alternative, or the model has nothing to act
    # on -- a bare "not found" is what left HA Subtype unsubmitted.
    assert expect_hint in err
    assert "list_gowe_workflows" in err


@pytest.mark.asyncio
async def test_catalog_unreachable_passes_through(monkeypatch):
    """A listing failure must not mask the real error from the actual call."""
    def boom(config=None):
        raise RuntimeError("gowe down")
    monkeypatch.setattr(gowe, "_get_client", boom)
    wid, err = await gowe._resolve_workflow_ref("whatever", None)
    assert err is None and wid == "whatever"


@pytest.mark.asyncio
async def test_get_workflow_inputs_accepts_a_name(patched, monkeypatch):
    class C(_Client):
        async def get_workflow_inputs(self, wid, **_kw):
            assert wid == HA, f"should have resolved to the id, got {wid}"
            return [{"id": "types", "type": "string[]"}]
    monkeypatch.setattr(gowe, "_get_client", lambda config=None: C())
    r = await gowe.get_workflow_inputs("HA Subtype Numbering Conversion", config=None)
    assert r["workflow_id"] == HA
    assert r["count"] == 1


@pytest.mark.asyncio
async def test_get_workflow_inputs_bad_ref_returns_the_hint(patched):
    r = await gowe.get_workflow_inputs("wf_893a5074-258b-4682-b9fa-6cb79a52943e", config=None)
    assert "error" in r and "MetaCATS" in r["error"]
    assert "inputs" not in r
