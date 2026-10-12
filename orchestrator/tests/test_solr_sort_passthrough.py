"""search_data(sort=...) must reach Solr.

Every resource's stream_all_solr accepted ``sort`` and then hard-coded
``sort=f"{unique_key} asc"``, so every "largest / longest / latest / highest"
answer came back with the lowest IDs: "longest genome" = 100.11 at 4.83 Mbp
(FigCore, Sep), "latest complete RSV genome" from 2018 (cmao).

The Solr client lives in mcp_server/bvbrc-python-api, which is a copied
folder with no git history here, so this test is what keeps the fix from
being lost if that folder is replaced.
"""

import importlib
import inspect
import pkgutil
import sys
from pathlib import Path

import pytest

SOLR_API = Path(__file__).resolve().parents[2] / "mcp_server" / "bvbrc-python-api"
if not SOLR_API.is_dir():
    pytest.skip("mcp_server/bvbrc-python-api is not present", allow_module_level=True)
sys.path.insert(0, str(SOLR_API))

import bvbrc_solr_api.resources as resources_pkg  # noqa: E402

CTX = {"base_url": "https://www.bv-brc.org/api", "solr_base_url": "https://www.bv-brc.org/api"}


def _resource_classes():
    for mod_info in pkgutil.iter_modules(resources_pkg.__path__):
        mod = importlib.import_module(f"bvbrc_solr_api.resources.{mod_info.name}")
        for _, cls in inspect.getmembers(mod, inspect.isclass):
            if cls.__module__ == mod.__name__ and hasattr(cls, "stream_all_solr"):
                yield cls


CLASSES = sorted(_resource_classes(), key=lambda c: c.__name__)


def test_found_the_resources():
    assert len(CLASSES) >= 30


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_caller_sort_is_used_with_a_tiebreaker(cls):
    pager = cls(CTX).stream_all_solr(sort="date_inserted desc")
    assert pager.sort.startswith("date_inserted desc")
    # cursorMark needs the unique key as the final tie-breaker
    if pager.unique_key:
        assert pager.sort.endswith(f"{pager.unique_key} asc")


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_default_sort_is_the_unique_key(cls):
    pager = cls(CTX).stream_all_solr()
    if pager.unique_key:
        assert pager.sort == f"{pager.unique_key} asc"
