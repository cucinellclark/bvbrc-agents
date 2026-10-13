"""list_collections / get_collection_fields must find data_types.xlsx.

The shared-tools refactor (dbceff9) dropped the file's real location from the
search list, so both tools returned nothing for every collection; on Oct 13 the
planning agent was told "Collection 'genome_amr' not found."
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.tools.collections import get_collection_fields, list_collections  # noqa: E402


def test_list_collections_is_not_empty():
    assert asyncio.run(list_collections())["count"] > 10


def test_genome_amr_fields():
    result = asyncio.run(get_collection_fields("genome_amr"))
    assert "error" not in result
    names = {f["name"] for f in result["fields"]}
    assert {"genome_id", "antibiotic", "resistant_phenotype"} <= names
