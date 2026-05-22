"""
Machine-readable field registry parsed from the collection reference.

Provides fast lookup of valid field names per collection and reverse
lookup of which collections contain a given field.  Built automatically
at import time from the same COLLECTION_REFERENCE string that is
included in the system prompt, so the two can never drift apart.

Usage:
    from data_agent.prompts.field_registry import (
        COLLECTION_FIELDS,
        FIELD_TO_COLLECTIONS,
        validate_fields,
        suggest_collections,
    )
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Set

from data_agent.prompts.collection_reference import COLLECTION_REFERENCE

# ---------------------------------------------------------------------------
# Parse the reference text
# ---------------------------------------------------------------------------

# Matches "[collection_name]  (primary key: ...)"
_COLLECTION_HEADER_RE = re.compile(r"^\[(\w+)\]")

# Matches "  field_name (type): description"
_FIELD_LINE_RE = re.compile(r"^  (\w+)\s+\(")

_collection_fields: Dict[str, Set[str]] = {}
_field_to_collections: Dict[str, List[str]] = defaultdict(list)

_current_collection: str | None = None

for line in COLLECTION_REFERENCE.splitlines():
    header_match = _COLLECTION_HEADER_RE.match(line)
    if header_match:
        _current_collection = header_match.group(1)
        _collection_fields[_current_collection] = set()
        continue

    if _current_collection is not None:
        field_match = _FIELD_LINE_RE.match(line)
        if field_match:
            field_name = field_match.group(1)
            _collection_fields[_current_collection].add(field_name)
            _field_to_collections[field_name].append(_current_collection)

# Freeze as public module-level constants
COLLECTION_FIELDS: Dict[str, Set[str]] = dict(_collection_fields)
FIELD_TO_COLLECTIONS: Dict[str, List[str]] = dict(_field_to_collections)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def validate_fields(collection: str, fields: List[str]) -> List[str]:
    """Return field names that do NOT exist on *collection*.

    Returns an empty list when all fields are valid.  If the collection
    itself is unknown (not in the registry), returns an empty list so we
    don't block queries to undocumented collections.
    """
    valid = COLLECTION_FIELDS.get(collection)
    if valid is None:
        return []  # unknown collection — skip validation
    return [f for f in fields if f not in valid]


def suggest_collections(field: str) -> List[str]:
    """Return collections that *do* have *field*, sorted alphabetically."""
    return sorted(FIELD_TO_COLLECTIONS.get(field, []))
