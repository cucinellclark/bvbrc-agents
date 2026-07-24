"""
Tool dispatcher for the BV-BRC Workspace Exploration Agent.

Maps tool names (from the LLM's tool_calls) to their async implementation
functions.  Core implementations come from the shared tool library.

The workspace agent keeps its own sophisticated ``truncate_result`` that
does representative sampling across types, UI grid stripping, and
aggregate summary generation.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from typing import Any, Dict

from shared.tools import execute_tool as _shared_execute_tool
from shared.tools.workspace import (
    workspace_browse,
    get_file_metadata,
    read_file_preview,
)
from shared.tools.similar_genome import find_similar_genomes
from shared.tools.literature import search_literature

# ---------------------------------------------------------------------------
# Dispatch table: tool name -> async callable
# ---------------------------------------------------------------------------
TOOL_DISPATCH: Dict[str, Any] = {
    "workspace_browse": workspace_browse,
    "get_file_metadata": get_file_metadata,
    "read_file_preview": read_file_preview,
    "find_similar_genomes": find_similar_genomes,
    "search_literature": search_literature,
}


async def execute_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    timeout_seconds: float = 30.0,
    config: Any = None,
    headers: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Execute a tool by name with the given arguments.

    Delegates to the shared ``execute_tool`` with this agent's dispatch table.
    """
    return await _shared_execute_tool(
        tool_name=tool_name,
        arguments=arguments,
        dispatch_table=TOOL_DISPATCH,
        timeout_seconds=timeout_seconds,
        config=config,
        headers=headers,
    )


# ---------------------------------------------------------------------------
# Workspace-agent-specific truncation (representative sampling)
# ---------------------------------------------------------------------------


def _build_items_summary(items: list) -> Dict[str, Any]:
    """Build an aggregate summary of workspace items for the LLM.

    Returns counts by type, folder distribution, size stats, etc. so the LLM
    can answer aggregation questions even when the full item list is truncated.
    """
    type_counts: Counter = Counter()
    folder_counts: Counter = Counter()
    total_size = 0
    dates: list[str] = []

    for item in items:
        if isinstance(item, dict):
            item_type = item.get("type", "unknown")
            item_path = item.get("path", "")
            item_size = item.get("size", 0)
            item_date = item.get("creation_time", "")
        elif isinstance(item, list) and len(item) >= 4:
            item_type = item[1] if len(item) > 1 else "unknown"
            item_path = item[2] if len(item) > 2 else ""
            item_size = item[6] if len(item) > 6 else 0
            item_date = item[3] if len(item) > 3 else ""
        else:
            continue

        type_counts[item_type] += 1

        parent = os.path.dirname(item_path.rstrip("/")) if item_path else ""
        if parent:
            folder_counts[parent] += 1

        if isinstance(item_size, (int, float)):
            total_size += item_size
        if item_date:
            dates.append(str(item_date))

    summary: Dict[str, Any] = {
        "total_items": len(items),
        "counts_by_type": dict(type_counts.most_common()),
        "unique_types": sorted(type_counts.keys()),
    }

    if folder_counts:
        summary["top_folders"] = dict(folder_counts.most_common(10))
    if total_size > 0:
        summary["total_size_bytes"] = total_size
    if dates:
        sorted_dates = sorted(dates)
        summary["date_range"] = {
            "earliest": sorted_dates[0],
            "latest": sorted_dates[-1],
        }

    return summary


def _slim_item(item: Any) -> Any:
    """Strip verbose metadata fields from a workspace item dict."""
    if not isinstance(item, dict):
        return item
    drop_keys = {
        "userMeta",
        "autoMeta",
        "link_reference",
        "user_permissions",
        "global_permission",
    }
    return {k: v for k, v in item.items() if k not in drop_keys}


def _sample_representative(items: list, budget: int) -> list:
    """Pick a representative sample of items, ensuring each type is included."""
    if len(items) <= budget:
        return [_slim_item(i) for i in items]

    by_type: dict[str, list] = defaultdict(list)
    for item in items:
        if isinstance(item, dict):
            t = item.get("type", "unknown")
        elif isinstance(item, list) and len(item) > 1:
            t = item[1]
        else:
            t = "unknown"
        by_type[t].append(item)

    sampled: list = []
    remaining_budget = budget

    # Phase 1: one item per type
    for t, group in by_type.items():
        if remaining_budget <= 0:
            break
        sampled.append(_slim_item(group[0]))
        remaining_budget -= 1

    # Phase 2: fill proportionally from each type
    if remaining_budget > 0:
        for t, group in by_type.items():
            share = max(0, math.floor((len(group) / len(items)) * remaining_budget))
            for item in group[1 : 1 + share]:
                sampled.append(_slim_item(item))

    # Fill remaining budget
    if len(sampled) < budget:
        seen = {id(s) for s in sampled}
        for item in items:
            if len(sampled) >= budget:
                break
            if id(item) not in seen:
                sampled.append(_slim_item(item))
                seen.add(id(item))

    return sampled[:budget]


def _strip_ui_grid(obj: Any) -> Any:
    """Recursively remove ``ui_grid`` keys from a result dict."""
    if isinstance(obj, dict):
        return {k: _strip_ui_grid(v) for k, v in obj.items() if k != "ui_grid"}
    if isinstance(obj, list):
        return [_strip_ui_grid(v) for v in obj]
    return obj


def truncate_result(result: Dict[str, Any], max_chars: int = 8000) -> str:
    """Workspace-agent-specific truncation with representative sampling.

    1. Strips ``ui_grid`` payloads (duplicate item data for UI rendering).
    2. Adds aggregate summary (type counts, folder distribution, etc.).
    3. Samples items representatively across types (not just first N).
    4. Strips verbose metadata from sampled items.
    """
    result = _strip_ui_grid(result)
    serialized = json.dumps(result, indent=2, default=str)

    if len(serialized) <= max_chars:
        return serialized

    for list_key in ("items", "results", "records", "files"):
        target = result
        if "result" in result and isinstance(result["result"], dict):
            target = result["result"]

        if list_key in target and isinstance(target[list_key], list):
            all_items = target[list_key]
            num_items = len(all_items)
            summary = _build_items_summary(all_items)

            truncated = dict(result)
            if "result" in truncated and isinstance(truncated["result"], dict):
                truncated["result"] = dict(truncated["result"])
                inner = truncated["result"]
            else:
                inner = truncated

            inner["_summary"] = summary

            # Binary search for the largest representative sample that fits
            lo, hi = 1, num_items
            best_n = 1
            while lo <= hi:
                mid = (lo + hi) // 2
                inner[list_key] = _sample_representative(all_items, mid)
                inner["_truncated"] = {
                    "total": num_items,
                    "shown": mid,
                    "note": f"Showing {mid} of {num_items} items (representative sample across types).",
                }
                candidate = json.dumps(truncated, indent=2, default=str)
                if len(candidate) <= max_chars:
                    best_n = mid
                    serialized = candidate
                    lo = mid + 1
                else:
                    hi = mid - 1

            inner[list_key] = _sample_representative(all_items, best_n)
            inner["_truncated"] = {
                "total": num_items,
                "shown": best_n,
                "note": f"Showing {best_n} of {num_items} items (representative sample across types).",
            }
            return json.dumps(truncated, indent=2, default=str)

    # Fallback: hard truncate
    return serialized[:max_chars] + f"\n... [TRUNCATED at {max_chars} chars]"
