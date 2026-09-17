"""
Shared tool infrastructure for BV-BRC agents.

Provides:
  - ``execute_tool``: Generic tool dispatcher with timeout and error handling.
  - ``truncate_result``: Smart result-to-JSON serializer with representative
    sampling across types, aggregate summaries, and binary-search sizing.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import traceback
from collections import Counter, defaultdict
from typing import Any, Dict


# Tools that legitimately run longer than the agent's default tool timeout.
TOOL_TIMEOUT_OVERRIDES: dict[str, float] = {
    "find_similar_genomes": 120.0,
}


async def execute_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    dispatch_table: Dict[str, Any],
    timeout_seconds: float = 30.0,
    config: Any = None,
    headers: Dict[str, str] | None = None,
    inject_config: bool = True,
    inject_headers: bool = True,
) -> Dict[str, Any]:
    """Execute a tool by name from the given dispatch table.

    Handles:
      - Looking up the tool function in *dispatch_table*
      - Optionally injecting *config* and *headers* into the call arguments
      - Timeout enforcement
      - Error wrapping (tool errors become structured dicts, not exceptions)

    Args:
        tool_name: Name of the tool to execute.
        arguments: Arguments from the LLM's ``tool_call``.
        dispatch_table: Mapping of ``{tool_name: async_callable}``.
        timeout_seconds: Maximum execution time.
        config: Agent config to inject (any Pydantic model).
        headers: HTTP headers to inject (e.g. auth).
        inject_config: Whether to inject ``config`` into arguments.
        inject_headers: Whether to inject ``headers`` into arguments.

    Returns:
        Dict with the tool's result, or an error dict if execution failed.
    """
    func = dispatch_table.get(tool_name)
    if func is None:
        return {
            "error": f"Unknown tool: '{tool_name}'",
            "available_tools": sorted(dispatch_table.keys()),
        }

    # Apply per-tool timeout overrides (use the larger of caller vs override)
    timeout_seconds = max(timeout_seconds, TOOL_TIMEOUT_OVERRIDES.get(tool_name, 0.0))

    # Inject config and headers if the tool accepts them
    if inject_config and config is not None and "config" not in arguments:
        arguments["config"] = config
    if inject_headers and headers is not None and "headers" not in arguments:
        arguments["headers"] = headers

    try:
        result = await asyncio.wait_for(
            func(**arguments),
            timeout=timeout_seconds,
        )
        return result

    except asyncio.TimeoutError:
        return {
            "error": f"Tool '{tool_name}' timed out after {timeout_seconds}s",
            "tool": tool_name,
            "arguments": _safe_args(arguments),
        }

    except TypeError as e:
        return {
            "error": f"Invalid arguments for tool '{tool_name}': {str(e)}",
            "tool": tool_name,
            "arguments": _safe_args(arguments),
        }

    except Exception as e:
        return {
            "error": f"Tool '{tool_name}' failed: {type(e).__name__}: {str(e)}",
            "tool": tool_name,
            "arguments": _safe_args(arguments),
            "traceback": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# Smart truncation with representative sampling
# ---------------------------------------------------------------------------


def _build_items_summary(items: list) -> Dict[str, Any]:
    """Build an aggregate summary of items for the LLM.

    Returns counts by type, folder distribution, size stats, date range, etc.
    so the LLM can answer aggregation questions even when items are truncated.
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
    """Serialize a tool result to JSON, truncating if too large.

    Smart truncation strategy:
      1. Strips ``ui_grid`` payloads (duplicate data for UI rendering).
      2. Adds aggregate summary (type counts, folder distribution, etc.).
      3. Samples items representatively across types (not just first N).
      4. Strips verbose metadata from sampled items.
      5. Uses binary search to find the largest sample that fits.
      6. Falls back to hard truncation if nothing else works.

    Args:
        result: The tool result dict.
        max_chars: Maximum characters for the serialized output.

    Returns:
        JSON string of the result, possibly truncated.
    """
    result = _strip_ui_grid(result)
    serialized = json.dumps(result, indent=2, default=str)

    if len(serialized) <= max_chars:
        return serialized

    # Find the list-valued key to truncate (check nested "result" envelope too)
    for list_key in ("items", "results", "records", "files", "ids"):
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
                    "note": (
                        f"Showing {mid} of {num_items} items "
                        f"(representative sample across types)."
                    ),
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
                "note": (
                    f"Showing {best_n} of {num_items} items "
                    f"(representative sample across types)."
                ),
            }
            return json.dumps(truncated, indent=2, default=str)

    # Fallback: hard truncate
    return serialized[:max_chars] + f"\n... [TRUNCATED at {max_chars} chars]"


def _safe_args(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Strip config/headers from arguments for safe error reporting."""
    return {k: v for k, v in arguments.items() if k not in ("config", "headers")}
