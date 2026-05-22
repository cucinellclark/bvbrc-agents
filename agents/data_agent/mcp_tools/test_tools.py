"""
Test the new MCP tool functions against the real BV-BRC Solr API.

Usage (from Data/ directory):
    python -m data_agent.mcp_tools.test_tools
"""

from __future__ import annotations

import asyncio
import json
import sys
import traceback
from typing import Any, Dict

from data_agent.mcp_tools.agent_data_tools import solr_query, solr_facet_query, _auto_quote_query


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_passed = 0
_failed = 0


def _print_result(name: str, result: Dict[str, Any], truncate: int = 800) -> None:
    """Pretty-print a result dict, truncating if necessary."""
    text = json.dumps(result, indent=2, default=str)
    if len(text) > truncate:
        text = text[:truncate] + f"\n  ... [truncated at {truncate} chars]"
    print(f"  Result: {text}")


def _check(name: str, condition: bool, result: Dict[str, Any], detail: str = "") -> bool:
    """Assert a condition and track pass/fail."""
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS: {detail}" if detail else "  PASS")
        return True
    else:
        _failed += 1
        print(f"  FAIL: {detail}" if detail else "  FAIL")
        _print_result(name, result)
        return False


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

async def test_count_only() -> None:
    """Test 1: solr_query with count_only=True (Salmonella genomes)."""
    name = "test_count_only"
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"  Query: genome, genus:Salmonella, count_only=True")

    result = await solr_query(
        collection="genome",
        query="genus:Salmonella",
        count_only=True,
    )

    _print_result(name, result)
    _check(name, "error" not in result, result, "No error returned")
    _check(name, "numFound" in result, result, "Has numFound key")
    if "numFound" in result:
        _check(
            name,
            result["numFound"] > 10000,
            result,
            f"numFound={result['numFound']} > 10000 (expected ~48000+)",
        )
    _check(name, result.get("source") == "bvbrc-mcp-data", result, "Has source marker")


async def test_amr_count() -> None:
    """Test 2: solr_query count for AMR data (ciprofloxacin resistant)."""
    name = "test_amr_count"
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"  Query: genome_amr, ciprofloxacin resistant, count_only=True")

    result = await solr_query(
        collection="genome_amr",
        query="antibiotic:ciprofloxacin AND resistant_phenotype:Resistant",
        count_only=True,
    )

    _print_result(name, result)
    _check(name, "error" not in result, result, "No error returned")
    _check(name, "numFound" in result, result, "Has numFound key")
    if "numFound" in result:
        _check(
            name,
            result["numFound"] > 10000,
            result,
            f"numFound={result['numFound']} > 10000 (expected ~116000+)",
        )


async def test_return_records() -> None:
    """Test 3: solr_query returning records with select fields."""
    name = "test_return_records"
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"  Query: sp_gene, E. coli virulence factors, select=[gene,product], limit=5")

    result = await solr_query(
        collection="sp_gene",
        query='property:"Virulence Factor" AND organism:"Escherichia coli"',
        select=["gene", "product"],
        limit=5,
    )

    _print_result(name, result)
    _check(name, "error" not in result, result, "No error returned")
    _check(name, "results" in result, result, "Has results key")
    if "results" in result:
        _check(
            name,
            len(result["results"]) <= 5,
            result,
            f"Got {len(result['results'])} results (expected <= 5)",
        )
        _check(
            name,
            len(result["results"]) > 0,
            result,
            "Got at least 1 result",
        )
        if result["results"]:
            first = result["results"][0]
            # Check that selected fields are present
            has_fields = "gene" in first or "product" in first
            _check(name, has_fields, result, f"First record has gene or product fields: {list(first.keys())}")
    _check(name, "numFound" in result, result, "Has numFound key")
    _check(name, "count" in result, result, "Has count key")


async def test_facet_query() -> None:
    """Test 4: solr_facet_query (Staphylococcus host organisms)."""
    name = "test_facet_query"
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"  Query: genome, genus:Staphylococcus, facet host_scientific_name, limit=10")

    result = await solr_facet_query(
        collection="genome",
        query="genus:Staphylococcus",
        facet_fields=["host_scientific_name"],
        facet_limit=10,
    )

    _print_result(name, result)
    _check(name, "error" not in result, result, "No error returned")
    _check(name, "numFound" in result, result, "Has numFound key")
    _check(name, "facets" in result, result, "Has facets key")
    if "facets" in result:
        _check(
            name,
            "host_scientific_name" in result["facets"],
            result,
            "Has host_scientific_name facet",
        )
        facet_values = result["facets"].get("host_scientific_name", [])
        _check(
            name,
            len(facet_values) > 0,
            result,
            f"Got {len(facet_values)} facet values",
        )
        _check(
            name,
            len(facet_values) <= 10,
            result,
            f"Got {len(facet_values)} values (<= 10 limit)",
        )
        if facet_values:
            first_val = facet_values[0]
            _check(
                name,
                "value" in first_val and "count" in first_val,
                result,
                f"Facet entry has value+count: {first_val}",
            )
    _check(name, result.get("source") == "bvbrc-mcp-data", result, "Has source marker")


async def test_pagination() -> None:
    """Test 5: solr_query with pagination (cursor_id)."""
    name = "test_pagination"
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"  Query: genome, genus:Salmonella, limit=3, then paginate")

    # First page
    result1 = await solr_query(
        collection="genome",
        query="genus:Salmonella",
        select=["genome_id", "genome_name"],
        limit=3,
    )

    _print_result(name, result1, truncate=400)
    _check(name, "error" not in result1, result1, "Page 1: No error")
    _check(name, "results" in result1, result1, "Page 1: Has results")

    next_cursor = result1.get("nextCursorId")
    _check(name, next_cursor is not None, result1, f"Page 1: Has nextCursorId={next_cursor is not None}")

    if next_cursor:
        # Second page
        result2 = await solr_query(
            collection="genome",
            query="genus:Salmonella",
            select=["genome_id", "genome_name"],
            limit=3,
            cursor_id=next_cursor,
        )

        _print_result(name, result2, truncate=400)
        _check(name, "error" not in result2, result2, "Page 2: No error")
        _check(name, "results" in result2, result2, "Page 2: Has results")

        if "results" in result1 and "results" in result2:
            ids1 = {r.get("genome_id") for r in result1["results"]}
            ids2 = {r.get("genome_id") for r in result2["results"]}
            _check(
                name,
                len(ids1 & ids2) == 0,
                result2,
                f"Pages have different records (overlap={len(ids1 & ids2)})",
            )


async def test_sort() -> None:
    """Test 6: solr_query with sort expression."""
    name = "test_sort"
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"  Query: genome, genus:Salmonella, sort=genome_name asc, limit=3")

    result = await solr_query(
        collection="genome",
        query="genus:Salmonella",
        select=["genome_id", "genome_name"],
        sort="genome_name asc",
        limit=3,
    )

    _print_result(name, result)
    _check(name, "error" not in result, result, "No error returned")
    _check(name, "results" in result, result, "Has results")
    if "results" in result and len(result["results"]) >= 2:
        names = [r.get("genome_name", "") for r in result["results"]]
        _check(
            name,
            names == sorted(names),
            result,
            f"Results sorted ascending: {names}",
        )


async def test_error_invalid_collection() -> None:
    """Test 7: Error handling with invalid collection name."""
    name = "test_error_invalid_collection"
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"  Query: nonexistent_collection, *:*")

    result = await solr_query(
        collection="nonexistent_collection",
        query="*:*",
    )

    _print_result(name, result)
    _check(name, "error" in result, result, "Returns error dict")
    _check(name, result.get("source") == "bvbrc-mcp-data", result, "Has source marker even on error")


async def test_facet_no_fields() -> None:
    """Test 8: Facet query with no facet fields returns error."""
    name = "test_facet_no_fields"
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"  Query: genome, *:*, facet_fields=[]")

    result = await solr_facet_query(
        collection="genome",
        query="*:*",
        facet_fields=[],
    )

    _print_result(name, result)
    _check(name, "error" in result, result, "Returns error for empty facet_fields")


async def test_multi_facet_fields() -> None:
    """Test 9: Facet query with multiple facet fields."""
    name = "test_multi_facet_fields"
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"  Query: genome, genus:Salmonella, facet [host_name, isolation_country]")

    result = await solr_facet_query(
        collection="genome",
        query="genus:Salmonella",
        facet_fields=["host_name", "isolation_country"],
        facet_limit=5,
    )

    _print_result(name, result)
    _check(name, "error" not in result, result, "No error returned")
    if "facets" in result:
        _check(
            name,
            "host_name" in result["facets"],
            result,
            "Has host_name facet",
        )
        _check(
            name,
            "isolation_country" in result["facets"],
            result,
            "Has isolation_country facet",
        )


# ---------------------------------------------------------------------------
# Range query tests
# ---------------------------------------------------------------------------

async def test_numeric_range() -> None:
    """Test 10: solr_query with numeric range filter."""
    name = "test_numeric_range"
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"  Query: genome, Salmonella AND genome_length:[4000000 TO 5000000]")

    result = await solr_query(
        collection="genome",
        query="genus:Salmonella AND genome_length:[4000000 TO 5000000]",
        select=["genome_id", "genome_name", "genome_length"],
        limit=5,
    )

    _print_result(name, result)
    _check(name, "error" not in result, result, "No error returned")
    _check(name, "results" in result, result, "Has results key")
    if "results" in result and result["results"]:
        lengths = [r.get("genome_length", 0) for r in result["results"]]
        all_in_range = all(4000000 <= gl <= 5000000 for gl in lengths)
        _check(name, all_in_range, result, f"All genome_length values in range: {lengths}")
    _check(name, result.get("numFound", 0) > 0, result, f"numFound={result.get('numFound')} > 0")


async def test_year_range() -> None:
    """Test 11: solr_query with integer year range."""
    name = "test_year_range"
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"  Query: genome, collection_year:[2020 TO 2023], count_only")

    result = await solr_query(
        collection="genome",
        query="collection_year:[2020 TO 2023]",
        count_only=True,
    )

    _print_result(name, result)
    _check(name, "error" not in result, result, "No error returned")
    _check(name, "numFound" in result, result, "Has numFound key")
    if "numFound" in result:
        _check(
            name,
            result["numFound"] > 100000,
            result,
            f"numFound={result['numFound']} > 100000 (many genomes in 2020-2023)",
        )


async def test_string_date_range() -> None:
    """Test 12: solr_query with string date range on genome."""
    name = "test_string_date_range"
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f'  Query: genome, Salmonella AND collection_date:["2023-01" TO "2023-12"]')

    result = await solr_query(
        collection="genome",
        query='genus:Salmonella AND collection_date:["2023-01" TO "2023-12"]',
        select=["genome_id", "collection_date", "collection_year"],
        limit=5,
    )

    _print_result(name, result)
    _check(name, "error" not in result, result, "No error returned")
    _check(name, "results" in result, result, "Has results key")
    if "results" in result and result["results"]:
        dates = [r.get("collection_date", "") for r in result["results"]]
        all_2023 = all(d.startswith("2023") for d in dates if d)
        _check(name, all_2023, result, f"All collection_date values start with 2023: {dates}")
    _check(name, result.get("numFound", 0) > 0, result, f"numFound={result.get('numFound')} > 0")


async def test_open_ended_range() -> None:
    """Test 13: solr_query with open-ended range (greater than)."""
    name = "test_open_ended_range"
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"  Query: genome, gc_content:[70 TO *]")

    result = await solr_query(
        collection="genome",
        query="gc_content:[70 TO *]",
        select=["genome_id", "genome_name", "gc_content"],
        limit=5,
    )

    _print_result(name, result)
    _check(name, "error" not in result, result, "No error returned")
    _check(name, "results" in result, result, "Has results key")
    if "results" in result and result["results"]:
        gcs = [r.get("gc_content", 0) for r in result["results"]]
        all_above_70 = all(gc >= 70 for gc in gcs)
        _check(name, all_above_70, result, f"All gc_content >= 70: {gcs}")
    _check(name, result.get("numFound", 0) > 0, result, f"numFound={result.get('numFound')} > 0")


async def test_open_ended_range_upper() -> None:
    """Test 14: solr_query with open-ended range (less than or equal)."""
    name = "test_open_ended_range_upper"
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"  Query: genome, contigs:[* TO 10] AND genus:Salmonella")

    result = await solr_query(
        collection="genome",
        query="contigs:[* TO 10] AND genus:Salmonella",
        select=["genome_id", "genome_name", "contigs"],
        limit=5,
    )

    _print_result(name, result)
    _check(name, "error" not in result, result, "No error returned")
    _check(name, "results" in result, result, "Has results key")
    if "results" in result and result["results"]:
        contig_counts = [r.get("contigs", 0) for r in result["results"]]
        all_lte_10 = all(c <= 10 for c in contig_counts)
        _check(name, all_lte_10, result, f"All contigs <= 10: {contig_counts}")
    _check(name, result.get("numFound", 0) > 0, result, f"numFound={result.get('numFound')} > 0")


async def test_combined_range_filter() -> None:
    """Test 15: solr_query combining range with exact match filters."""
    name = "test_combined_range_filter"
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"  Query: genome, Staphylococcus AND collection_year:[2020 TO *] AND contigs:[* TO 50]")

    result = await solr_query(
        collection="genome",
        query="genus:Staphylococcus AND collection_year:[2020 TO *] AND contigs:[* TO 50]",
        select=["genome_id", "genome_name", "collection_year", "contigs"],
        limit=5,
        count_only=True,
    )

    _print_result(name, result)
    _check(name, "error" not in result, result, "No error returned")
    _check(name, "numFound" in result, result, "Has numFound key")
    if "numFound" in result:
        _check(
            name,
            result["numFound"] > 0,
            result,
            f"numFound={result['numFound']} > 0 (combined range query has results)",
        )


# ---------------------------------------------------------------------------
# Auto-quoting tests (synchronous, no API calls)
# ---------------------------------------------------------------------------

def test_auto_quote_fixes() -> None:
    """Test 16: _auto_quote_query fixes unquoted multi-word values."""
    name = "test_auto_quote_fixes"
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"  Testing auto-quoting of unquoted multi-word values")

    cases = [
        ('organism:Escherichia coli AND property:"Virulence Factor"',
         'organism:"Escherichia coli" AND property:"Virulence Factor"'),
        ('organism:Salmonella enterica AND host_name:Homo sapiens',
         'organism:"Salmonella enterica" AND host_name:"Homo sapiens"'),
        ('host_name:Homo sapiens',
         'host_name:"Homo sapiens"'),
        ('organism:Escherichia coli',
         'organism:"Escherichia coli"'),
    ]

    for inp, expected in cases:
        result = _auto_quote_query(inp)
        _check(name, result == expected, {"input": inp, "expected": expected, "got": result},
               f"'{inp}' -> '{result}'")


def test_auto_quote_preserves() -> None:
    """Test 17: _auto_quote_query preserves already-correct queries."""
    name = "test_auto_quote_preserves"
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"  Testing auto-quoting does not modify correct queries")

    cases = [
        'organism:"Escherichia coli" AND property:"Virulence Factor"',
        'genus:Salmonella AND host_name:Human',
        '*:*',
        'collection_year:[2020 TO 2024]',
        'gc_content:[60 TO *]',
        'product:*kinase*',
        'genome_id:(83332.12 OR 208964.12)',
        'genus:Salmonella',
        'resistant_phenotype:Resistant AND antibiotic:ciprofloxacin',
    ]

    for inp in cases:
        result = _auto_quote_query(inp)
        _check(name, result == inp, {"input": inp, "got": result},
               f"Preserved: '{inp}'")


async def test_auto_quote_live() -> None:
    """Test 18: Auto-quoting fixes the real E. coli virulence factors query."""
    name = "test_auto_quote_live"
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"  Query: sp_gene, organism:Escherichia coli (unquoted) AND Virulence Factor")

    # This is the exact query the LLM produces -- unquoted organism
    result = await solr_query(
        collection="sp_gene",
        query='organism:Escherichia coli AND property:"Virulence Factor"',
        select=["gene", "product"],
        limit=5,
    )

    _print_result(name, result)
    _check(name, "error" not in result, result, "No error returned")
    _check(name, result.get("numFound", 0) > 0, result,
           f"numFound={result.get('numFound')} > 0 (auto-quoting fixed the query)")
    _check(name, "results" in result, result, "Has results key")
    if "results" in result and result["results"]:
        first = result["results"][0]
        has_fields = "gene" in first or "product" in first
        _check(name, has_fields, result, f"First record has gene or product: {list(first.keys())}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def main() -> None:
    global _passed, _failed

    print("=" * 60)
    print("BV-BRC Agent Data Tools - Test Suite")
    print("Testing against real BV-BRC Solr API")
    print("=" * 60)

    # Synchronous tests first (no API calls)
    test_auto_quote_fixes()
    test_auto_quote_preserves()

    tests = [
        test_count_only,
        test_amr_count,
        test_return_records,
        test_facet_query,
        test_pagination,
        test_sort,
        test_error_invalid_collection,
        test_facet_no_fields,
        test_multi_facet_fields,
        # Range query tests
        test_numeric_range,
        test_year_range,
        test_string_date_range,
        test_open_ended_range,
        test_open_ended_range_upper,
        test_combined_range_filter,
        # Auto-quoting live test
        test_auto_quote_live,
    ]

    for test_fn in tests:
        try:
            await test_fn()
        except Exception as e:
            _failed += 1
            print(f"  EXCEPTION: {type(e).__name__}: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"RESULTS: {_passed} passed, {_failed} failed, {_passed + _failed} total")
    print(f"{'='*60}")

    if _failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
