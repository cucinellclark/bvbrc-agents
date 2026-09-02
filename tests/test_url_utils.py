"""Tests for shared.tools.url_utils — viewer URL construction.

Covers:
- Regression: already-mapped collections produce correct /view/ URLs
- genome_amr: GenomeList#view_tab=amr, Antibiotic page, catch-all
- Newly mapped collections: antibiotics, genome_sequence, experiment
- 17 unmapped collections: build_viewer_url returns None (no /search/)
- enrich_search_result: viewer_url absent when build_viewer_url is None
"""

import sys
import os
import pytest

# Ensure bvbrc-agents repo root is on sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from shared.tools.url_utils import (
    build_viewer_url,
    enrich_search_result,
    NO_VIEWER_COLLECTIONS,
    COLLECTION_VIEWER_MAP,
    BVBRC_BASE_URL,
)

BASE = BVBRC_BASE_URL


# ── Regression: already-mapped collections ───────────────────────────


class TestMappedCollections:
    """Collections in COLLECTION_VIEWER_MAP produce /view/<Widget>/ URLs."""

    def test_genome(self):
        url = build_viewer_url("genome", "and(eq(genus,Mycobacterium),eq(species,Mycobacterium tuberculosis))")
        assert url == f"{BASE}/view/GenomeList/?and(eq(genus,Mycobacterium),eq(species,Mycobacterium tuberculosis))"

    def test_genome_feature(self):
        url = build_viewer_url("genome_feature", "eq(gene,katG)")
        assert url == f"{BASE}/view/FeatureList/?eq(gene,katG)"

    def test_epitope(self):
        url = build_viewer_url("epitope", "eq(epitope_type,linear)")
        assert url == f"{BASE}/view/EpitopeList/?eq(epitope_type,linear)"

    def test_empty_collection_returns_none(self):
        assert build_viewer_url("", "eq(a,b)") is None

    def test_empty_query_returns_none(self):
        assert build_viewer_url("genome", "") is None


# ── Newly mapped collections ─────────────────────────────────────────


class TestNewlyMappedCollections:
    """antibiotics, genome_sequence, experiment now have list viewers."""

    def test_antibiotics(self):
        url = build_viewer_url("antibiotics", "eq(antibiotic_name,isoniazid)")
        assert url == f"{BASE}/view/AntibioticList/?eq(antibiotic_name,isoniazid)"

    def test_genome_sequence(self):
        url = build_viewer_url("genome_sequence", "eq(genome_id,83332.12)")
        assert url == f"{BASE}/view/SequenceList/?eq(genome_id,83332.12)"

    def test_experiment(self):
        url = build_viewer_url("experiment", "eq(organism,Mycobacterium tuberculosis)")
        assert url == f"{BASE}/view/ExperimentList/?eq(organism,Mycobacterium tuberculosis)"


# ── genome_amr special-case ──────────────────────────────────────────


class TestGenomeAmrViewerUrl:
    """genome_amr queries produce GenomeList#view_tab=amr or Antibiotic URLs."""

    def test_genome_name_plus_antibiotic(self):
        """Plan example 2: genome_name + antibiotic → GenomeList with AMR tab."""
        rql = "and(eq(antibiotic,isoniazid),eq(genome_name,Mycobacterium tuberculosis))"
        url = build_viewer_url("genome_amr", rql)
        assert url is not None
        assert "/view/GenomeList/?" in url
        assert "#view_tab=amr" in url
        # genome_name should be in the query string (before #)
        qs, fragment = url.split("#", 1)
        assert "genome_name" in qs
        # antibiotic should be in the filter (after #)
        assert "antibiotic" in fragment

    def test_genome_name_plus_antibiotic_plus_phenotype(self):
        """Plan example 3: genome_name + antibiotic + resistant_phenotype."""
        rql = "and(eq(antibiotic,isoniazid),eq(genome_name,Mycobacterium tuberculosis),eq(resistant_phenotype,Resistant))"
        url = build_viewer_url("genome_amr", rql)
        assert url is not None
        assert "/view/GenomeList/?" in url
        assert "#view_tab=amr" in url
        qs, fragment = url.split("#", 1)
        assert "genome_name" in qs
        assert "antibiotic" in fragment
        assert "resistant_phenotype" in fragment

    def test_single_antibiotic_only(self):
        """Single eq(antibiotic,X) with no genome fields → Antibiotic page."""
        rql = "eq(antibiotic,isoniazid)"
        url = build_viewer_url("genome_amr", rql)
        assert url is not None
        assert "/view/Antibiotic/?" in url
        assert "eq(antibiotic_name,isoniazid)" in url
        assert "#view_tab=amr" in url

    def test_antibiotic_plus_phenotype(self):
        """eq(antibiotic,X) + eq(resistant_phenotype,Y) → Antibiotic page with filter."""
        rql = "and(eq(antibiotic,isoniazid),eq(resistant_phenotype,Resistant))"
        url = build_viewer_url("genome_amr", rql)
        assert url is not None
        assert "/view/Antibiotic/?" in url
        assert "eq(antibiotic_name,isoniazid)" in url
        assert "#view_tab=amr" in url
        assert "resistant_phenotype" in url

    def test_amr_only_no_drug_catch_all(self):
        """AMR-only fields, no single antibiotic → GenomeList catch-all."""
        rql = "eq(resistant_phenotype,Resistant)"
        url = build_viewer_url("genome_amr", rql)
        assert url is not None
        assert "/view/GenomeList/?" in url
        assert "eq(genome_id,*)" in url
        assert "#view_tab=amr" in url
        assert "resistant_phenotype" in url

    def test_genome_id_is_genome_field(self):
        """genome_id is a shared field and belongs in the GenomeList query."""
        rql = "and(eq(genome_id,83332.12),eq(antibiotic,isoniazid))"
        url = build_viewer_url("genome_amr", rql)
        assert url is not None
        qs, fragment = url.split("#", 1)
        assert "genome_id" in qs
        assert "antibiotic" in fragment

    def test_taxon_id_is_genome_field(self):
        """taxon_id is a shared field."""
        rql = "and(eq(taxon_id,1773),eq(antibiotic,rifampin))"
        url = build_viewer_url("genome_amr", rql)
        assert url is not None
        qs, fragment = url.split("#", 1)
        assert "taxon_id" in qs
        assert "antibiotic" in fragment

    def test_no_search_fallback(self):
        """genome_amr never produces a /search/ URL."""
        rql = "eq(antibiotic,isoniazid)"
        url = build_viewer_url("genome_amr", rql)
        assert "/search/" not in url

    def test_genome_fields_only(self):
        """Only genome fields, no AMR fields → GenomeList + amr tab, no filter."""
        rql = "eq(genome_name,Mycobacterium tuberculosis)"
        url = build_viewer_url("genome_amr", rql)
        assert url is not None
        assert "/view/GenomeList/?" in url
        assert "#view_tab=amr" in url
        # No filter= because there are no AMR-specific predicates
        assert "filter=" not in url


# ── Unmapped collections return None ─────────────────────────────────


NO_VIEWER_LIST = sorted(NO_VIEWER_COLLECTIONS)


@pytest.mark.parametrize("collection", NO_VIEWER_LIST)
class TestNoViewerCollections:
    """Collections with no production list viewer return None."""

    def test_build_viewer_url_is_none(self, collection):
        url = build_viewer_url(collection, "eq(id,123)")
        assert url is None

    def test_no_search_url(self, collection):
        url = build_viewer_url(collection, "eq(id,123)")
        assert url is None  # specifically not /search/


# ── enrich_search_result integration ─────────────────────────────────


class TestEnrichSearchResult:
    """enrich_search_result skips viewer_url when build_viewer_url returns None."""

    def test_mapped_collection_has_viewer_url(self):
        result = enrich_search_result({}, "genome", "eq(genus,Salmonella)")
        assert "viewer_url" in result
        assert "/view/GenomeList/" in result["viewer_url"]

    def test_unmapped_collection_no_viewer_url(self):
        result = enrich_search_result({}, "bioset", "eq(id,123)")
        assert "viewer_url" not in result

    def test_unmapped_collection_still_has_tsv(self):
        result = enrich_search_result({}, "bioset", "eq(id,123)")
        assert "download_tsv_url" in result

    def test_genome_amr_has_viewer_url(self):
        result = enrich_search_result(
            {}, "genome_amr", "eq(antibiotic,isoniazid)"
        )
        assert "viewer_url" in result
        assert "/search/" not in result["viewer_url"]
        assert "#view_tab=amr" in result["viewer_url"]

    def test_genome_amr_still_has_tsv(self):
        result = enrich_search_result(
            {}, "genome_amr", "eq(antibiotic,isoniazid)"
        )
        assert "download_tsv_url" in result

    def test_antibiotics_has_viewer_url(self):
        result = enrich_search_result({}, "antibiotics", "eq(antibiotic_name,isoniazid)")
        assert "viewer_url" in result
        assert "/view/AntibioticList/" in result["viewer_url"]

    def test_error_result_skips_all_urls(self):
        result = enrich_search_result({"error": "timeout"}, "genome", "eq(genus,X)")
        assert "viewer_url" not in result
        assert "download_tsv_url" not in result
