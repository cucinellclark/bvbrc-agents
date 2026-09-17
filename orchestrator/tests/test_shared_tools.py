"""Tests for shared tool infrastructure: identifier checks and per-tool timeouts."""

from __future__ import annotations

import asyncio
import sys
import os
import pytest

# Ensure repo root is on sys.path so ``shared`` is importable.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from shared.tools.gowe import _check_identifiers, _find_bad_value, _SRR_RE, _GENOME_ID_RE
from shared.tools import execute_tool, TOOL_TIMEOUT_OVERRIDES


# ---------------------------------------------------------------------------
# _check_identifiers tests
# ---------------------------------------------------------------------------


class TestCheckIdentifiers:
    """Validate deterministic identifier format checks."""

    # --- Valid identifiers (should return None) ---

    def test_valid_srr_accession(self):
        assert _check_identifiers({"srr_ids": ["SRR1234567"]}) is None

    def test_valid_srr_single_string(self):
        assert _check_identifiers({"srr_ids": "SRR1234567"}) is None

    def test_valid_err_accession(self):
        assert _check_identifiers({"srr_ids": ["ERR1234567"]}) is None

    def test_valid_drr_accession(self):
        assert _check_identifiers({"srr_ids": ["DRR1234567"]}) is None

    def test_valid_genome_id(self):
        assert _check_identifiers({"genome_id": "83332.12"}) is None

    def test_valid_genome_ids_list(self):
        assert _check_identifiers({"genome_ids": ["83332.12", "1334187.3"]}) is None

    def test_valid_reference_genome_id(self):
        assert _check_identifiers({"reference_genome_id": "83332.12"}) is None

    def test_valid_db_genome_list(self):
        assert _check_identifiers({"db_genome_list": ["83332.12"]}) is None

    def test_no_identifier_fields(self):
        """Inputs without any identifier fields should pass."""
        assert _check_identifiers({"output_path": "/foo/bar", "recipe": "auto"}) is None

    def test_empty_inputs(self):
        assert _check_identifiers({}) is None

    # --- Invalid identifiers (should return error string) ---

    def test_srr_with_dot_suffix(self):
        """SRR29455647.3 is a workspace filename, not an accession."""
        result = _check_identifiers({"srr_ids": ["SRR29455647.3"]})
        assert result is not None
        assert "SRR29455647.3" in result
        assert "workspace file" in result.lower()

    def test_srr_with_fasta_extension(self):
        result = _check_identifiers({"srr_ids": ["SRR29455647.3.fasta"]})
        assert result is not None
        assert "SRR29455647.3.fasta" in result

    def test_genome_id_three_parts(self):
        r"""1334187.31 with extra decimals is not a valid genome ID pattern,
        but actually 1334187.31 DOES match ^\d+\.\d+$. Test a truly bad one."""
        result = _check_identifiers({"genome_id": "not_a_genome"})
        assert result is not None
        assert "not_a_genome" in result

    def test_genome_id_no_dot(self):
        result = _check_identifiers({"genome_id": "83332"})
        assert result is not None
        assert "83332" in result

    def test_genome_id_letters(self):
        result = _check_identifiers({"genome_id": "GCA_000005845.2"})
        assert result is not None

    def test_srr_lowercase(self):
        result = _check_identifiers({"srr_ids": ["srr1234567"]})
        assert result is not None

    # --- Nested record traversal ---

    def test_nested_srr_accession_valid(self):
        """srr_libs[].srr_accession should be validated."""
        inputs = {
            "srr_libs": [
                {"srr_accession": "SRR1234567", "sample_id": "sample1"},
            ]
        }
        assert _check_identifiers(inputs) is None

    def test_nested_srr_accession_invalid(self):
        inputs = {
            "srr_libs": [
                {"srr_accession": "SRR29455647.3", "sample_id": "sample1"},
            ]
        }
        result = _check_identifiers(inputs)
        assert result is not None
        assert "SRR29455647.3" in result
        assert "srr_libs" in result

    def test_mixed_valid_and_invalid(self):
        """First bad value should be reported."""
        inputs = {"srr_ids": ["SRR1234567", "SRR9999999.3"]}
        result = _check_identifiers(inputs)
        assert result is not None
        assert "SRR9999999.3" in result


# ---------------------------------------------------------------------------
# _find_bad_value tests
# ---------------------------------------------------------------------------


class TestFindBadValue:
    def test_valid_string(self):
        assert _find_bad_value("SRR1234567", _SRR_RE) is None

    def test_invalid_string(self):
        assert _find_bad_value("bad", _SRR_RE) == "bad"

    def test_valid_list(self):
        assert _find_bad_value(["SRR111111", "ERR222222"], _SRR_RE) is None

    def test_invalid_in_list(self):
        assert _find_bad_value(["SRR111111", "nope"], _SRR_RE) == "nope"

    def test_non_string_ignored(self):
        assert _find_bad_value(42, _SRR_RE) is None


# ---------------------------------------------------------------------------
# Per-tool timeout override tests
# ---------------------------------------------------------------------------


class TestToolTimeoutOverrides:
    """Verify TOOL_TIMEOUT_OVERRIDES and max() semantics in execute_tool."""

    def test_find_similar_genomes_in_overrides(self):
        assert "find_similar_genomes" in TOOL_TIMEOUT_OVERRIDES
        assert TOOL_TIMEOUT_OVERRIDES["find_similar_genomes"] == 120.0

    @pytest.mark.asyncio
    async def test_override_increases_timeout(self):
        """When the caller's timeout is lower than the override, the override wins."""
        recorded_timeout = None

        async def fake_tool(**kwargs):
            return {"ok": True}

        # Patch asyncio.wait_for to record the actual timeout used
        original_wait_for = asyncio.wait_for

        async def capturing_wait_for(coro, timeout):
            nonlocal recorded_timeout
            recorded_timeout = timeout
            return await original_wait_for(coro, timeout=timeout)

        import shared.tools as tools_mod
        old_wait_for = asyncio.wait_for
        asyncio.wait_for = capturing_wait_for
        try:
            await execute_tool(
                tool_name="find_similar_genomes",
                arguments={},
                dispatch_table={"find_similar_genomes": fake_tool},
                timeout_seconds=30.0,
                inject_config=False,
                inject_headers=False,
            )
            assert recorded_timeout == 120.0
        finally:
            asyncio.wait_for = old_wait_for

    @pytest.mark.asyncio
    async def test_caller_timeout_not_reduced(self):
        """When the caller's timeout exceeds the override, it is kept."""
        recorded_timeout = None

        async def fake_tool(**kwargs):
            return {"ok": True}

        original_wait_for = asyncio.wait_for

        async def capturing_wait_for(coro, timeout):
            nonlocal recorded_timeout
            recorded_timeout = timeout
            return await original_wait_for(coro, timeout=timeout)

        asyncio.wait_for = capturing_wait_for
        try:
            await execute_tool(
                tool_name="find_similar_genomes",
                arguments={},
                dispatch_table={"find_similar_genomes": fake_tool},
                timeout_seconds=300.0,
                inject_config=False,
                inject_headers=False,
            )
            assert recorded_timeout == 300.0
        finally:
            asyncio.wait_for = original_wait_for

    @pytest.mark.asyncio
    async def test_no_override_uses_caller_timeout(self):
        """Tools without an override use the caller's timeout as-is."""
        recorded_timeout = None

        async def fake_tool(**kwargs):
            return {"ok": True}

        original_wait_for = asyncio.wait_for

        async def capturing_wait_for(coro, timeout):
            nonlocal recorded_timeout
            recorded_timeout = timeout
            return await original_wait_for(coro, timeout=timeout)

        asyncio.wait_for = capturing_wait_for
        try:
            await execute_tool(
                tool_name="search_data",
                arguments={},
                dispatch_table={"search_data": fake_tool},
                timeout_seconds=30.0,
                inject_config=False,
                inject_headers=False,
            )
            assert recorded_timeout == 30.0
        finally:
            asyncio.wait_for = original_wait_for
