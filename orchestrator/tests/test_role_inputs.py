"""Role inputs: which item is the reference, which samples form a condition.

cmao's Proteome Comparison runs (Sep 17, Oct 16) wanted a feature group as the
reference but sent reference_genome_index 1 (the first genome id); an RNA-Seq
run (Oct 8) sent contrasts "Cd30 vs Cd0", which the app split into letters.
shared.tools.roles refuses inputs that cannot be what the user meant and names
the roles in the confirmation question.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from shared.tools import execute_tool, gowe  # noqa: E402
from shared.tools.roles import (  # noqa: E402
    check_roles,
    labels_for_numbered_conditions,
    with_roles,
)

GENOMES = ["1313.38530", "1313.39284", "1313.39446", "1313.49400", "1313.39440", "1313.38090"]
GROUP = "/u@x/home/Feature Groups/AMR Genes 1313.49400_32"


def _proteome(index):
    return {"genome_ids": GENOMES, "user_feature_groups": [GROUP], "reference_genome_index": index}


# ---- Proteome Comparison ---------------------------------------------------

def test_cmao_index_1_names_the_first_genome_not_the_group():
    error, lines = check_roles("Proteome Comparison", _proteome(1))
    assert error is None
    assert lines == ["reference: genome 1313.38530 (reference_genome_index 1)"]


def test_the_group_is_counted_after_the_genome_ids():
    error, lines = check_roles("Proteome Comparison", _proteome(7))
    assert error is None
    assert lines == ["reference: feature group `AMR Genes 1313.49400_32` (reference_genome_index 7)"]


def test_fasta_files_count_between_genomes_and_groups():
    inputs = {"genome_ids": ["1.1"], "user_genomes": ["/u/a.faa", "/u/b.faa"],
              "user_feature_groups": ["/u/g"], "reference_genome_index": "3"}
    assert check_roles("GenomeComparison", inputs)[1] == ["reference: FASTA file `b.faa` (reference_genome_index 3)"]


def test_default_index_is_the_first_item():
    inputs = {"genome_ids": GENOMES[:2]}
    assert check_roles("proteome comparison", inputs)[1][0].startswith("reference: genome 1313.38530")


def test_out_of_range_or_non_numeric_index_is_refused_with_the_counting_order():
    for bad in (0, 8, "first", None):
        error, lines = check_roles("Proteome Comparison", _proteome(bad))
        assert error and lines == []
        assert "genome_ids first, then user_genomes, then user_feature_groups" in error
        assert "7 = feature group `AMR Genes 1313.49400_32`" in error


def test_no_inputs_and_unknown_workflows_pass_untouched():
    assert check_roles("Proteome Comparison", {}) == (None, [])
    assert check_roles("Genome Assembly", _proteome(99)) == (None, [])
    assert check_roles("Proteome Comparison", "not a dict") == (None, [])


# ---- RNA-Seq -----------------------------------------------------------------

def _rnaseq(**over):
    inputs = {
        "recipe": "HTSeq-DESeq",
        "experimental_conditions": ["ctrl", "treated"],
        "srr_libs": [
            {"sample_id": "c1", "srr_accession": "SRR0000001", "condition": "ctrl"},
            {"sample_id": "c2", "srr_accession": "SRR0000002", "condition": "ctrl"},
            {"sample_id": "t1", "srr_accession": "SRR0000003", "condition": "treated"},
            {"sample_id": "t2", "srr_accession": "SRR0000004", "condition": "treated"},
        ],
        "contrasts": [["treated", "ctrl"]],
    }
    inputs.update(over)
    return inputs


def test_labels_give_the_groups_and_contrasts():
    error, lines = check_roles("RNASeq", _rnaseq())
    assert error is None
    assert lines == ["conditions: ctrl = c1, c2; treated = t1, t2", "contrasts: treated vs ctrl"]


def test_text_contrast_is_refused_like_the_oct_8_job():
    error, _ = check_roles("RNASeq", _rnaseq(contrasts="treated vs ctrl"))
    assert "must be a list of pairs" in error and "Use labels" in error


def test_numbered_conditions_are_refused():
    libs = _rnaseq()["srr_libs"]
    libs[0]["condition"] = 0
    assert "a number" in check_roles("RNASeq", _rnaseq(srr_libs=libs))[0]
    assert "uses a number" in check_roles("RNASeq", _rnaseq(contrasts=[[2, 1]]))[0]


def test_unknown_label_missing_condition_and_self_contrast_are_refused():
    libs = _rnaseq()["srr_libs"]
    libs[0]["condition"] = "control"
    assert "not in experimental_conditions" in check_roles("RNASeq", _rnaseq(srr_libs=libs))[0]
    libs = _rnaseq()["srr_libs"]
    del libs[3]["condition"]
    assert "t2 has no condition" in check_roles("RNASeq", _rnaseq(srr_libs=libs))[0]
    assert "with itself" in check_roles("RNASeq", _rnaseq(contrasts=[["ctrl", "ctrl"]]))[0]
    assert "not a condition" in check_roles("RNASeq", _rnaseq(contrasts=[["ctrl", "other"]]))[0]


def test_deseq_needs_two_replicates_per_condition():
    libs = _rnaseq()["srr_libs"][:3]
    assert "at least 2 replicates" in check_roles("RNASeq", _rnaseq(srr_libs=libs))[0]
    assert check_roles("RNASeq", _rnaseq(srr_libs=libs, recipe="cufflinks"))[0] is None


def test_no_design_is_fine_but_contrasts_without_conditions_are_not():
    plain = {"srr_libs": [{"sample_id": "s", "srr_accession": "SRR0000001"}]}
    assert check_roles("RNASeq", plain) == (None, [])
    assert "no experimental_conditions" in check_roles("RNASeq", {**plain, "contrasts": [["a", "b"]]})[0]


def test_a_cli_job_with_numbered_conditions_is_rerun_with_labels():
    recorded = _rnaseq(contrasts=[[2, 1]])
    for lib, n in zip(recorded["srr_libs"], (1, 1, 2, 2)):
        lib["condition"] = n
    converted = labels_for_numbered_conditions("RNASeq", recorded)
    assert [lib["condition"] for lib in converted["srr_libs"]] == ["ctrl", "ctrl", "treated", "treated"]
    assert converted["contrasts"] == [["treated", "ctrl"]]
    assert check_roles("RNASeq", converted)[0] is None
    assert recorded["srr_libs"][0]["condition"] == 1  # the recorded inputs are not changed
    assert labels_for_numbered_conditions("Proteome Comparison", recorded) is recorded


# ---- MSA SNP -----------------------------------------------------------------

def test_msa_reference_lines_and_refusals():
    assert check_roles("MSA SNP", {"ref_type": "none"}) == (None, [])
    assert check_roles("MSA SNP", {})[1] == []
    assert "first input sequence" in check_roles("MSA SNP", {"ref_type": "first"})[1][0]
    assert check_roles("MSA SNP", {"ref_type": "genome_id", "ref_string": "83332.12"})[1] == ["reference: genome id 83332.12"]
    assert "needs ref_string" in check_roles("MSA SNP", {"ref_type": "feature_id"})[0]
    assert "not a BV-BRC genome id" in check_roles("MSA SNP", {"ref_type": "genome_id", "ref_string": "H37Rv"})[0]


def test_with_roles_appends_to_the_summary():
    assert with_roles("Proteome Comparison → out", []) == "Proteome Comparison → out"
    assert with_roles("X → out", ["a", "b"]) == "X → out (a; b)"


# ---- Where the check runs ----------------------------------------------------

async def test_plan_mode_question_names_the_reference():
    table = {"submit_gowe_job": AsyncMock()}
    args = {"workflow_id": "Proteome Comparison", "inputs": {**_proteome(1), "output_path": "out"}}
    result = await execute_tool("submit_gowe_job", args, table, config=SimpleNamespace(execution_mode="plan"))
    table["submit_gowe_job"].assert_not_called()
    assert result["blocked_by_mode"] is True
    assert result["summary"] == "Proteome Comparison → out (reference: genome 1313.38530 (reference_genome_index 1))"
    assert result["roles"] == ["reference: genome 1313.38530 (reference_genome_index 1)"]
    assert "In the summary, state: reference: genome 1313.38530" in result["error"]
    assert args["inputs"]["reference_genome_index"] == 1  # replayed as shown


async def test_plan_mode_refuses_an_impossible_index_before_the_question():
    table = {"submit_gowe_job": AsyncMock()}
    args = {"workflow_id": "wf_proteome", "inputs": _proteome(9)}
    with patch.dict(gowe._workflow_names, {"wf_proteome": "Proteome Comparison"}):
        result = await execute_tool("submit_gowe_job", args, table, config=SimpleNamespace(execution_mode="plan"))
    table["submit_gowe_job"].assert_not_called()
    assert not result.get("blocked_by_mode")
    assert "does not point at an input" in result["error"]


async def test_submit_refuses_before_calling_gowe():
    with patch.object(gowe, "_resolve_workflow_ref", AsyncMock(return_value=("wf_r", None))), \
         patch.dict(gowe._workflow_names, {"wf_r": "RNASeq"}), \
         patch.object(gowe, "_get_client") as client:
        result = await gowe.submit_gowe_job(
            "wf_r", _rnaseq(contrasts="treated vs ctrl"), config=SimpleNamespace(bvbrc_auth_token="t"),
        )
    client.assert_not_called()
    assert "must be a list of pairs" in result["error"]
