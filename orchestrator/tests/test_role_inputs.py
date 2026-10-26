"""Role inputs: which item is the reference, which samples form a condition.

cmao's Proteome Comparison runs (Sep 17, Oct 16) wanted a feature group as the
reference but sent reference_genome_index 1 (the first genome id); an RNA-Seq
run (Oct 8) sent contrasts "Cd30 vs Cd0", which the app split into letters.
bvbrc_mcp.tools.roles refuses inputs that cannot be what the user meant and names
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

from shared.tools import execute_tool  # noqa: E402
from bvbrc_mcp.tools import gowe  # noqa: E402
from bvbrc_mcp.tools.roles import (  # noqa: E402
    check_roles,
    labels_for_numbered_conditions,
    with_roles,
)

GENOMES = ["1313.38530", "1313.39284", "1313.39446", "1313.49400", "1313.39440", "1313.38090"]
GROUP = "/u@x/home/Feature Groups/AMR Genes 1313.49400_32"


def _proteome(index):
    return {"genome_ids": GENOMES, "user_feature_groups": [GROUP], "reference_genome_index": index}


# ---- Proteome Comparison ---------------------------------------------------













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
















# ---- MSA SNP -----------------------------------------------------------------





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


