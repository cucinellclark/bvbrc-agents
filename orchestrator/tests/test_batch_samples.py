"""D4 Phase 4: ``list_samples`` pairs read files by name; ``prepare_batch``
prepares one job per sample from one populated template, with no LLM per job.

The model used to pair R1/R2 files by eye (and invent paths, F16) and to
build one ``submit_gowe_job`` call per sample.  Both are code now.  The
result is ``prepared_actions`` -> ``blocked_actions`` -> one Yes / No / Other
question, replayed verbatim like a rerun.  Workflow input shapes below are
copied from the registered GoWe workflows (2026-10-14).
"""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import shared.agent_dispatch  # noqa: E402, F401  -- sets up agent package paths
from shared.models import BaseAgentState, ToolCall  # noqa: E402
from shared.step_outputs import build_step_outputs  # noqa: E402
from shared.tools import (  # noqa: E402
    HANDOFF_TARGETS,
    TOOL_AGENT_ALLOWLIST,
    TOOL_RESULT_CHAR_LIMITS,
    TOOL_TIMEOUT_OVERRIDES,
    execute_tool,
)
from shared.tools import batch, gowe, samples  # noqa: E402
from service_agent import agent as service_agent  # noqa: E402
from service_agent.models import AgentConfig, AgentState, Intent  # noqa: E402
from service_agent.phases import populate  # noqa: E402
from service_agent.phases import replay as replay_mod  # noqa: E402

USER = "alice@patricbrc.org"
HOME = f"/{USER}/home"
SID = "s-123"
CHAT = f"{HOME}/chats/{SID}"
TOKEN = f"un={USER}|tokenid=x|sig=y"
READS = f"{HOME}/reads"


def _files(*names):
    return [{"name": n, "type": "reads", "path": f"{READS}/{n}"} for n in names]


# ---------------------------------------------------------------------------
# list_samples: pairing
# ---------------------------------------------------------------------------

def test_illumina_names_pair_on_r1_r2_and_keep_the_sample_prefix():
    out = samples.pair_read_files(_files(
        "EC1_S1_L001_R1_001.fastq.gz", "EC1_S1_L001_R2_001.fastq.gz",
        "EC2_S2_L001_R2_001.fastq.gz", "EC2_S2_L001_R1_001.fastq.gz",
    ))
    assert out["paired"] == [
        {"sample_id": "EC1_S1_L001", "read1": f"{READS}/EC1_S1_L001_R1_001.fastq.gz",
         "read2": f"{READS}/EC1_S1_L001_R2_001.fastq.gz"},
        {"sample_id": "EC2_S2_L001", "read1": f"{READS}/EC2_S2_L001_R1_001.fastq.gz",
         "read2": f"{READS}/EC2_S2_L001_R2_001.fastq.gz"},
    ]
    assert out["single"] == [] and out["unpaired"] == []


@pytest.mark.parametrize("r1,r2,sid", [
    ("SRR8238942_1.fastq", "SRR8238942_2.fastq", "SRR8238942"),
    ("liver.R1.fq.gz", "liver.R2.fq.gz", "liver"),
    ("A-R1.fq", "A-R2.fq", "A"),
    ("B_1.FASTQ.GZ", "B_2.FASTQ.GZ", "B"),
    # Trim Galore output (seen in a real workspace, 2026-10-14)
    ("SRR21835899_1_val_1.fq", "SRR21835899_2_val_2.fq", "SRR21835899"),
    ("SRR21835899_1_val_1.sampled.fq", "SRR21835899_2_val_2.sampled.fq", "SRR21835899"),
])
def test_the_usual_mate_patterns_pair(r1, r2, sid):
    out = samples.pair_read_files(_files(r2, r1))
    assert out["paired"] == [{"sample_id": sid, "read1": f"{READS}/{r1}", "read2": f"{READS}/{r2}"}]


def test_a_digit_in_the_sample_name_is_not_a_mate_when_r1_r2_are_there():
    # sample_1_R1 must not pair with sample_2_R1 just because 1/2 differ.
    out = samples.pair_read_files(_files("sample_1_R1.fq", "sample_2_R1.fq", "sample_2_R2.fq"))
    assert out["paired"] == [{"sample_id": "sample_2", "read1": f"{READS}/sample_2_R1.fq",
                              "read2": f"{READS}/sample_2_R2.fq"}]
    assert out["unpaired"] == [{"read": f"{READS}/sample_1_R1.fq", "mate": 1}]


def test_lone_mates_are_unpaired_and_plain_files_are_single_end():
    out = samples.pair_read_files(_files("x_R1.fastq", "nanopore_run.fastq.gz", "notes.txt", "x.fasta"))
    assert out["unpaired"] == [{"read": f"{READS}/x_R1.fastq", "mate": 1}]
    assert out["single"] == [{"sample_id": "nanopore_run", "read": f"{READS}/nanopore_run.fastq.gz"}]
    assert out["paired"] == []


def _cfg(session=True):
    return SimpleNamespace(
        bvbrc_auth_token=TOKEN, gowe_url="http://gowe-test", mcp_server_path=None,
        session_id=SID if session else None, workspace_path=HOME if session else None,
    )


async def _list(items, **kwargs):
    browse = AsyncMock(return_value={"result": {"path": READS, "items": items}})
    with patch.object(samples, "workspace_browse", browse):
        out = await samples.list_samples(config=_cfg(), headers={"Authorization": TOKEN}, **kwargs)
    return out, browse


@pytest.mark.asyncio
async def test_list_samples_reads_one_folder_level_and_checks_sra_ids():
    items = _files("a_R1.fq", "a_R2.fq") + [{"name": "sub", "type": "folder", "path": f"{READS}/sub"}]
    out, browse = await _list(items, path=READS, sra_ids=["srr1234567", "SRR1234567", "SRR12", "ERR7916233"])
    kwargs = browse.await_args.kwargs
    assert kwargs["path"] == READS and "file_extensions" not in kwargs and "name_contains" not in kwargs
    assert out["paired"][0]["sample_id"] == "a"
    assert out["sra"] == [{"sample_id": "SRR1234567", "srr_accession": "SRR1234567"},
                          {"sample_id": "ERR7916233", "srr_accession": "ERR7916233"}]
    assert out["invalid_sra"] == ["SRR12"]
    assert out["sample_count"] == 3


@pytest.mark.asyncio
async def test_a_folder_of_subfolders_says_so_and_errors_pass_through():
    out, _ = await _list([{"name": "run1", "type": "folder", "path": f"{READS}/run1"}], path=READS)
    assert out["sample_count"] == 0 and "subfolder" in out["note"]
    browse = AsyncMock(return_value={"error": "not found", "errorType": "PATH_NOT_FOUND"})
    with patch.object(samples, "workspace_browse", browse):
        err = await samples.list_samples(path="/nope", config=_cfg())
    assert err["errorType"] == "PATH_NOT_FOUND"
    assert (await samples.list_samples(config=_cfg()))["errorType"] == "INVALID_PARAMETERS"


# ---------------------------------------------------------------------------
# prepare_batch
# ---------------------------------------------------------------------------

def _rec(id_, type_, fields):
    return {"id": id_, "type": type_, "fields": [{"name": f} for f in fields]}


SPECS = {
    "wf_asm": ("GenomeAssembly", [
        _rec("paired_end_libs", "record:paired_end_lib[]?", ["read1", "read2", "platform", "interleaved"]),
        _rec("single_end_libs", "record:single_end_lib[]?", ["read", "platform"]),
        {"id": "srr_ids", "type": "string[]?"}, {"id": "recipe", "type": "string?"},
        {"id": "output_path", "type": "string"}, {"id": "output_file", "type": "string"},
    ]),
    "wf_tc": ("Taxonomic Classification", [
        _rec("paired_end_libs", "record:paired_end_lib[]?", ["sample_id", "read1", "read2"]),
        _rec("srr_libs", "record:srr_lib[]?", ["sample_id", "srr_accession"]),
        {"id": "database", "type": "string?"},
        {"id": "output_path", "type": "string"}, {"id": "output_file", "type": "string"},
    ]),
    "wf_bin": ("Metagenomic Binning", [
        _rec("paired_end_libs", "record:paired_end_lib?", ["read1", "read2"]),
        _rec("single_end_libs", "record:single_end_lib?", ["read"]),
        {"id": "srr_ids", "type": "string[]?"},
        {"id": "output_path", "type": "string"}, {"id": "output_file", "type": "string"},
    ]),
    "wf_cga": ("ComprehensiveGenomeAnalysis", [
        _rec("paired_end_libs", "record:paired_end_lib[]?", ["read1", "read2"]),
        {"id": "scientific_name", "type": "string?"}, {"id": "taxonomy_id", "type": "int?"},
        {"id": "output_path", "type": "string"}, {"id": "output_file", "type": "string"},
    ]),
    "wf_ts": ("Tree Sort", [{"id": "input_fasta_file", "type": "string"}]),
}


class FakeGoWe:
    def __init__(self, refuse=None):
        self.refuse = refuse or {}
        self.dry_runs = []

    async def list_workflows(self, auth_token=None, limit=50, offset=0):
        return [{"id": i, "name": n} for i, (n, _) in SPECS.items()], {}

    async def get_workflow_inputs(self, workflow_id, auth_token=None):
        return copy.deepcopy(SPECS[workflow_id][1])

    async def dry_run(self, workflow_id, inputs, auth_token=None):
        self.dry_runs.append((workflow_id, inputs))
        why = self.refuse.get(inputs.get("output_file"))
        if why:
            return {"inputs_valid": False, "errors": [{"message": why}]}
        return {"valid": False, "inputs_valid": True}


async def _batch(fake=None, **kwargs):
    fake = fake or FakeGoWe()
    with patch.object(gowe, "_get_client", lambda config=None: fake):
        kwargs.setdefault("config", _cfg())
        out = await batch.prepare_batch(**kwargs)
    return out, fake


PAIRS = [
    {"sample_id": "EC1", "read1": f"{READS}/EC1_R1.fq.gz", "read2": f"{READS}/EC1_R2.fq.gz"},
    {"sample_id": "EC2", "read1": f"{READS}/EC2_R1.fq.gz", "read2": f"{READS}/EC2_R2.fq.gz"},
]


@pytest.mark.asyncio
async def test_one_job_per_pair_with_the_template_shared_and_untouched():
    template = {"recipe": "unicycler", "paired_end_libs": [{"read1": "/x", "read2": "/y", "platform": "illumina"}],
                "srr_ids": ["SRR9999999"], "output_path": "Ecoli_asm", "output_file": "ecoli"}
    before = copy.deepcopy(template)
    out, fake = await _batch(workflow_id="GenomeAssembly", template_inputs=template, samples=PAIRS)
    assert template == before
    jobs = [a["arguments"] for a in out["prepared_actions"]]
    assert [j["workflow_id"] for j in jobs] == ["wf_asm", "wf_asm"]
    assert jobs[0]["inputs"] == {
        "recipe": "unicycler",
        # platform comes from the template's record; its reads do not.
        "paired_end_libs": [{"platform": "illumina", "read1": PAIRS[0]["read1"], "read2": PAIRS[0]["read2"]}],
        "output_path": f"{CHAT}/Ecoli_asm", "output_file": "ecoli_EC1",
    }
    assert "srr_ids" not in jobs[1]["inputs"]       # no other sample's (or the template's) reads
    assert jobs[1]["inputs"]["output_file"] == "ecoli_EC2"
    a = out["prepared_actions"][0]
    assert a["tool"] == "submit_gowe_job" and a["origin"] == "prepare_batch" and a["samples"] == ["EC1"]
    assert a["summary"] == "GenomeAssembly ← EC1 (EC1_R1.fq.gz + EC1_R2.fq.gz)"
    assert len(fake.dry_runs) == 2


@pytest.mark.asyncio
async def test_sra_samples_use_the_declared_sra_input():
    sra = [{"sample_id": "ERR7916233", "srr_accession": "ERR7916233"}]
    asm, _ = await _batch(workflow_id="GenomeAssembly", template_inputs={}, samples=sra)
    assert asm["prepared_actions"][0]["arguments"]["inputs"]["srr_ids"] == ["ERR7916233"]
    tc, _ = await _batch(workflow_id="Taxonomic Classification", template_inputs={"database": "standard"}, samples=sra)
    assert tc["prepared_actions"][0]["arguments"]["inputs"]["srr_libs"] == [
        {"srr_accession": "ERR7916233", "sample_id": "ERR7916233"}]


@pytest.mark.asyncio
async def test_a_single_record_input_gets_an_object_and_one_job_cannot_hold_two():
    out, _ = await _batch(workflow_id="Metagenomic Binning", template_inputs={}, samples=PAIRS)
    assert out["prepared_actions"][0]["arguments"]["inputs"]["paired_end_libs"] == {
        "read1": PAIRS[0]["read1"], "read2": PAIRS[0]["read2"]}
    together, _ = await _batch(workflow_id="Metagenomic Binning", template_inputs={}, samples=PAIRS, one_job=True)
    assert len(together["prepared_actions"]) == 1
    assert together["skipped"] == [{"sample": "EC2", "reason": "paired_end_libs takes one library per job, "
                                    "so these samples cannot share a job"}]


@pytest.mark.asyncio
async def test_one_job_puts_every_sample_in_one_job():
    mixed = PAIRS + [{"srr_accession": "SRR1234567"}]
    out, _ = await _batch(workflow_id="Taxonomic Classification", template_inputs={"output_file": "gut"},
                          samples=mixed, one_job=True)
    (action,) = out["prepared_actions"]
    inputs = action["arguments"]["inputs"]
    assert [r["sample_id"] for r in inputs["paired_end_libs"]] == ["EC1", "EC2"]
    assert inputs["srr_libs"] == [{"srr_accession": "SRR1234567", "sample_id": "SRR1234567"}]
    assert inputs["output_file"] == "gut" and action["samples"] == ["EC1", "EC2", "SRR1234567"]


@pytest.mark.asyncio
async def test_more_than_ten_jobs_prepares_nothing():
    many = [{"srr_accession": f"SRR10000{i:02d}"} for i in range(11)]
    out, fake = await _batch(workflow_id="GenomeAssembly", template_inputs={}, samples=many)
    assert out["errorType"] == "TOO_MANY" and out["job_count"] == 11
    assert "prepared_actions" not in out and fake.dry_runs == []


@pytest.mark.asyncio
async def test_refused_bad_and_unusable_samples_are_skipped_with_reasons():
    fake = FakeGoWe(refuse={"GenomeAssembly_EC2": "read2 not found"})
    out, _ = await _batch(
        fake=fake, workflow_id="GenomeAssembly", template_inputs={"pilon_iter": 2},
        samples=PAIRS + [{"srr_accession": "SRR12"}, {"name": "what"}],
    )
    assert [a["samples"] for a in out["prepared_actions"]] == [["EC1"]]
    reasons = {str(s["sample"]): s["reason"] for s in out["skipped"]}
    assert "read2 not found" in reasons["EC2"]
    assert "not a valid SRA run accession" in reasons["SRR12"]
    assert "needs read1+read2" in reasons[str({"name": "what"})]
    # An input the workflow does not declare is never sent.
    assert "pilon_iter" not in out["prepared_actions"][0]["arguments"]["inputs"]
    assert out["dropped_inputs"] == ["pilon_iter"]


@pytest.mark.asyncio
async def test_duplicate_samples_collapse_and_sample_ids_stay_unique():
    dup = [dict(PAIRS[0]), dict(PAIRS[0]), {**PAIRS[1], "sample_id": "EC1"}]
    out, _ = await _batch(workflow_id="GenomeAssembly", template_inputs={}, samples=dup)
    assert [a["samples"] for a in out["prepared_actions"]] == [["EC1"], ["EC1_2"]]


@pytest.mark.asyncio
async def test_experimental_workflow_and_missing_organism_stop_before_preparing():
    exp, fake = await _batch(workflow_id="Tree Sort", template_inputs={}, samples=PAIRS)
    assert exp["errorType"] == "EXPERIMENTAL" and fake.dry_runs == []
    cga, _ = await _batch(workflow_id="ComprehensiveGenomeAnalysis", template_inputs={}, samples=PAIRS)
    assert cga["errorType"] == "MISSING_INPUT" and cga["needs_user_input"] is True
    ok, _ = await _batch(workflow_id="ComprehensiveGenomeAnalysis",
                         template_inputs={"scientific_name": "Escherichia coli", "taxonomy_id": 562},
                         samples=PAIRS)
    assert len(ok["prepared_actions"]) == 2


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def test_prepare_batch_is_service_only_and_list_samples_is_for_everyone():
    assert TOOL_AGENT_ALLOWLIST["prepare_batch"] == frozenset({"service"})
    assert HANDOFF_TARGETS["prepare_batch"] == "service"
    assert "list_samples" not in TOOL_AGENT_ALLOWLIST
    for name in ("list_samples", "prepare_batch"):
        assert name in TOOL_TIMEOUT_OVERRIDES and name in TOOL_RESULT_CHAR_LIMITS


@pytest.mark.asyncio
async def test_planner_calling_prepare_batch_hands_off_to_service():
    cfg = SimpleNamespace(agent_key="planning", handoff_from=None, execution_mode="plan")
    tool = AsyncMock()
    out = await execute_tool("prepare_batch", {"workflow_id": "GenomeAssembly", "template_inputs": {},
                                               "samples": []}, {"prepare_batch": tool}, config=cfg)
    tool.assert_not_awaited()
    assert out["handoff_to"] == "service"


def test_record_execution_keeps_the_batch_samples_on_blocked_actions():
    state = BaseAgentState(query="q")
    action = {"tool": "submit_gowe_job", "summary": "GenomeAssembly ← EC1", "origin": "prepare_batch",
              "samples": ["EC1"], "arguments": {"workflow_id": "wf_asm", "inputs": {}}}
    state.record_execution(tc=ToolCall(id="t1", name="prepare_batch", arguments={}),
                           result={"prepared_actions": [action]}, error=None, duration_ms=1)
    assert state.blocked_actions == [action]


def test_list_samples_folder_is_a_step_output():
    out = build_step_outputs([{"tool": "list_samples", "arguments": {"path": READS},
                               "result": {"paired": [], "sample_count": 0}}])
    assert out["paths"] == [READS]


# ---------------------------------------------------------------------------
# Service agent
# ---------------------------------------------------------------------------

BATCH_ACTIONS = [
    {"tool": "submit_gowe_job", "summary": f"GenomeAssembly ← EC{i}", "origin": "prepare_batch", "samples": [f"EC{i}"],
     "arguments": {"workflow_id": "wf_asm", "inputs": {"paired_end_libs": [{"read1": f"/r{i}_1", "read2": f"/r{i}_2"}],
                                                     "output_path": f"{CHAT}/asm", "output_file": f"asm_EC{i}"}}}
    for i in (1, 2)
]


def _reply(content="", tool_calls=None):
    calls = [SimpleNamespace(id=f"c{i}", type="function",
                             function=SimpleNamespace(name=n, arguments=json.dumps(a)))
             for i, (n, a) in enumerate(tool_calls or [])]
    msg = SimpleNamespace(content=content, tool_calls=calls or None, reasoning=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])


def _agent_cfg(mode):
    return AgentConfig(llm_base_url="http://llm", llm_api_key="k", llm_model="m",
                       execution_mode=mode, max_iterations=20)


@pytest.mark.asyncio
async def test_populate_batch_result_ends_the_turn_with_the_question():
    state = AgentState(query="assemble every pair in reads", context={"execution_mode": "plan"}, multi_job=True)
    chat = AsyncMock(side_effect=[
        _reply(tool_calls=[("prepare_batch", {"workflow_id": "GenomeAssembly", "template_inputs": {},
                                              "samples": []})]),
        _reply("Prepared 2 jobs. Answer **Yes** below to submit them."),
    ])
    submit = AsyncMock(side_effect=AssertionError("must not submit"))
    with patch.object(populate, "create_client", lambda c: MagicMock()), \
         patch.object(populate, "chat_completion", chat), \
         patch.dict(populate.TOOL_DISPATCH, {"prepare_batch": AsyncMock(return_value={"prepared_actions": BATCH_ACTIONS}),
                                             "submit_gowe_job": submit}):
        out = await populate.populate_and_submit(state.query, _agent_cfg("plan"), state)
    assert out.status == "completed" and out.current_phase == "ready"
    assert [a["samples"] for a in out.blocked_actions] == [["EC1"], ["EC2"]]
    assert chat.await_args_list[-1].kwargs["tool_choice"] == "none"


@pytest.mark.asyncio
async def test_a_one_job_batch_in_execute_mode_still_asks():
    """Phase 1 submits a lone prepared job in Execute mode; a batch does not:
    its reads were chosen and paired by code, not named by the user."""
    replies = [_reply(tool_calls=[("prepare_batch", {"workflow_id": "GenomeAssembly", "template_inputs": {},
                                                     "samples": []})]),
               _reply("Prepared 1 job. Answer **Yes** below to submit it.")]
    submit = AsyncMock(side_effect=AssertionError("must not submit"))
    with patch.object(populate, "create_client", lambda c: MagicMock()), \
         patch.object(populate, "chat_completion", AsyncMock(side_effect=replies)), \
         patch.dict(populate.TOOL_DISPATCH, {"prepare_batch": AsyncMock(return_value={"prepared_actions": BATCH_ACTIONS[:1]}),
                                             "submit_gowe_job": submit}), \
         patch.dict(replay_mod.TOOL_DISPATCH, {"submit_gowe_job": submit}), \
         patch.object(service_agent, "classify_intent", AsyncMock(return_value=Intent(job_count="many"))):
        result = await service_agent.run_agent(
            "assemble every pair in reads", config=_agent_cfg("execute"), context={"execution_mode": "execute"})
    submit.assert_not_awaited()
    assert result.submission_ids == [] and len(result.blocked_actions) == 1


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def test_prompts_send_batches_to_the_service_agent_tools():
    from orchestrator.router.prompts import ROUTING_SYSTEM_PROMPT
    from planning_agent.prompts.system import SYSTEM_PROMPT as PLANNING_SYSTEM_PROMPT
    from service_agent.classifier import CLASSIFIER_SYSTEM_PROMPT
    from service_agent.prompts.populate import build_populate_prompt

    flat = " ".join(ROUTING_SYSTEM_PROMPT.split())
    assert "Route to the **service** agent when the user wants to run the SAME workflow/service on MULTIPLE samples" in flat
    assert "(batch operation), route to **planning**" not in flat
    assert "detect paired-end patterns" not in PLANNING_SYSTEM_PROMPT
    assert "call prepare_batch right away" in " ".join(PLANNING_SYSTEM_PROMPT.split())
    assert "every read pair in this folder" in " ".join(CLASSIFIER_SYSTEM_PROMPT.split())
    prompt = build_populate_prompt()
    assert "== BATCH ==" in prompt and "list_samples" in prompt and "prepare_batch" in prompt
