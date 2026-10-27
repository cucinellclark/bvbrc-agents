"""Structured progress steps (testing session 2026-10-16, #5).

Every tool call and model call a loop makes is reported as a step dict the
UI can render in any shape; the model's one-line intent becomes the step
label; retries and compaction are notes.  ``shared/progress.py``.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import shared.agent_dispatch  # noqa: E402, F401  -- sets up agent package paths
from shared import progress as prog  # noqa: E402
from shared.agent_loop import run_agent_loop  # noqa: E402
from shared.agent_utils import emit_progress  # noqa: E402
from shared.models import BaseAgentConfig, BaseAgentState  # noqa: E402
from shared.tools.schemas import ALL_TOOL_SCHEMAS  # noqa: E402

# ---------------------------------------------------------------------------
# Describers
# ---------------------------------------------------------------------------


def test_every_tool_has_a_code_description_without_its_name():
    names = [s.get("function", s)["name"] for s in ALL_TOOL_SCHEMAS]
    assert len(names) == 32
    for name in names:
        label, _ = prog.describe_tool_call(name, {})
        assert label and not label.startswith("Running "), name
        assert name not in label, name
        status, text = prog.describe_result(name, {}, {"anything": 1})
        assert status == "done" and text


def test_search_label_reads_as_words():
    label, detail = prog.describe_tool_call("search_data", {
        "collection": "genome", "query": 'genome_name:"Streptococcus pneumoniae" AND (antibiotic:tetracycline OR resistant_phenotype:Resistant)'})
    assert label == "Searching genome"
    assert detail == "genome name: Streptococcus pneumoniae, antibiotic: tetracycline or resistant phenotype: Resistant"
    assert '"' not in detail and "(" not in detail


def test_other_users_workspaces_and_group_paths():
    assert prog.describe_tool_call("workspace_browse", {"scope": "shared"}) == ("Listing workspaces shared with you", None)
    label, detail = prog.describe_tool_call("workspace_browse", {"scope": "public", "name_contains": ["NVDDTHD"]})
    assert label == "Listing public workspaces" and "NVDDTHD" in detail
    assert prog.describe_result("workspace_browse", {"scope": "public"}, {"scope": "public", "result_type": "workspaces", "items": [{}, {}]})[1] == "2 workspaces"
    label, _ = prog.describe_tool_call("get_group_ids", {"group_name": "/allan@patricbrc.org/AllanShare/Streptococcus public"})
    assert label == "Reading group 'Streptococcus public'"


def test_count_only_and_facets():
    assert prog.describe_tool_call("search_data", {"collection": "genome_amr", "query": "*:*", "count_only": True}) == ("Counting genome amr records", "everything")
    label, _ = prog.describe_tool_call("facet_query", {"collection": "genome", "query": "genus:Vibrio", "facet_fields": ["species", "genome_status"]})
    assert label == "Counting genome by species, genome status"


def test_file_tools_show_the_file_name_not_the_path():
    label, detail = prog.describe_tool_call("read_file_preview", {"path": "/u@patricbrc.org/home/chats/x/report.tsv", "start_byte": 65536})
    assert label == "Reading report.tsv" and detail == "page 3"
    label, _ = prog.describe_tool_call("search_file", {"path": "/a/b/c.fasta", "pattern": "gyrA"})
    assert label == "Searching c.fasta for 'gyrA'"
    assert prog.describe_tool_call("workspace_browse", {}) == ("Listing your home folder", None)


def test_workspace_browse_says_listing_or_searching_and_how_many_of_how_many():
    assert prog.describe_tool_call("workspace_browse", {"path": "Assembly", "recursive": True})[0] == "Searching Assembly"
    assert prog.describe_tool_call("workspace_browse", {"file_extensions": ["fastq"], "recursive": False})[0] == "Listing your home folder"
    # Filters without recursive: the tool searches, so the label says so.
    assert prog.describe_tool_call("workspace_browse", {"file_extensions": ["fastq"]})[0] == "Searching your home folder"
    page = {"items": [{}] * 50, "count": 50, "total": 812}
    assert prog.describe_result("workspace_browse", {}, page) == ("done", "50 of 812 items")
    assert prog.describe_result("workspace_browse", {}, {"items": [{}, {}], "total": 2}) == ("done", "2 items")
    assert prog.describe_result("workspace_browse", {"scope": "public"},
                                {"scope": "public", "result_type": "workspaces", "items": [{}], "total": 13})[1] == "1 of 13 workspaces"
    # Part 3: one search inside every shared workspace; several folders at once.
    label, detail = prog.describe_tool_call("workspace_browse", {"scope": "shared", "recursive": True, "name_contains": ["strep"]})
    assert label == "Searching the workspaces shared with you" and "strep" in detail
    assert prog.describe_tool_call("workspace_browse", {"paths": ["/a/b", "/c/d"], "recursive": True})[0] == "Searching 2 folders"
    assert prog.describe_tool_call("workspace_browse", {"paths": ["/a/Reads"]})[0] == "Listing Reads"
    found = {"scope": "shared", "result_type": "search_result", "items": [{}] * 3, "total": 3}
    assert prog.describe_result("workspace_browse", {}, found) == ("done", "3 items")
    assert prog.describe_result("list_groups", {}, {"groups": ["a"] * 300, "total": 320}) == ("done", "300 of 320 groups")
    # Filter / sort review: any-of names, job filters and dates read as words.
    label, detail = prog.describe_tool_call("workspace_browse", {"name_contains": ["flu", "influenza"], "name_match": "any",
                                                                 "created_after": "2026-10-01"})
    assert label == "Searching your home folder" and detail.startswith("names containing any of") and detail.endswith("since 2026-10-01")
    assert prog.describe_result("list_jobs", {}, {"items": [{}] * 20, "total": 156}) == ("done", "20 of 156 jobs")


def test_job_tools_use_the_workflow_name_when_known():
    from bvbrc_mcp.tools import gowe
    with patch.dict(gowe._workflow_names, {"wf_1": "Genome Assembly"}):
        label, detail = prog.describe_tool_call("submit_gowe_job", {"workflow_id": "wf_1", "inputs": {"srr_ids": ["SRR1", "SRR2"]}})
        assert label == "Submitting a Genome Assembly job" and detail == "SRR1, SRR2"
        assert prog.describe_tool_call("get_workflow_inputs", {"workflow_id": "wf_1"})[0] == "Checking the inputs of Genome Assembly"
    assert "wf_9" not in prog.describe_tool_call("get_workflow_inputs", {"workflow_id": "wf_9"})[0]


def test_results_are_counts_or_short_errors():
    assert prog.describe_result("search_data", {}, {"numFound": 1234, "results": [1, 2]}) == ("done", "1,234 found, showing 2")
    assert prog.describe_result("facet_query", {}, {"distinct_counts": {"genome_id": 57}}) == ("done", "57 distinct genome id")
    assert prog.describe_result("workspace_browse", {}, {"items": [1, 2, 3]}) == ("done", "3 items")
    assert prog.describe_result("search_file", {}, {"matches": [1], "next_start": 100}) == ("done", "1 match, more to scan")
    assert prog.describe_result("list_samples", {}, {"paired": [1, 2], "single": [], "sra": [1], "unpaired": []}) == ("done", "2 pairs, 1 SRA run")
    assert prog.describe_result("get_file_metadata", {}, {"errorType": "PATH_NOT_FOUND", "error": "Object not found"}) == ("error", "not found")
    assert prog.describe_result("search_data", {}, {"error": "Solr timed out\nstack"}) == ("error", "Solr timed out")
    assert prog.describe_result("submit_gowe_job", {}, {"blocked_by_mode": True, "error": "x"}) == ("done", "prepared, waiting for your answer")
    assert prog.describe_result("create_group", {}, {"handoff_to": "data"}) == ("done", "handing over to: Searching BV-BRC data")
    assert prog.describe_result("rerun_jobs", {}, {"prepared_actions": [1, 2], "skipped": [1]}) == ("done", "2 jobs prepared, 1 skipped")


def test_intent_line_takes_one_short_sentence_only():
    assert prog.intent_line("Looking up tetracycline-resistant S. pneumoniae genomes.") == "Looking up tetracycline-resistant S. pneumoniae genomes."
    assert prog.intent_line("<think>long</think>**Checking** your workspace for FASTQ files\nmore") == "Checking your workspace for FASTQ files"
    assert prog.intent_line(None) is None
    assert prog.intent_line("") is None
    assert prog.intent_line('{"a": 1}') is None
    assert prog.intent_line("x " * 150) is None               # an answer draft, not an intent
    assert prog.intent_line("Here is the plan:\n" + "step\n" * 200) is None


def test_agent_labels_are_plain():
    assert prog.agent_label("data") == "Searching BV-BRC data"
    assert prog.agent_label("service") == "Preparing a job"
    assert prog.agent_label(None) is None


# ---------------------------------------------------------------------------
# Sink and legacy callback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_report_goes_to_the_installed_sink_with_the_run_clock():
    got = []

    async def sink(step):
        got.append(step)

    tokens = prog.start_run(sink, "data")
    try:
        step = await prog.start("tool", "Searching genome", "genus: Vibrio", tool="search_data")
        await prog.done(step, "12 found")
        await emit_progress(None, 0, None, "Done.")       # legacy string path
    finally:
        prog.end_run(tokens)
    assert [s["status"] for s in got] == ["running", "done", "done"]
    assert got[0]["step_id"] == got[1]["step_id"]
    assert got[1]["result"] == "12 found" and got[1]["took_ms"] >= 0 and "ts_ms" in got[1]
    assert got[0]["agent_label"] == "Searching BV-BRC data"
    assert got[2]["transient"] is True and got[2]["label"] == "Done."
    assert not any(k.startswith("_") for s in got for k in s)
    await prog.note("after the run")                         # no sink: a no-op
    assert len(got) == 3


@pytest.mark.asyncio
async def test_without_a_sink_the_callback_gets_the_step():
    got = []

    async def cb(progress, total, step):
        got.append(step)

    await emit_progress(cb, 0, None, "Analyzing your question...")
    await prog.note("x", cb=cb)
    assert got[0]["label"] == "Analyzing your question..." and got[1]["label"] == "x"


# ---------------------------------------------------------------------------
# The shared loop
# ---------------------------------------------------------------------------

SEARCH = ("search_data", {"collection": "genome", "query": 'genus:"Vibrio"'})


def _reply(content="", calls=()):
    tcs = [SimpleNamespace(id=f"c{i}", type="function",
                           function=SimpleNamespace(name=n, arguments=json.dumps(a)))
           for i, (n, a) in enumerate(calls)]
    msg = SimpleNamespace(content=content, tool_calls=tcs or None, reasoning=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls" if tcs else "stop")])


def _cfg():
    return BaseAgentConfig(llm_base_url="http://llm", llm_api_key="k", llm_model="m", max_iterations=20)


async def _run(replies, tool_result=None):
    steps = []

    async def cb(progress, total, step):
        steps.append(dict(step))

    state = BaseAgentState(query="how many Vibrio genomes", context={})
    tool = AsyncMock(return_value=tool_result or {"numFound": 4321, "results": [1, 2, 3]})
    chat = AsyncMock(side_effect=list(replies))
    with patch.dict("shared.tools.registry.TOOL_DISPATCH", {"search_data": tool}):
        await run_agent_loop(state=state, config=_cfg(), client=MagicMock(), system_prompt="sys",
                             chat_completion_fn=chat, progress_callback=cb)
    return state, steps, chat


@pytest.mark.asyncio
async def test_a_turn_reports_thinking_tool_and_answer_steps():
    replies = [_reply("Looking up Vibrio genomes", calls=[SEARCH]), _reply("There are 4,321 Vibrio genomes.")]
    state, steps, chat = await _run(replies)
    rows = [(s["phase"], s["status"], s["label"]) for s in steps if not s.get("transient")]
    assert rows == [
        ("thinking", "running", prog.THINKING_FIRST),
        ("thinking", "done", prog.THINKING_FIRST),
        ("tool", "running", "Looking up Vibrio genomes"),      # the model's intent
        ("tool", "done", "Looking up Vibrio genomes"),
        ("thinking", "running", prog.THINKING_NEXT),
        ("thinking", "done", prog.THINKING_ANSWER),
    ]
    tool_done = [s for s in steps if s["phase"] == "tool" and s["status"] == "done"][0]
    assert tool_done["detail"] == "Searching genome — genus: Vibrio"   # the code description
    assert tool_done["result"] == "4,321 found, showing 3"
    assert tool_done["tool"] == "search_data"
    assert steps[0]["transient"] and steps[-1]["transient"]           # start / done messages
    assert chat.await_count == 2


@pytest.mark.asyncio
async def test_without_an_intent_the_code_description_is_the_label():
    replies = [_reply(calls=[SEARCH]), _reply("done")]
    _, steps, _ = await _run(replies)
    tool = [s for s in steps if s["phase"] == "tool"][0]
    assert tool["label"] == "Searching genome" and tool["detail"] == "genus: Vibrio"


@pytest.mark.asyncio
async def test_the_prompt_asks_for_the_intent_sentence():
    replies = [_reply("done")]
    _, _, chat = await _run(replies)
    system = chat.await_args_list[0].kwargs["messages"][0]["content"]
    assert "=== PROGRESS NOTE ===" in system


@pytest.mark.asyncio
async def test_a_repeated_call_is_a_note_and_the_forced_answer_is_a_step():
    replies = [_reply(calls=[SEARCH])] * 3 + [_reply("Could not finish.")]
    _, steps, _ = await _run(replies)
    labels = [s["label"] for s in steps if s["phase"] == "note" and not s.get("transient")]
    assert labels == [prog.NOTE_REPEAT_SKIPPED, prog.NOTE_REPEAT_SKIPPED, prog.NOTE_STUCK]
    answer = [s for s in steps if s["phase"] == "answer"]
    assert [s["status"] for s in answer] == ["running", "done"]
    assert answer[0]["detail"] == prog.DETAIL_FROM_SO_FAR


@pytest.mark.asyncio
async def test_a_tool_error_is_an_error_step():
    replies = [_reply(calls=[SEARCH]), _reply("The collection was not found.")]
    _, steps, _ = await _run(replies, tool_result={"error": "Unknown collection", "errorType": "INVALID_PARAMETERS"})
    tool = [s for s in steps if s["phase"] == "tool" and s["status"] != "running"][0]
    assert tool["status"] == "error" and tool["result"] == "invalid parameters"


# ---------------------------------------------------------------------------
# Retries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_timeout_retry_is_reported_as_a_note():
    import httpx
    from shared.agent_utils import llm_call_with_retry, RETRY_TIMEOUT_NOTE

    got = []

    async def sink(step):
        got.append(step)

    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("slow")
        return "ok"

    tokens = prog.start_run(sink, "data")
    try:
        with patch("shared.agent_utils.asyncio.sleep", new=AsyncMock()):
            assert await llm_call_with_retry(flaky) == "ok"
    finally:
        prog.end_run(tokens)
    assert [s["label"] for s in got] == [RETRY_TIMEOUT_NOTE]
    assert got[0]["phase"] == "note" and got[0]["status"] == "done"


# ---------------------------------------------------------------------------
# Orchestrator: the executor bridge and the routing step
# ---------------------------------------------------------------------------


def test_routing_step_labels():
    from orchestrator.orchestrate import _routing_done_label, _routing_step

    step = _routing_step("running", "Deciding how to handle your request")
    assert step["step_id"] == "routing" and step["phase"] == "routing" and step["message"] == step["label"]
    direct = SimpleNamespace(decision="direct", plan=None)
    assert _routing_done_label(direct) == "Answering directly"
    agent = SimpleNamespace(decision="agent", plan=SimpleNamespace(steps=[SimpleNamespace(agent_key="helpdesk")]))
    assert _routing_done_label(agent) == "Handing to: Looking up documentation"
