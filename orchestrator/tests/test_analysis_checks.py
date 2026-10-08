"""Analysis-agent fixes from the 2026-10-08 baseline
(current_plans/analysis_and_literature.md in the docs repo):

- token estimate that holds for tables/FASTA, and last-resort trimming of an
  oversized latest exchange (ComparativeSystems turn died with HTTP 400);
- compact workspace_browse items (a folder of job results showed the LLM ~1
  item name; failed jobs were indistinguishable);
- get_job_details output_check (a failed job listed as "completed" was
  analysed from a previous run's files);
- output_knowledge table.
No network: workspace calls are mocked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for p in (str(_REPO_ROOT), str(_REPO_ROOT / "agents")):
    if p not in sys.path:
        sys.path.insert(0, p)

from shared.agent_utils import estimate_tokens, trim_messages_to_fit  # noqa: E402
from shared.tools.workspace import compact_workspace_item  # noqa: E402
import analysis_agent.tools as analysis_tools  # noqa: E402
from analysis_agent.output_knowledge import (  # noqa: E402
    GENERAL_RULES,
    SERVICE_KNOWLEDGE,
    build_service_knowledge_text,
    get_expected_outputs,
    resolve_output_path,
)


# ---------------------------------------------------------------- tokens


def test_estimate_tokens_counts_digits_individually():
    # A DESeq2-like table measured at ~1.1 chars/token on Qwen; the old
    # len/4 estimate was ~4x low.
    table = "\n".join(f"gene_{i}\t{1234.5678 + i}\t-2.3456\t1.2e-05" for i in range(500))
    assert estimate_tokens(table) > len(table) / 2


def test_estimate_tokens_prose_close_to_len_over_4():
    prose = "The assembly produced a contiguous genome with few contigs and good depth. " * 40
    est = estimate_tokens(prose)
    assert len(prose) / 5 < est < len(prose) / 2.5


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0


def _tool_exchange(i, content):
    return [
        {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function",
                                                            "function": {"name": "read_file_preview", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": f"c{i}", "content": content},
    ]


def test_trim_cuts_oversized_latest_exchange():
    """Parallel calls in ONE response returning large results: nothing older
    to compress, so the newest results must be cut instead of overflowing."""
    big = json.dumps({"data": "\t".join(str(x) for x in range(20000))})
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": f"c{i}", "type": "function", "function": {"name": "search_file", "arguments": "{}"}}
                for i in range(4)]}]
    msgs += [{"role": "tool", "tool_call_id": f"c{i}", "content": big} for i in range(4)]
    out = trim_messages_to_fit(msgs, None, max_tokens=30000)
    from shared.agent_utils import estimate_messages_tokens
    assert estimate_messages_tokens(out) <= 30000
    assert any("CUT TO FIT CONTEXT" in m.get("content", "") for m in out if m["role"] == "tool")
    # tool-call / tool-result pairing preserved
    assert [m["tool_call_id"] for m in out if m["role"] == "tool"] == ["c0", "c1", "c2", "c3"]


def test_trim_noop_under_budget():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}] + _tool_exchange(0, "small")
    assert trim_messages_to_fit(msgs, None, 80000) == msgs


# ---------------------------------------------------------------- browse items

APP_SPEC = {"id": "Homology", "parameters": [{"id": f"p{i}", "label": "x" * 100} for i in range(30)]}


def _ls_tuple(name, typ, parent="/u@p.org/home/Blast/", created="2026-09-18T18:49:41Z", size=95,
              user_meta=None, auto_meta=None):
    return [name, typ, parent, created, "ID", "u@p.org", size, user_meta or {}, auto_meta or {},
            "o", "n", "https://p3.theseed.org/services/shock_api/node/abc"]


def test_compact_job_result_keeps_success_drops_spec_and_shock_url():
    item = _ls_tuple(
        "Brucella_Omp31_blastp", "job_result",
        user_meta={"task_data": {"success": 0, "task_id": "23589316", "app_id": "Homology",
                                 "start_time": 1765217877.6, "end_time": 1765217897.3}},
        auto_meta={"app": APP_SPEC, "parameters": {"a": "b" * 500}},
    )
    out = compact_workspace_item(item)
    assert out["path"] == "/u@p.org/home/Blast/Brucella_Omp31_blastp"
    assert out["output_folder"] == "/u@p.org/home/Blast/.Brucella_Omp31_blastp"
    assert out["job"]["success"] == 0 and out["job"]["task_id"] == "23589316"
    assert out["job"]["start_time"].endswith("Z")
    blob = json.dumps(out)
    assert "shock_api" not in blob and "label" not in blob
    assert len(blob) < 600


def test_compact_plain_file_and_passthrough():
    out = compact_workspace_item(_ls_tuple("blast_out.txt", "txt", size=2793))
    assert out == {"name": "blast_out.txt", "type": "txt", "path": "/u@p.org/home/Blast/blast_out.txt",
                   "creation_time": "2026-09-18T18:49:41Z", "size": 2793}
    assert compact_workspace_item({"already": "dict"}) == {"already": "dict"}


async def test_workspace_browse_compacts_items(monkeypatch):
    from shared.tools import workspace as ws

    class FakeWs:
        async def workspace_browse(self, **kw):
            return {"result": {"items": [_ls_tuple("a.txt", "txt")], "count": 1}}

    monkeypatch.setattr(ws, "get_workspace_functions", lambda *a, **k: FakeWs())
    monkeypatch.setattr(ws, "_get_api", lambda config=None: object())
    r = await ws.workspace_browse(path="/u@p.org/home/Blast", headers={"Authorization": "un=u@p.org|x"})
    assert r["result"]["items"][0]["name"] == "a.txt"


# ---------------------------------------------------------------- output_check

JOB = {"id": 21474665, "status": "completed", "start_time": "2026-02-15 03:48:39",
       "parameters": {"output_path": "/u/home/TC", "output_file": "out"}}


def _browse_factory(parent_items, folder_items):
    async def fake_browse(path=None, **kw):
        if path == "/u/home/TC":
            return {"result": {"items": parent_items}}
        if path == "/u/home/TC/.out":
            if folder_items is None:
                return {"errorType": "PATH_NOT_FOUND", "error": "not found"}
            return {"result": {"items": folder_items}}
        raise AssertionError(path)
    return fake_browse


def _jr(task_id="21474665", success=1):
    return {"name": "out", "type": "job_result", "job": {"task_id": task_id, "success": success}}


def _f(name, created, typ="txt"):
    return {"name": name, "type": typ, "path": f"/u/home/TC/.out/{name}", "creation_time": created}


async def _check(monkeypatch, parent, folder, job=JOB):
    monkeypatch.setattr(analysis_tools, "workspace_browse", _browse_factory(parent, folder))
    return await analysis_tools.check_job_outputs(job)


async def test_check_ok(monkeypatch):
    c = await _check(monkeypatch, [_jr()], [_f("summary.csv", "2026-02-15T03:57:00Z")])
    assert c["verdict"] == "ok"


async def test_check_failed_success_zero_with_stale_files(monkeypatch):
    """The real case: success=0, JobFailed.html from this run, 19 files from 4 days earlier."""
    folder = [_f(f"old{i}.csv", "2026-02-11T18:41:21Z") for i in range(19)]
    folder.append(_f("JobFailed.html", "2026-02-15T03:57:50Z"))
    c = await _check(monkeypatch, [_jr(success=0)], folder)
    assert c["verdict"] == "failed"
    assert c["stale_files"] == 19
    assert c["job_failed_report"].endswith("JobFailed.html")
    assert "FAILED" in c["message"]


async def test_check_old_jobfailed_is_stale_not_failure(monkeypatch):
    folder = [_f("JobFailed.html", "2026-02-01T00:00:00Z"), _f("new.csv", "2026-02-15T03:50:00Z")]
    c = await _check(monkeypatch, [_jr()], folder)
    assert c["verdict"] == "stale_files" and "job_failed_report" not in c


async def test_check_other_run_owns_job_result(monkeypatch):
    c = await _check(monkeypatch, [_jr(task_id="99999")], [_f("a", "2026-02-15T03:50:00Z")])
    assert c["verdict"] == "outputs_from_other_run" and "99999" in c["message"]


async def test_check_old_subfolder_not_stale(monkeypatch):
    folder = [_f("load_files", "2026-01-01T00:00:00Z", typ="folder"), _f("a", "2026-02-15T03:50:00Z")]
    c = await _check(monkeypatch, [_jr()], folder)
    assert c["verdict"] == "ok"


async def test_check_clock_slack(monkeypatch):
    c = await _check(monkeypatch, [_jr()], [_f("a", "2026-02-15T03:45:00Z")])  # 3.5 min before start
    assert c["verdict"] == "ok"


async def test_check_missing_folder(monkeypatch):
    c = await _check(monkeypatch, [], None)
    assert c["verdict"] == "no_outputs"


async def test_check_no_output_params(monkeypatch):
    assert await analysis_tools.check_job_outputs({"id": 1, "parameters": {}}) is None


async def test_get_job_details_attaches_output_check(monkeypatch):
    class FakeService:
        async def query_tasks(self, **kw):
            return {"data": [{"21474665": dict(JOB)}], "source": "bvbrc-service"}

    import shared.tools._mcp_imports as mcp_imports
    monkeypatch.setattr(mcp_imports, "get_service_functions", lambda: FakeService())
    monkeypatch.setattr(analysis_tools, "_get_service_api", lambda: object())
    monkeypatch.setattr(analysis_tools, "workspace_browse",
                        _browse_factory([_jr(success=0)], [_f("JobFailed.html", "2026-02-15T03:57:50Z")]))
    r = await analysis_tools.get_job_details(task_ids=["21474665"], headers={"Authorization": "tok"})
    assert r["data"][0]["21474665"]["output_check"]["verdict"] == "failed"


# ---------------------------------------------------------------- knowledge


def test_verified_services_have_files_and_checks():
    for name, entry in SERVICE_KNOWLEDGE.items():
        if entry["verified"]:
            assert entry["read_first"], name
            assert entry["checks"], name
            assert entry["next_steps"], name


def test_get_expected_outputs_shapes():
    r = get_expected_outputs("ComparativeSystems")
    assert r["known"] and any("pathways.tsv" in k for k in r["avoid"])
    assert get_expected_outputs("NoSuchApp")["known"] is False


def test_amr_rules_reach_annotation_services():
    for svc in ("GenomeAnnotation", "ComprehensiveGenomeAnalysis"):
        text = " ".join(get_expected_outputs(svc)["checks"])
        assert "NOT resistance genes" in text and "IDENTITY CHECK" in text


def test_resolve_output_folder_template():
    t = "${output_folder}/details/${params.output_file}_run_details.json"
    assert resolve_output_path(t, "/u/home/A", "S1") == "/u/home/A/.S1/details/S1_run_details.json"


def test_prompt_block_contains_rules_and_stays_small():
    text = build_service_knowledge_text()
    assert GENERAL_RULES[0][:40] in text
    assert len(text) < 5000
