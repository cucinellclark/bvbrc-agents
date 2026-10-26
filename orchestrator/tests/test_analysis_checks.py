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
from bvbrc_mcp.tools.workspace import compact_workspace_item  # noqa: E402
import bvbrc_mcp.tools.analysis as analysis_tools  # noqa: E402
from bvbrc_mcp.tools.output_knowledge import (  # noqa: E402
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




















# ---------------------------------------------------------------- knowledge










