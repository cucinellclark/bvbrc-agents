"""Context budget: compaction, hard stop, overflow recovery (testing session #12).

acapria's data turn (2026-10-16) added ~3k tokens per search until call 23
overflowed and the whole turn was discarded; marcnguy's service turns grew to
76k-token prompts whose latency ate the 300 s limit.  Now older results are
compacted past the soft line, the hard line forces an answer, and an
overflow compacts and answers instead of failing.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import shared.agent_dispatch  # noqa: E402, F401  -- sets up agent package paths
from shared import context_budget as cb  # noqa: E402
from shared.agent_loop import run_agent_loop  # noqa: E402
from shared.agent_messages import BUDGET_ANSWER_NOW, BUDGET_FALLBACK, OVERFLOW_ANSWER_NOW  # noqa: E402
from shared.agent_utils import estimate_messages_tokens  # noqa: E402
from shared.models import BaseAgentConfig, BaseAgentState  # noqa: E402

OVERFLOW = Exception("Error code: 400 - This model's maximum context length is 131072 tokens. "
                     "However, you requested 134000 tokens. Please reduce the length of the messages.")


def _search_result(n_rows=25, seed=0):
    rows = [{"genome_id": f"1000{seed}{i:03d}.10", "genome_name": f"Rosellinia necatrix quadrivirus strain W{i}",
             "genome_status": "Complete", "family": "Quadriviridae", "genome_length": 4000 + i} for i in range(n_rows)]
    return {"results": rows, "count": n_rows, "numFound": 170617, "source": "bvbrc-mcp-data",
            "viewer_url": "https://www.bv-brc.org/view/GenomeList/?eq(family,Quadriviridae)",
            "download_tsv_url": "https://www.bv-brc.org/api-bulk/genome/?eq(family,Quadriviridae)&http_download=true"}


# ---- compaction -------------------------------------------------------------------

def test_a_search_result_keeps_counts_ids_links_and_a_sample():
    full = json.dumps(_search_result(), indent=2)
    compact = cb.compact_tool_result(full)
    data = json.loads(compact)
    assert len(compact) < len(full) / 3
    assert data["numFound"] == 170617 and data["count"] == 25
    assert data["viewer_url"].startswith("https://www.bv-brc.org/view/GenomeList/")
    assert len(data["results_sample"]) == 3 and data["results_returned"] == 25
    assert len(data["ids"]) == 25 and data["ids"][0] == "10000000.10"
    assert data[cb.COMPACT_MARK] is True and "rows dropped" in data["_note"]
    assert "results" not in data
    assert cb.compact_tool_result(compact) == compact  # idempotent


def test_small_results_text_facets_and_summaries():
    small = json.dumps({"numFound": 3, "results": [{"a": 1}]})
    assert cb.compact_tool_result(small) == small
    text = "x" * 5000
    assert cb.compact_tool_result(text).startswith("x" * 400) and "5000 chars" in cb.compact_tool_result(text)
    facets = json.dumps({"numFound": 9, "facets": {"family": [{"value": f"f{i}", "count": i} for i in range(20)]},
                         "bucket_counts": {"family": 20}, "pad": "p" * 700})
    data = json.loads(cb.compact_tool_result(facets))
    assert len(data["facets"]["family"]) == 10 and data["bucket_counts"] == {"family": 20}
    summary = json.dumps({"data": "L" * 2000, "_summary": {"path": "/u/a.tsv", "next_start": 32768}, "is_complete": False})
    data = json.loads(cb.compact_tool_result(summary))
    assert data["_summary"]["next_start"] == 32768 and data["is_complete"] is False and len(data["data"]) == 301
    ids = json.dumps([{"feature_id": f"fig|1.1.peg.{i}", "product": "hypothetical protein " * 3} for i in range(40)])
    data = json.loads(cb.compact_tool_result(ids))
    assert data["count"] == 40 and len(data["ids"]) == 25 and data["ids_truncated"] == 40


def test_a_nested_result_envelope_keeps_its_totals_summary_and_paging():
    # Some tool results nest everything under "result" (workspace_browse did
    # until 2026-10-27; the old MCP server's copies still do); compaction used to
    # replace the whole envelope with "[object with N fields omitted]".
    items = [{"name": f"f{i}.fastq", "type": "reads", "path": f"/u/home/f{i}.fastq", "size": 1} for i in range(50)]
    full = json.dumps({"result": {"items": items, "count": 50, "total": 812, "truncated": True,
                                  "next_offset": 50, "summary": {"by_type": {"reads": 800, "txt": 12}}},
                       "workspace_browser_url": "https://www.bv-brc.org/workspace/u/home"}, indent=2)
    data = json.loads(cb.compact_tool_result(full))
    inner = data["result"]
    assert inner["total"] == 812 and inner["next_offset"] == 50 and inner["truncated"] is True
    assert inner["summary"]["by_type"]["reads"] == 800
    assert inner["items_count"] == 50 and len(inner["items_sample"]) == 5
    assert data["workspace_browser_url"].endswith("/u/home")


def test_only_results_before_the_latest_exchange_are_compacted():
    big = json.dumps(_search_result())
    messages = [
        {"role": "system", "content": "sys"}, {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]}, {"role": "tool", "tool_call_id": "c1", "content": big},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c2"}]}, {"role": "tool", "tool_call_id": "c2", "content": big},
    ]
    assert cb.compact_older_results(messages) == 1
    assert json.loads(messages[3]["content"])[cb.COMPACT_MARK] is True
    assert messages[5]["content"] == big
    assert cb.compact_older_results(messages) == 0  # nothing new to do
    assert cb.compact_older_results(messages, include_latest=True) == 1
    assert json.loads(messages[5]["content"])[cb.COMPACT_MARK] is True


# ---- the budget -------------------------------------------------------------------

def test_budget_lines_and_calibration():
    b = cb.ContextBudget(100000, 40000)
    assert (b.usable, b.soft, b.hard) == (60000, 30000, 48000)
    assert cb.ContextBudget(20000, 19000).usable == cb.MIN_USABLE_TOKENS  # a wrong reservation cannot stop the loop
    cfg = BaseAgentConfig(llm_base_url="http://x", llm_api_key="k", llm_model="m", max_tokens=40000)
    assert cb.ContextBudget.for_config(cfg).window == cb.DEFAULT_CONTEXT_WINDOW
    cfg.context_window_tokens = 200000
    assert cb.ContextBudget.for_config(cfg).usable == 160000
    messages = [{"role": "user", "content": "hello " * 1000}]
    estimated = estimate_messages_tokens(messages, None)
    b.record(SimpleNamespace(usage=SimpleNamespace(prompt_tokens=estimated // 2)), messages, None)
    assert b.factor == 0.5 and b.measured == estimated // 2
    b.record({"usage": {"prompt_tokens": estimated * 10}}, messages, None)
    assert b.factor == 2.0  # clamped
    b.record(SimpleNamespace(usage=None), messages, None)
    assert b.factor == 2.0  # no usage: unchanged
    assert b.prepare(messages, None) == {"predicted": estimated * 2, "compacted": 0, "exhausted": False}


def test_prepare_compacts_past_soft_and_reports_the_hard_line():
    b = cb.ContextBudget(18000, 2000)  # usable 16000, soft 8000, hard 12800
    big = json.dumps(_search_result(60))
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
    for i in range(3):
        messages += [{"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}"}]},
                     {"role": "tool", "tool_call_id": f"c{i}", "content": big}]
    before = b.predict(messages, None)
    assert before > b.soft
    prep = b.prepare(messages, None)
    assert prep["compacted"] == 2 and prep["predicted"] < before
    assert json.loads(messages[3]["content"])[cb.COMPACT_MARK] and messages[7]["content"] == big
    huge = [{"role": "tool", "tool_call_id": "h", "content": json.dumps(_search_result(400))}]
    assert b.prepare(messages[:3] + huge, None)["exhausted"] is True


# ---- the shared loop ---------------------------------------------------------------

def _reply(content="", calls=(), messages=None, tools=None):
    tcs = [SimpleNamespace(id=f"c{i}", type="function", function=SimpleNamespace(name=n, arguments=json.dumps(a)))
           for i, (n, a) in enumerate(calls)]
    msg = SimpleNamespace(content=content, tool_calls=tcs or None, reasoning=None)
    usage = SimpleNamespace(prompt_tokens=estimate_messages_tokens(messages or [], tools)) if messages is not None else None
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls" if tcs else "stop")], usage=usage)


SCHEMA = [{"type": "function", "function": {"name": "search_data", "parameters": {"type": "object"}}}]


def _cfg(window, reserve=2000):
    return BaseAgentConfig(llm_base_url="http://llm", llm_api_key="k", llm_model="m", max_iterations=200,
                           max_tokens=reserve, context_window_tokens=window)


class _Model:
    """Keeps searching with a new query each call; answers once tools are off."""

    def __init__(self, rows=60, calls_per_response=1, fail_on=None, fail_times=1):
        self.n = 0
        self.rows = rows
        self.per = calls_per_response
        self.fail_on, self.fail_times, self.failed = fail_on, fail_times, 0
        self.tool_choices = []

    async def __call__(self, client, messages, tools, config, tool_choice=None):
        self.n += 1
        self.tool_choices.append(tool_choice)
        if self.fail_on and self.n >= self.fail_on and self.failed < self.fail_times:
            self.failed += 1
            raise OVERFLOW
        if tool_choice == "none":
            return _reply("Final answer: 170,617 complete genomes; smallest families listed.", messages=messages, tools=tools)
        calls = [("search_data", {"collection": "genome", "query": f"family:F{self.n}_{j}"}) for j in range(self.per)]
        return _reply(calls=calls, messages=messages, tools=tools)


async def _run(window, model, rows=60):
    state = BaseAgentState(query="pull small complete ones", context={})
    tool = AsyncMock(side_effect=lambda **kw: _search_result(rows, seed=len(state.tool_executions)))
    with patch.dict("shared.tools.registry.TOOL_DISPATCH", {"search_data": tool}):
        await run_agent_loop(state=state, config=_cfg(window), client=MagicMock(), system_prompt="sys",
                             tool_schemas=SCHEMA, chat_completion_fn=model, max_tool_result_chars=50000)
    return state, tool


async def test_a_long_turn_stays_under_the_line_by_compacting_older_results():
    # 24000 - 2000 = 22000 usable (soft 11000, hard 17600); a full 60-row
    # result is ~5k tokens, a compacted one ~800, so eleven results fit only
    # because the older ones are compacted.
    class Stop(_Model):
        async def __call__(self, *a, **kw):
            if self.n == 11:  # eleven searches made: answer on the twelfth call
                return _reply("Done: here is what the twelve searches found.", messages=kw["messages"], tools=kw["tools"])
            return await super().__call__(*a, **kw)

    model = Stop(rows=60)
    state, tool = await _run(24000, model)
    assert tool.await_count == 11 and state.status == "completed"
    assert state.final_answer.startswith("Done:")
    assert all(c is None for c in model.tool_choices)  # tools stayed on: the budget never forced the answer
    tool_msgs = [m for m in state.messages if m.get("role") == "tool"]
    compacted = [m for m in tool_msgs if cb.COMPACT_MARK in m["content"]]
    assert len(compacted) >= 8 and cb.COMPACT_MARK not in tool_msgs[-1]["content"]  # the latest stays full
    # Eleven full results would not have fit; the compacted history does.
    assert estimate_messages_tokens(state.messages, SCHEMA) < cb.ContextBudget(24000, 2000).hard
    assert 11 * estimate_messages_tokens([tool_msgs[-1]], None) > cb.ContextBudget(24000, 2000).usable


async def test_an_exchange_too_big_for_the_window_forces_an_answer_with_tools_off():
    model = _Model(rows=400, calls_per_response=3)  # one response, three huge results
    state, tool = await _run(18000, model)
    assert tool.await_count == 3
    assert model.tool_choices[-1] == "none" and model.n == 2
    assert state.status == "completed" and state.final_answer.startswith("Final answer")
    assert state.messages[-1]["content"] == BUDGET_ANSWER_NOW  # the nudge is the last message before the answer


async def test_an_overflow_compacts_everything_and_still_answers():
    model = _Model(rows=60, fail_on=4)  # the 4th call is refused as too long
    state, tool = await _run(60000, model)
    assert tool.await_count == 3 and model.n == 5  # 3 searches, the refused call, the forced answer
    assert model.tool_choices[-1] == "none"
    assert state.status == "completed" and state.final_answer.startswith("Final answer")
    assert all(cb.COMPACT_MARK in m["content"] for m in state.messages if m.get("role") == "tool")
    assert any(m.get("content") == OVERFLOW_ANSWER_NOW for m in state.messages)


async def test_a_second_overflow_falls_back_to_the_list_of_lookups():
    model = _Model(rows=60, fail_on=4, fail_times=2)
    state, tool = await _run(60000, model)
    assert state.status == "completed"
    assert state.final_answer.startswith(BUDGET_FALLBACK.split("{")[0][:30])
    assert "`search_data`" in state.final_answer


# ---- the service populate loop -------------------------------------------------------

async def test_service_loop_turns_tools_off_when_the_budget_is_spent():
    from service_agent.models import AgentConfig, AgentState
    from service_agent.phases import populate

    # The populate system prompt alone is ~15k estimated tokens, so the window
    # must hold it (usable 30000, hard 24000).  The loop cuts each result to
    # ~8000 chars (~3.5k tokens), so one exchange of five passes the hard line.
    cfg = AgentConfig(llm_base_url="http://llm", llm_api_key="k", llm_model="m", max_iterations=30,
                      max_tokens=2000, context_window_tokens=32000)
    state = AgentState(query="build a tree of my two groups", context={})
    model = _Model(rows=400, calls_per_response=5)
    tool = AsyncMock(side_effect=lambda **kw: _search_result(400))
    with patch.object(populate, "create_client", lambda c: MagicMock()), \
         patch.object(populate, "chat_completion", model), \
         patch.object(populate, "POPULATE_TOOLS", SCHEMA), \
         patch.dict(populate.TOOL_DISPATCH, {"search_data": tool}):
        out = await populate.populate_and_submit(state.query, cfg, state)
    assert tool.await_count == 5 and model.tool_choices[-1] == "none"
    assert out.status != "error"
    # _add_nudge wraps the text; the phrase is enough to show which nudge it was.
    assert any("room available for this turn" in str(m.get("content")) for m in out.messages if m.get("role") == "user")


# ---- the window lookup ---------------------------------------------------------------

async def test_context_window_comes_from_the_model_server_and_is_cached():
    cb._WINDOW_CACHE.clear()
    body = {"data": [{"id": "other", "max_model_len": 8192}, {"id": "Qwen3.8-27B", "max_model_len": 131072}]}
    response = MagicMock(status_code=200, json=lambda: body)
    response.raise_for_status = lambda: None
    client = MagicMock()
    client.get = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    with patch("httpx.AsyncClient", return_value=client):
        assert await cb.resolve_context_window("http://mango:8008/v1/", "k", "Qwen3.8-27B") == 131072
        assert await cb.resolve_context_window("http://mango:8008/v1", "k", "Qwen3.8-27B") == 131072
    assert client.get.await_count == 1  # cached
    assert client.get.await_args.args[0] == "http://mango:8008/v1/models"
    cb._WINDOW_CACHE.clear()
    client.get = AsyncMock(side_effect=OSError("down"))
    with patch("httpx.AsyncClient", return_value=client):
        assert await cb.resolve_context_window("http://mango:8008/v1", "k", "m") is None
    assert await cb.resolve_context_window(None, None, None) is None


# ---- the planning loop ----------------------------------------------------------------

async def test_planning_loop_may_only_plan_or_ask_once_the_budget_is_spent():
    from planning_agent import agent as planning
    from planning_agent.models import AgentConfig, AgentState

    cfg = AgentConfig(llm_base_url="http://llm", llm_api_key="k", llm_model="m", max_iterations=30,
                      max_tokens=2000, context_window_tokens=18000, max_recon_tools=100)
    state = AgentState(query="plan a tree of two groups", context={})
    state.add_system_message("sys")
    state.add_user_message(state.query)
    seen_tools = []

    async def model(client, messages, tools, config, tool_choice=None):
        seen_tools.append(tools)
        if tools is planning.PLAN_ONLY_SCHEMAS:  # the recon cap's state: plan or ask
            return _reply("I have enough to plan; here is the plan.", messages=messages, tools=tools)
        # Results are cut to ~8000 chars (~3.5k tokens) by the loop; five at
        # once pass the hard line (usable 16000, hard 12800) in one exchange.
        calls = [("search_data", {"collection": "genome", "query": f"family:F{len(seen_tools)}_{j}"}) for j in range(5)]
        return _reply(calls=calls, messages=messages, tools=tools)

    tool = AsyncMock(side_effect=lambda **kw: _search_result(400))
    with patch.object(planning, "chat_completion", model), patch.object(planning, "TOOL_SCHEMAS", SCHEMA), \
         patch.dict("shared.tools.registry.TOOL_DISPATCH", {"search_data": tool}):
        result = await planning._run_planning_loop(state, MagicMock(), cfg, None)
    assert tool.await_count == 5                       # one exchange of five large results
    assert seen_tools[-1] is planning.PLAN_ONLY_SCHEMAS
    assert result.status == "completed" and result.answer.startswith("I have enough")
    assert any(m.get("content") == planning.RECON_CAP_NUDGE for m in state.messages)
