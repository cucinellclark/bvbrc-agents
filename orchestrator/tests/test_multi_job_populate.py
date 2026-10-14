"""Multi-job mode: N > 1 jobs are prepared under the plan gate and confirmed once.

The populate loop never submits more than one job per turn.  When the
classifier says ``job_count: "many"`` (or the turn revises a prepared set) the
service agent forces the plan gate on, whatever the session mode, keeps tools
enabled after each refused ``submit_gowe_job`` so every job in the set becomes
a ``blocked_action``, and stops at MAX_PREPARED_JOBS (10).  The user answers
one Yes / No / Other question; Yes replays the set verbatim.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import shared.agent_dispatch  # noqa: E402, F401  -- sets up agent package paths
from service_agent import agent as service_agent  # noqa: E402
from service_agent.models import AgentConfig, AgentState, Intent  # noqa: E402
from service_agent.phases import populate  # noqa: E402
from service_agent.phases import replay as replay_mod  # noqa: E402
from service_agent.prompts.populate import build_populate_prompt  # noqa: E402

TC = {"workflow_id": "TaxonomicClassification",
      "inputs": {"srr_libs": [{"srr_accession": "ERR7916233", "sample_id": "ERR7916233"}], "output_path": "tc_ERR7916233"}}
BIN = {"workflow_id": "MetagenomicBinning",
       "inputs": {"srr_ids": ["ERR7916233"], "output_path": "binning_ERR7916233"}}
RM = {"workflow_id": "MetagenomicReadMapping",
      "inputs": {"srr_ids": ["ERR7916233"], "output_path": "readmap_ERR7916233"}}


def _reply(content="", tool_calls=None):
    calls = [SimpleNamespace(id=f"c{i}", type="function",
                             function=SimpleNamespace(name=n, arguments=json.dumps(a)))
             for i, (n, a) in enumerate(tool_calls or [])]
    msg = SimpleNamespace(content=content, tool_calls=calls or None, reasoning=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])


def _ok(n):
    return {"workflow_id": "wf", "submission_id": f"sub_{n}", "status": "PENDING",
            "output_path": f"/u/home/chats/s1/job{n}"}


def _submit(i, args):
    return (f"submit_gowe_job", args)


async def _run_populate(replies, mode="plan", multi_job=True, batch=False, submit=None, context=None):
    """Run the populate loop with the real execute_tool (so the gate runs)."""
    cfg = AgentConfig(llm_base_url="http://llm", llm_api_key="k", llm_model="m",
                      execution_mode=mode, max_iterations=20)
    ctx = {"execution_mode": mode, **({"batch_mode": True} if batch else {}), **(context or {})}
    state = AgentState(query="run TC and binning on ERR7916233", context=ctx, multi_job=multi_job)
    submit = submit or AsyncMock(side_effect=[_ok(n) for n in range(1, 30)])
    chat = AsyncMock(side_effect=list(replies))
    with patch.object(populate, "create_client", lambda c: MagicMock()), \
         patch.object(populate, "chat_completion", chat), \
         patch.dict(populate.TOOL_DISPATCH, {"submit_gowe_job": submit}):
        out = await populate.populate_and_submit(state.query, cfg, state)
    return out, submit, chat


@pytest.mark.asyncio
async def test_three_submits_across_three_iterations_all_become_blocked_actions():
    replies = [
        _reply(tool_calls=[("submit_gowe_job", TC)]),
        _reply(tool_calls=[("submit_gowe_job", BIN)]),
        _reply(tool_calls=[("submit_gowe_job", RM)]),
        _reply("Ready: three jobs prepared for ERR7916233."),
    ]
    out, submit, chat = await _run_populate(replies)
    submit.assert_not_awaited()
    assert out.status == "completed" and out.current_phase == "ready"
    assert [a["arguments"]["workflow_id"] for a in out.blocked_actions] == [
        "TaxonomicClassification", "MetagenomicBinning", "MetagenomicReadMapping"]
    # Tools stayed enabled after the first and second block ...
    assert [c.kwargs["tool_choice"] for c in chat.await_args_list[1:3]] == [None, None]
    # ... and the final call was a normal text turn, not forced.
    assert "three jobs" in out.operation_message
    assert "Not submitted" not in out.operation_message


@pytest.mark.asyncio
async def test_single_job_mode_still_forces_text_after_one_block():
    replies = [
        _reply(tool_calls=[("submit_gowe_job", TC)]),
        _reply("Ready to submit: Taxonomic Classification on ERR7916233."),
    ]
    out, submit, chat = await _run_populate(replies, multi_job=False)
    submit.assert_not_awaited()
    assert len(out.blocked_actions) == 1
    assert chat.await_args_list[-1].kwargs["tool_choice"] == "none"


@pytest.mark.asyncio
async def test_eleventh_job_is_refused_and_the_summary_says_the_set_was_capped():
    jobs = [("submit_gowe_job", {"workflow_id": "GenomeAssembly",
                                 "inputs": {"srr_ids": [f"SRR{100 + i}"], "output_path": f"asm_{i}"}})
            for i in range(11)]
    replies = [_reply(tool_calls=jobs), _reply("Prepared assemblies.")]
    out, submit, chat = await _run_populate(replies)
    submit.assert_not_awaited()
    assert len(out.blocked_actions) == populate.MAX_PREPARED_JOBS == 10
    assert out.capped_submits == ["GenomeAssembly → asm_10"]
    # The refused call got a tool result saying so, and the next call was text-only.
    tool_msgs = [m["content"] for m in out.messages if m.get("role") == "tool"]
    assert any("_skipped" in m and "10" in m for m in tool_msgs)
    assert chat.await_args_list[-1].kwargs["tool_choice"] == "none"
    assert "10 jobs per message" in out.operation_message
    assert "GenomeAssembly → asm_10" in out.operation_message


@pytest.mark.asyncio
async def test_batch_mode_execute_submits_at_most_ten():
    jobs = [("submit_gowe_job", {"workflow_id": "GenomeAssembly",
                                 "inputs": {"srr_ids": [f"SRR{100 + i}"], "output_path": f"asm_{i}"}})
            for i in range(11)]
    replies = [_reply(tool_calls=jobs), _reply("Submitted the assemblies.")]
    out, submit, _ = await _run_populate(replies, mode="execute", multi_job=False, batch=True)
    assert submit.await_count == 10
    assert len(out.submission_ids) == 10
    assert "10 jobs per message" in out.operation_message
    assert "asm_10" in out.operation_message


@pytest.mark.asyncio
async def test_success_claim_after_blocks_is_overwritten_with_the_set_summary():
    replies = [
        _reply(tool_calls=[("submit_gowe_job", TC), ("submit_gowe_job", BIN)]),
        _reply("Both jobs have been submitted successfully!"),
        _reply("The jobs were submitted and are running."),
    ]
    out, submit, _ = await _run_populate(replies)
    submit.assert_not_awaited()
    assert out.status == "completed"
    msg = out.operation_message.lower()
    assert "submitted successfully" not in msg and "are running" not in msg
    assert "tc_ERR7916233" in out.operation_message and "binning_ERR7916233" in out.operation_message
    assert "Yes" in out.operation_message and "No" in out.operation_message


# ---------------------------------------------------------------------------
# run_agent: the classifier decides, the session mode does not
# ---------------------------------------------------------------------------

def _cfg(mode):
    return AgentConfig(llm_base_url="http://llm", llm_api_key="k", llm_model="m",
                       execution_mode=mode, max_iterations=20)


async def _run_agent(replies, intent, mode="execute", context=None, submit=None):
    submit = submit or AsyncMock(side_effect=[_ok(n) for n in range(1, 30)])
    chat = AsyncMock(side_effect=list(replies))
    seen = {}

    async def fake_populate(query, config, state, progress_callback=None):
        seen["config_mode"] = config.execution_mode
        seen["multi_job"] = state.multi_job
        seen["prompt"] = None
        return await populate.populate_and_submit(query, config, state, progress_callback)

    with patch.object(populate, "create_client", lambda c: MagicMock()), \
         patch.object(populate, "chat_completion", chat), \
         patch.dict(populate.TOOL_DISPATCH, {"submit_gowe_job": submit}), \
         patch.dict(replay_mod.TOOL_DISPATCH, {"submit_gowe_job": submit}), \
         patch.object(service_agent, "classify_intent", AsyncMock(return_value=intent)), \
         patch.object(service_agent, "populate_and_submit", fake_populate):
        result = await service_agent.run_agent(
            "run TC and binning on ERR7916233", config=_cfg(mode),
            context={"execution_mode": mode, **(context or {})})
    return result, submit, seen


@pytest.mark.asyncio
async def test_many_in_execute_mode_prepares_instead_of_submitting():
    replies = [
        _reply(tool_calls=[("submit_gowe_job", TC)]),
        _reply(tool_calls=[("submit_gowe_job", BIN)]),
        _reply("Two jobs prepared."),
    ]
    result, submit, seen = await _run_agent(replies, Intent(job_count="many"), mode="execute")
    submit.assert_not_awaited()
    assert seen["config_mode"] == "plan" and seen["multi_job"] is True
    assert len(result.blocked_actions) == 2
    assert result.submission_ids == []
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_one_prepared_job_in_execute_mode_is_submitted_directly():
    """The classifier over-counted; Execute mode still means one job needs no click."""
    replies = [_reply(tool_calls=[("submit_gowe_job", TC)]), _reply("One job prepared.")]
    result, submit, seen = await _run_agent(replies, Intent(job_count="many"), mode="execute")
    submit.assert_awaited_once()
    sent = submit.await_args.kwargs
    assert sent["workflow_id"] == "TaxonomicClassification"
    assert sent["inputs"] == TC["inputs"]
    assert result.status == "completed"
    assert result.submission_ids == ["sub_1"]
    assert result.blocked_actions == []
    assert "Submitted **" in result.operation_message
    # The populate trace is kept ahead of the replay call.
    assert [t.tool_call.name for t in result.tool_trace] == ["submit_gowe_job", "submit_gowe_job"]


@pytest.mark.asyncio
async def test_one_prepared_job_in_plan_mode_stays_a_question():
    replies = [_reply(tool_calls=[("submit_gowe_job", TC)]), _reply("Ready to submit.")]
    result, submit, _ = await _run_agent(replies, Intent(job_count="many"), mode="plan")
    submit.assert_not_awaited()
    assert len(result.blocked_actions) == 1


@pytest.mark.asyncio
async def test_one_keeps_todays_behaviour_in_execute_mode():
    replies = [_reply(tool_calls=[("submit_gowe_job", TC)]), _reply("Submitted.")]
    result, submit, seen = await _run_agent(replies, Intent(job_count="one"), mode="execute")
    submit.assert_awaited_once()
    assert seen["config_mode"] == "execute" and seen["multi_job"] is False
    assert result.submission_ids == ["sub_1"]


@pytest.mark.asyncio
async def test_revising_a_prepared_set_always_asks_again():
    """'Other' / No-then-text: the stored set comes back as prepared_actions.

    The classifier is skipped, the gate is forced on even though the session
    is in Execute mode and the revised set has one job, and the prompt
    carries the set so the model edits it instead of starting over.
    """
    prepared = [{"tool": "submit_gowe_job", "summary": "TaxonomicClassification → tc_ERR7916233", "arguments": TC},
                {"tool": "submit_gowe_job", "summary": "MetagenomicBinning → binning_ERR7916233", "arguments": BIN}]
    replies = [_reply(tool_calls=[("submit_gowe_job", TC)]), _reply("Dropped binning; one job prepared.")]
    submit = AsyncMock(side_effect=[_ok(1)])
    chat = AsyncMock(side_effect=list(replies))
    classify = AsyncMock(side_effect=AssertionError("classifier must not run on a revision turn"))
    with patch.object(populate, "create_client", lambda c: MagicMock()), \
         patch.object(populate, "chat_completion", chat), \
         patch.dict(populate.TOOL_DISPATCH, {"submit_gowe_job": submit}), \
         patch.object(service_agent, "classify_intent", classify):
        result = await service_agent.run_agent(
            "drop the binning job", config=_cfg("execute"),
            context={"execution_mode": "execute", "prepared_actions": prepared})
    submit.assert_not_awaited()
    assert len(result.blocked_actions) == 1
    assert result.blocked_actions[0]["arguments"]["workflow_id"] == "TaxonomicClassification"
    system = chat.await_args_list[0].kwargs["messages"][0]["content"]
    assert "=== PREPARED JOB SET ===" in system
    assert "MetagenomicBinning → binning_ERR7916233" in system
    # The set is not dumped a second time in the JSON context.
    assert "binning_ERR7916233" not in system.split("=== ADDITIONAL CONTEXT ===")[-1]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def test_multi_job_prompt_sections():
    p = build_populate_prompt(execution_mode="execute", multi_job=True)
    assert "== MULTI-JOB ==" in p
    assert "== BATCH / MULTI-SAMPLE ==" not in p
    assert "RELEVANT SERVICES BY INPUT" in p
    assert "never say" in p.lower() or "NEVER say" in p
    # Multi-job is gated whatever the session mode: the execute-mode text is gone.
    assert "This session is in EXECUTE mode" not in p
    assert "Switch this chat to" not in p


def test_single_job_prompt_keeps_mode_sections():
    assert "This session is in EXECUTE mode" in build_populate_prompt(execution_mode="execute")
    assert "This session is in PLAN mode" in build_populate_prompt(execution_mode="plan")
    assert "RELEVANT SERVICES BY INPUT" in build_populate_prompt()


def test_prepared_set_section_lists_each_job():
    prepared = [{"tool": "submit_gowe_job", "summary": "A → a", "arguments": {"workflow_id": "A", "inputs": {"x": 1}}}]
    p = build_populate_prompt(multi_job=True, prepared_actions=prepared)
    assert "=== PREPARED JOB SET ===" in p
    assert "A → a" in p and '"x": 1' in p
