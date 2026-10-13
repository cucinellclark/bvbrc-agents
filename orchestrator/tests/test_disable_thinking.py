"""D1: per-agent switch that turns the model's thinking off.

On Qwen3.8 a one-line question took 70 s / 3,652 tokens with thinking and
0.4 s without (2026-10-13).  21 of 152 Oct 9 replies hit the 300 s cap.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import shared.agent_dispatch as agent_dispatch  # noqa: E402
from shared.agent_utils import thinking_extra_body  # noqa: E402

OFF = {"chat_template_kwargs": {"enable_thinking": False}}


def test_extra_body_only_for_qwen_when_asked():
    assert thinking_extra_body("Qwen/Qwen3.8-27B", True) == OFF
    assert thinking_extra_body("Qwen/Qwen3.8-27B", False) is None
    assert thinking_extra_body("gpt41", True) is None          # Argo/OpenAI may reject it
    assert thinking_extra_body(None, True) is None


def test_options_reach_the_agent_config():
    kw = agent_dispatch._build_config_kwargs("tok", {"llm_options": {"disable_thinking": True}})
    assert kw["disable_thinking"] is True and "disable_thinking_classifier" not in kw
    assert "disable_thinking" not in agent_dispatch._build_config_kwargs("tok", {})


@pytest.mark.parametrize("agent", ["data", "workspace", "helpdesk", "analysis", "service", "planning"])
def test_each_agent_client_sends_it(agent):
    mod = __import__(f"{agent}_agent.llm_client", fromlist=["_build_kwargs"])
    build = getattr(mod, "_build_kwargs", None) or getattr(mod, "build_kwargs")
    cfg = SimpleNamespace(llm_model="Qwen/Qwen3.8-27B", temperature=0.1, max_tokens=1000, disable_thinking=True)
    assert build(cfg, [{"role": "user", "content": "hi"}])["extra_body"] == OFF
    cfg.disable_thinking = False
    assert "extra_body" not in build(cfg, [{"role": "user", "content": "hi"}])


def test_classifier_uses_its_own_switch():
    from service_agent.classifier import _build_classifier_kwargs
    cfg = SimpleNamespace(disable_thinking=False, disable_thinking_classifier=True)
    assert _build_classifier_kwargs("Qwen/Qwen3.8-27B", cfg, [])["extra_body"] == OFF
    cfg.disable_thinking_classifier = False
    assert "extra_body" not in _build_classifier_kwargs("Qwen/Qwen3.8-27B", cfg, [])


@pytest.mark.asyncio
async def test_executor_forwards_the_agents_yaml_switch():
    from orchestrator.executor.agent_executor import execute_agent_step
    from orchestrator.models import OrchestratorRequest
    from orchestrator.router.models import Step

    seen = {}

    class Agent:
        key = "data"
        config = SimpleNamespace(chat_tool_params=None, mcp_server_name=None, disable_thinking=True,
                                 disable_thinking_classifier=False)
        tool_names = ["agent_chat"]

        async def call_tool(self, name, arguments, progress_handler=None):
            seen.update(arguments)
            return {"answer": "ok", "status": "completed", "tool_trace": []}

    req = OrchestratorRequest(query="q", auth_token="t")
    [e async for e in execute_agent_step(Step(agent_key="data", task="q"), Agent(), req)]
    assert seen["context"]["llm_options"] == {"disable_thinking": True}


@pytest.mark.asyncio
async def test_llm_override_never_reaches_the_prompt():
    """The model API key was dumped into === ADDITIONAL CONTEXT === before."""
    from unittest.mock import AsyncMock
    from shared.agent_loop import run_agent_loop
    from shared.models import BaseAgentState

    msg = SimpleNamespace(content="done", tool_calls=None, reasoning=None)
    llm = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")]))
    cfg = SimpleNamespace(agent_key="data", execution_mode="plan", bvbrc_auth_token=None, max_iterations=2,
                          tool_timeout_seconds=30, max_context_tokens=100000)
    state = BaseAgentState(query="q", context={"llm_override": {"api_key": "SECRET-KEY"},
                                               "llm_options": {"disable_thinking": True}, "session_id": "s"})
    await run_agent_loop(state=state, config=cfg, client=object(), system_prompt="sys", chat_completion_fn=llm)
    system = state.messages[0]["content"]
    assert "SECRET-KEY" not in system and "llm_options" not in system
