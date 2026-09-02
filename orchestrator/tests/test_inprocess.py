"""Tests for in-process agent dispatch (no MCP HTTP hop)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import AgentConfig
from orchestrator.registry.agent_handle import InProcessAgentHandle

_AGENTS_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_AGENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENTS_ROOT))
_agents_dir = str(_AGENTS_ROOT / "agents")
if _agents_dir not in sys.path:
    sys.path.insert(0, _agents_dir)

import shared.agent_dispatch  # noqa: E402, F401  -- sets up agent package paths


def _inprocess_config(key: str = "helpdesk") -> AgentConfig:
    return AgentConfig(
        name=f"{key.title()} Agent",
        description=f"Test {key} agent",
        protocol="inprocess",
        chat_tool_params={"agent_type": key},
    )


class TestInProcessAgentHandle:
    def test_creation_is_healthy_without_connect(self):
        handle = InProcessAgentHandle("helpdesk", _inprocess_config())
        assert handle.is_healthy
        assert handle.is_connected
        assert handle.tool_names == ["agent_chat"]
        assert handle.summary()["protocol"] == "inprocess"

    @pytest.mark.asyncio
    async def test_call_tool_returns_dict(self):
        handle = InProcessAgentHandle("helpdesk", _inprocess_config())
        canned = {
            "status": "completed",
            "answer": "BV-BRC is a pathogen resource.",
            "tool_trace": [],
            "iterations_used": 1,
        }

        with patch(
            "shared.agent_dispatch.dispatch_agent",
            new_callable=AsyncMock,
            return_value=canned,
        ) as mock_dispatch:
            result = await handle.call_tool(
                "agent_chat",
                {
                    "query": "What is BV-BRC?",
                    "agent_type": "helpdesk",
                    "context": {"session_id": "abc"},
                    "token": "tok",
                },
            )

        assert isinstance(result, dict)
        assert result["status"] == "completed"
        assert result["answer"] == canned["answer"]
        mock_dispatch.assert_called_once()
        kwargs = mock_dispatch.call_args.kwargs
        assert kwargs["agent_type"] == "helpdesk"
        assert kwargs["query"] == "What is BV-BRC?"
        assert kwargs["context"] == {"session_id": "abc"}
        assert kwargs["token"] == "tok"


class TestDispatchAgent:
    @pytest.mark.asyncio
    async def test_dispatch_agent_unknown_type(self):
        from shared.agent_dispatch import dispatch_agent

        result = await dispatch_agent("not-an-agent", "hello", {}, "tok")
        assert result["status"] == "error"
        assert "Unknown agent type" in result["answer"]
        assert result["tool_trace"] == []

    @pytest.mark.asyncio
    async def test_dispatch_agent_helpdesk_mocked_llm(self):
        """dispatch_agent returns a result dict when the agent LLM is mocked."""
        canned = MagicMock()
        canned.answer = "BV-BRC is a bioinformatics resource center."
        canned.status = "completed"
        canned.sources = []
        canned.iterations_used = 1
        canned.elapsed_seconds = 0.01
        canned.tool_trace = []

        with patch(
            "helpdesk_agent.agent.run_agent",
            new_callable=AsyncMock,
            return_value=canned,
        ):
            from shared.agent_dispatch import dispatch_agent

            result = await dispatch_agent(
                "helpdesk",
                "What is BV-BRC?",
                {},
                "test-token",
            )

        assert result["status"] == "completed"
        assert result["answer"] == canned.answer
        assert "tool_trace" in result
        assert result["iterations_used"] == 1
