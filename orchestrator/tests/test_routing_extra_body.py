"""Routing request shaping: thinking toggle and structured output.

Routing is a constrained classification, not a reasoning task. These tests
pin the two behaviours measured against the live vLLM on 2026-10-05:

  * thinking on  -> 200+ completion tokens, content=None (the empty-response
    retry in router.py exists because of this)
  * thinking off -> 47-81 tokens, valid content
  * no schema    -> the model invented keys {agent, action, parameters}
  * response_format -> exactly {decision, reasoning, agent_key}

`guided_json` is accepted by that vLLM build but silently ignored, which is
why `response_format` is the mechanism used here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator.llm.client import _SDK_NATIVE_PARAMS, LLMClient
from orchestrator.llm.config import LLMConfig
from orchestrator.router.models import (
    ROUTING_JSON_SCHEMA,
    build_routing_extra_body,
)


class TestBuildRoutingExtraBody:
    def test_both_enabled(self):
        extra = build_routing_extra_body(True, True)
        assert extra["chat_template_kwargs"] == {"enable_thinking": False}
        assert extra["response_format"]["type"] == "json_schema"
        assert (
            extra["response_format"]["json_schema"]["schema"] is ROUTING_JSON_SCHEMA
        )

    def test_both_disabled_is_empty(self):
        # An empty dict means the request is byte-identical to the old
        # behaviour, so the flags are a true off-switch for a model swap.
        assert build_routing_extra_body(False, False) == {}

    def test_independently_switchable(self):
        assert "response_format" not in build_routing_extra_body(True, False)
        assert "chat_template_kwargs" not in build_routing_extra_body(False, True)

    def test_schema_covers_all_three_decision_shapes(self):
        props = ROUTING_JSON_SCHEMA["properties"]
        assert props["decision"]["enum"] == ["direct", "agent", "pipeline"]
        assert "direct_response" in props          # direct
        assert {"agent_key", "task"} <= set(props)  # agent
        assert props["steps"]["type"] == "array"    # pipeline
        step = props["steps"]["items"]["properties"]
        assert {"agent_key", "task", "depends_on"} <= set(step)
        assert ROUTING_JSON_SCHEMA["required"] == ["decision", "reasoning"]


class TestExtraBodyRequestShaping:
    """The OpenAI SDK validates kwargs, so vLLM extensions must be nested.

    Passing chat_template_kwargs top-level raises
    TypeError: AsyncCompletions.create() got an unexpected keyword argument.
    """

    def _kwargs(self, extra):
        client = LLMClient(
            LLMConfig(
                base_url="http://example.invalid/v1",
                api_key="k",
                model="Qwen/Qwen3.6-35B-A3B",
                extra_body=extra,
            )
        )
        return client._build_create_kwargs([{"role": "user", "content": "hi"}])

    def test_vllm_extension_is_nested_under_extra_body(self):
        kw = self._kwargs({"chat_template_kwargs": {"enable_thinking": False}})
        assert "chat_template_kwargs" not in kw, "would raise TypeError in the SDK"
        assert kw["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}

    def test_sdk_native_param_stays_top_level(self):
        rf = {"type": "json_schema", "json_schema": {"name": "r", "schema": {}}}
        kw = self._kwargs({"response_format": rf})
        assert kw["response_format"] == rf
        assert "extra_body" not in kw

    def test_mixed_params_are_split(self):
        kw = self._kwargs(build_routing_extra_body(True, True))
        assert kw["response_format"]["type"] == "json_schema"
        assert kw["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}

    def test_core_fields_cannot_be_overridden(self):
        kw = self._kwargs({"model": "evil", "messages": [], "seed": 7})
        assert kw["model"] == "Qwen/Qwen3.6-35B-A3B"
        assert kw["messages"] == [{"role": "user", "content": "hi"}]
        assert kw["seed"] == 7

    def test_no_extra_body_leaves_request_unchanged(self):
        kw = self._kwargs(None)
        assert "extra_body" not in kw
        assert "response_format" not in kw

    def test_response_format_is_sdk_native(self):
        assert "response_format" in _SDK_NATIVE_PARAMS
        assert "chat_template_kwargs" not in _SDK_NATIVE_PARAMS
        assert "guided_json" not in _SDK_NATIVE_PARAMS
