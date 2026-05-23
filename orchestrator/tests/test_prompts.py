"""Prompt regression tests.

Verify that all prompt files are importable and contain expected content.
These tests catch syntax errors, missing sections, and accidental deletions
introduced by the self-evolving prompt system.

These tests require sys.path entries for the agent packages and shared
modules, since they live outside the orchestrator package.
"""

import sys
from pathlib import Path

import pytest

# Add agent packages and shared modules to sys.path so we can import them
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "agents"))
sys.path.insert(0, str(_REPO_ROOT / "shared"))


# -----------------------------------------------------------------------
# Shared modules
# -----------------------------------------------------------------------


class TestSharedAgentUtils:
    """Tests for shared/agent_utils.py."""

    def test_importable(self):
        from agent_utils import (
            call_fingerprint,
            normalize_arguments,
            parse_tool_calls,
            get_response_content,
            build_tool_calls_message,
            emit_progress,
        )
        assert callable(call_fingerprint)
        assert callable(normalize_arguments)
        assert callable(parse_tool_calls)
        assert callable(get_response_content)
        assert callable(build_tool_calls_message)
        assert callable(emit_progress)

    def test_normalize_arguments_bool(self):
        from agent_utils import normalize_arguments

        result = normalize_arguments({"a": "true", "b": "false", "c": "hello"})
        assert result == {"a": True, "b": False, "c": "hello"}

    def test_normalize_arguments_int(self):
        from agent_utils import normalize_arguments

        result = normalize_arguments({"n": "42", "neg": "-5"})
        assert result == {"n": 42, "neg": -5}

    def test_normalize_arguments_null(self):
        from agent_utils import normalize_arguments

        result = normalize_arguments({"a": "null", "b": "none"})
        assert result == {"a": None, "b": None}

    def test_normalize_arguments_passthrough(self):
        from agent_utils import normalize_arguments

        result = normalize_arguments({"a": 42, "b": True, "c": None})
        assert result == {"a": 42, "b": True, "c": None}

    def test_call_fingerprint_deterministic(self):
        from agent_utils import call_fingerprint
        from dataclasses import dataclass

        @dataclass
        class FakeTC:
            name: str
            arguments: dict

        tc1 = FakeTC(name="search_data", arguments={"b": 2, "a": 1})
        tc2 = FakeTC(name="search_data", arguments={"a": 1, "b": 2})
        assert call_fingerprint(tc1) == call_fingerprint(tc2)

    def test_build_tool_calls_message_format(self):
        from agent_utils import build_tool_calls_message
        from dataclasses import dataclass

        @dataclass
        class FakeTC:
            id: str
            name: str
            arguments: dict

        tc = FakeTC(id="call_1", name="search_data", arguments={"q": "test"})
        result = build_tool_calls_message([tc])
        assert len(result) == 1
        assert result[0]["id"] == "call_1"
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "search_data"


class TestSharedAgentMessages:
    """Tests for shared/agent_messages.py."""

    def test_importable(self):
        from agent_messages import (
            DUPLICATE_CALL_WARNING,
            DUPLICATE_CALL_WARNING_SHORT,
            DUPLICATE_CALL_WARNING_SERVICE,
            MAX_ITERATIONS_SYNTHESIS,
            MAX_ITERATIONS_FALLBACK,
            MAX_PLANNING_ITERATIONS_FALLBACK,
            STUCK_IN_LOOP_DECOMPOSE,
            STUCK_IN_LOOP_BUILD,
        )
        # All should be non-empty strings
        for msg in [
            DUPLICATE_CALL_WARNING,
            DUPLICATE_CALL_WARNING_SHORT,
            DUPLICATE_CALL_WARNING_SERVICE,
            MAX_ITERATIONS_SYNTHESIS,
            MAX_ITERATIONS_FALLBACK,
            MAX_PLANNING_ITERATIONS_FALLBACK,
            STUCK_IN_LOOP_DECOMPOSE,
            STUCK_IN_LOOP_BUILD,
        ]:
            assert isinstance(msg, str)
            assert len(msg) > 10

    def test_duplicate_warning_content(self):
        from agent_messages import DUPLICATE_CALL_WARNING

        assert "DUPLICATE" in DUPLICATE_CALL_WARNING
        assert "Do NOT repeat" in DUPLICATE_CALL_WARNING

    def test_max_iterations_content(self):
        from agent_messages import MAX_ITERATIONS_SYNTHESIS

        assert "final answer" in MAX_ITERATIONS_SYNTHESIS
        assert "Do NOT request" in MAX_ITERATIONS_SYNTHESIS

    def test_stuck_loop_content(self):
        from agent_messages import STUCK_IN_LOOP_DECOMPOSE, STUCK_IN_LOOP_BUILD

        assert "STOP calling tools" in STUCK_IN_LOOP_DECOMPOSE
        assert "STOP calling tools" in STUCK_IN_LOOP_BUILD

    def test_fallback_format_strings(self):
        from agent_messages import MAX_ITERATIONS_FALLBACK, MAX_PLANNING_ITERATIONS_FALLBACK

        # Verify format strings work with {n}
        assert "5" in MAX_ITERATIONS_FALLBACK.format(n=5)
        assert "10" in MAX_PLANNING_ITERATIONS_FALLBACK.format(n=10)


# -----------------------------------------------------------------------
# Data agent prompts
# -----------------------------------------------------------------------


class TestDataAgentPrompts:
    def test_system_prompt_importable(self):
        from data_agent.prompts.system import SYSTEM_PROMPT

        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 1000

    def test_required_sections_present(self):
        from data_agent.prompts.system import (
            _PREAMBLE,
            _QUERY_SYNTAX,
            _ID_RELATIONSHIPS,
            _STRATEGY,
            _PROBE_STRATEGY,
            _EFFICIENCY,
            _CONSTRAINTS,
        )
        for section in [_PREAMBLE, _QUERY_SYNTAX, _STRATEGY, _PROBE_STRATEGY]:
            assert len(section) > 50

    def test_plan_only_addendum(self):
        from data_agent.prompts.system import PLAN_ONLY_ADDENDUM

        assert "PLANNING MODE" in PLAN_ONLY_ADDENDUM

    def test_simulated_results_importable(self):
        from data_agent.prompts.simulated_results import build_simulated_result

        assert callable(build_simulated_result)

    def test_simulated_results_search_data(self):
        from data_agent.prompts.simulated_results import build_simulated_result
        from dataclasses import dataclass

        @dataclass
        class FakeTC:
            name: str
            arguments: dict

        tc = FakeTC(name="search_data", arguments={"collection": "genome", "count_only": True})
        result = build_simulated_result(tc)
        assert result["_mode"] == "plan_only"
        assert result["_tool"] == "search_data"
        assert "_note" in result

    def test_simulated_results_unknown_tool(self):
        from data_agent.prompts.simulated_results import build_simulated_result
        from dataclasses import dataclass

        @dataclass
        class FakeTC:
            name: str
            arguments: dict

        tc = FakeTC(name="unknown_tool", arguments={})
        result = build_simulated_result(tc)
        assert "_note" in result
        assert "Plan your next step" in result["_note"]


# -----------------------------------------------------------------------
# Orchestrator prompts
# -----------------------------------------------------------------------


class TestRoutingPrompt:
    def test_routing_prompt_importable(self):
        from orchestrator.router.prompts import ROUTING_SYSTEM_PROMPT

        assert "{agent_catalog}" in ROUTING_SYSTEM_PROMPT

    def test_build_routing_prompt(self):
        from orchestrator.router.prompts import build_routing_prompt

        system, user = build_routing_prompt("test query", "agent catalog text")
        assert "agent catalog text" in system
        assert "test query" in user


class TestSynthesisPrompt:
    def test_synthesis_prompt_importable(self):
        from orchestrator.synthesizer.prompts import SYNTHESIS_SYSTEM_PROMPT

        assert isinstance(SYNTHESIS_SYSTEM_PROMPT, str)
        assert len(SYNTHESIS_SYSTEM_PROMPT) > 50


# -----------------------------------------------------------------------
# Workspace agent prompts
# -----------------------------------------------------------------------


class TestWorkspacePrompt:
    def test_system_prompt_importable(self):
        from workspace_agent.prompts.system import SYSTEM_PROMPT

        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 500

    def test_required_sections_present(self):
        from workspace_agent.prompts.system import (
            _PREAMBLE,
            _WORKSPACE_STRUCTURE,
            _FILE_TYPES,
            _PATH_HANDLING,
            _STRATEGY,
            _RESPONSE_FORMAT,
            _CONSTRAINTS,
        )
        for section in [_PREAMBLE, _WORKSPACE_STRUCTURE, _FILE_TYPES,
                        _PATH_HANDLING, _STRATEGY, _RESPONSE_FORMAT, _CONSTRAINTS]:
            assert isinstance(section, str)
            assert len(section) > 30

    def test_strategy_section_content(self):
        from workspace_agent.prompts.system import _STRATEGY

        assert "START BROAD" in _STRATEGY
        assert "NARROW DOWN" in _STRATEGY
        assert "USE THE RIGHT FILTER" in _STRATEGY

    def test_constraints_section_content(self):
        from workspace_agent.prompts.system import _CONSTRAINTS

        assert "READ-ONLY" in _CONSTRAINTS

    def test_assembled_prompt_has_all_sections(self):
        from workspace_agent.prompts.system import SYSTEM_PROMPT

        required = [
            "=== WORKSPACE STRUCTURE ===",
            "=== WORKSPACE FILE TYPES ===",
            "=== PATH HANDLING ===",
            "=== STRATEGY ===",
            "=== RESPONSE FORMAT ===",
            "=== CONSTRAINTS ===",
        ]
        for section_header in required:
            assert section_header in SYSTEM_PROMPT, f"Missing: {section_header}"


# -----------------------------------------------------------------------
# Service agent prompts
# -----------------------------------------------------------------------


class TestServicePrompt:
    def test_phase1_prompt_callable(self):
        from service_agent.prompts.phase1 import build_phase1_prompt

        prompt = build_phase1_prompt()
        assert isinstance(prompt, str)
        assert "workflow" in prompt.lower() or "service" in prompt.lower()
