"""Tests for session workspace path builder and prompt formatter.

Covers:
  - build_session_workspace_path (shared.agent_utils)
  - format_session_workspace (shared.agent_utils)
"""

import sys
from pathlib import Path

# Ensure the bvbrc-agents repo root is importable
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest

from shared.agent_utils import (
    build_session_workspace_path,
    format_session_workspace,
)


# -----------------------------------------------------------------------
# build_session_workspace_path
# -----------------------------------------------------------------------


class TestBuildSessionWorkspacePath:
    """Unit tests for the path constructor."""

    def test_both_present(self):
        result = build_session_workspace_path(
            "/alice@patricbrc.org/home",
            "550e8400-e29b-41d4-a716-446655440000",
        )
        assert result == "/alice@patricbrc.org/home/chats/550e8400-e29b-41d4-a716-446655440000"

    def test_trailing_slash_stripped(self):
        result = build_session_workspace_path(
            "/alice@patricbrc.org/home/",
            "abc-123",
        )
        assert result == "/alice@patricbrc.org/home/chats/abc-123"

    def test_multiple_trailing_slashes(self):
        result = build_session_workspace_path(
            "/alice@patricbrc.org/home///",
            "abc-123",
        )
        assert result == "/alice@patricbrc.org/home/chats/abc-123"

    def test_missing_workspace_path_returns_none(self):
        assert build_session_workspace_path(None, "abc-123") is None

    def test_empty_workspace_path_returns_none(self):
        assert build_session_workspace_path("", "abc-123") is None

    def test_missing_session_id_returns_none(self):
        assert build_session_workspace_path("/alice@patricbrc.org/home", None) is None

    def test_empty_session_id_returns_none(self):
        assert build_session_workspace_path("/alice@patricbrc.org/home", "") is None

    def test_both_missing_returns_none(self):
        assert build_session_workspace_path(None, None) is None

    def test_both_empty_returns_none(self):
        assert build_session_workspace_path("", "") is None


# -----------------------------------------------------------------------
# format_session_workspace
# -----------------------------------------------------------------------


class TestFormatSessionWorkspace:
    """Unit tests for the prompt section formatter."""

    def test_renders_section_with_both_keys(self):
        context = {
            "workspace_path": "/alice@patricbrc.org/home",
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
        }
        result = format_session_workspace(context)
        assert result.startswith("=== SESSION WORKSPACE ===")
        assert "/alice@patricbrc.org/home/chats/550e8400-e29b-41d4-a716-446655440000" in result
        # Should contain key instruction text
        assert "workspace_browse" in result
        assert "Genome Groups" in result
        assert "Do not invent other session UUIDs" in result
        assert "may not exist yet" in result

    def test_empty_string_when_context_is_none(self):
        assert format_session_workspace(None) == ""

    def test_empty_string_when_context_is_empty(self):
        assert format_session_workspace({}) == ""

    def test_empty_string_when_session_id_missing(self):
        context = {"workspace_path": "/alice@patricbrc.org/home"}
        assert format_session_workspace(context) == ""

    def test_empty_string_when_workspace_path_missing(self):
        context = {"session_id": "abc-123"}
        assert format_session_workspace(context) == ""

    def test_empty_string_when_both_keys_empty(self):
        context = {"workspace_path": "", "session_id": ""}
        assert format_session_workspace(context) == ""

    def test_extra_context_keys_ignored(self):
        """Other keys in context should not affect the output."""
        context = {
            "workspace_path": "/bob@patricbrc.org/home",
            "session_id": "test-session",
            "bvbrc_auth_token": "secret",
            "gowe_url": "http://localhost:8080",
        }
        result = format_session_workspace(context)
        assert "=== SESSION WORKSPACE ===" in result
        assert "/bob@patricbrc.org/home/chats/test-session" in result
        # Should NOT leak other context
        assert "secret" not in result
        assert "gowe_url" not in result
