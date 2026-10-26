"""Tests for shared tool infrastructure: identifier checks and per-tool timeouts."""

from __future__ import annotations

import asyncio
import sys
import os
import pytest

# Ensure repo root is on sys.path so ``shared`` is importable.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from bvbrc_mcp.tools.gowe import _check_identifiers, _find_bad_value, _SRR_RE, _GENOME_ID_RE
from shared.tools import execute_tool, TOOL_TIMEOUT_OVERRIDES


# ---------------------------------------------------------------------------
# _check_identifiers tests
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# _find_bad_value tests
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Per-tool timeout override tests
# ---------------------------------------------------------------------------




# --- Workflow health gate -------------------------------------------------

class TestWorkflowStatus:
    """list_gowe_workflows flags workflows known to fail inside the BV-BRC app.

    A submission costs the user real compute, so an agent must be able to warn
    before spending it on a workflow with a 0% success record.
    """






    def test_the_skill_prompt_teaches_the_flag(self):
        """The flag is useless if no agent is told to read it."""
        from shared.prompts.gowe_skill import GOWE_SKILL_PROMPT

        assert "experimental" in GOWE_SKILL_PROMPT
        assert "status_reason" in GOWE_SKILL_PROMPT
