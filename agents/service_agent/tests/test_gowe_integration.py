"""Integration tests for the GoWe-first service agent flow.

The legacy 3-phase pipeline tests have been archived to
old_service_documentation/test_gowe_integration.py.

TODO: Add tests for the slimmed service agent:
  - test populate_and_submit() with mocked GoWe tools
  - test classify_intent() dispatching
  - test handle_submit/status/cancel
  - test that shared tools (workspace_browse, search_data) are accessible
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the agents, config, and repo root are importable
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
_AGENTS_DIR = str(Path(_REPO_ROOT) / "agents")
_CONFIG_DIR = str(Path(_REPO_ROOT) / "config")

for _dir in (_REPO_ROOT, _AGENTS_DIR, _CONFIG_DIR):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)


def test_shared_tools_importable():
    """Verify shared tools are importable from the service agent context."""
    from shared.tools.gowe import (
        list_gowe_workflows,
        get_workflow_inputs,
        submit_gowe_job,
    )
    from shared.tools.workspace import workspace_browse, get_file_metadata
    from shared.tools.data import search_data
    from shared.tools.groups import get_genome_group, get_feature_group
    from shared.tools.sra import get_sra_metadata

    assert callable(list_gowe_workflows)
    assert callable(get_workflow_inputs)
    assert callable(submit_gowe_job)
    assert callable(workspace_browse)
    assert callable(get_file_metadata)
    assert callable(search_data)
    assert callable(get_genome_group)
    assert callable(get_feature_group)
    assert callable(get_sra_metadata)


def test_service_agent_tools_dispatch():
    """Verify the service agent's tool dispatch table is correctly wired."""
    from service_agent.tools import TOOL_DISPATCH

    expected_tools = {
        "list_gowe_workflows",
        "get_workflow_inputs",
        "submit_gowe_job",
        "workspace_browse",
        "read_file_info",
        "search_data",
        "get_genome_group",
        "get_feature_group",
        "get_sra_metadata",
        "find_similar_genomes",
        "search_literature",
    }
    assert set(TOOL_DISPATCH.keys()) == expected_tools


def test_tool_registry_populate_tools():
    """Verify the tool registry only contains GoWe populate flow tools."""
    from service_agent.tool_registry import POPULATE_TOOLS, TOOL_MAP

    assert len(POPULATE_TOOLS) == 11
    assert "list_gowe_workflows" in TOOL_MAP
    assert "get_workflow_inputs" in TOOL_MAP
    assert "submit_gowe_job" in TOOL_MAP
    # Legacy tools should NOT be present
    assert "create_workflow_plan" not in TOOL_MAP
    assert "plan_service" not in TOOL_MAP
    assert "get_service_schema" not in TOOL_MAP
    assert "compose_workflow" not in TOOL_MAP
