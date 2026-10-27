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
    from bvbrc_mcp.tools.gowe import (
        list_gowe_workflows,
        get_workflow_inputs,
        submit_gowe_job,
    )
    from bvbrc_mcp.tools.workspace import workspace_browse, get_file_metadata
    from bvbrc_mcp.tools.data import search_data
    from bvbrc_mcp.tools.groups import create_group, list_groups, get_group_ids
    from bvbrc_mcp.tools.sra import get_sra_metadata

    assert callable(list_gowe_workflows)
    assert callable(get_workflow_inputs)
    assert callable(submit_gowe_job)
    assert callable(workspace_browse)
    assert callable(get_file_metadata)
    assert callable(search_data)
    assert callable(create_group)
    assert callable(list_groups)
    assert callable(get_group_ids)
    assert callable(get_sra_metadata)
