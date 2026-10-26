"""The workspace agent's result keeps a count of listed files, not the listing.

2026-10-26: one recursive listing (11,215 items) put the items and a UI grid
of them into the agent's result (7.9 MB).  Nothing rendered them; the
gateway stored them three times on the message and MongoDB refused it.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "agents")):
    if p not in sys.path:
        sys.path.insert(0, p)

from shared.models import ToolCall  # noqa: E402
from workspace_agent.models import AgentState  # noqa: E402


def _listing(n):
    items = [[f"file_{i}.tsv", "tsv", "/u/home/", "2026-10-26", f"id{i}", "u", i, {}, {}, "o", "n", ""] for i in range(n)]
    return {"result": {"count": n, "items": items, "ui_grid": {"items": items}}}


def test_listings_are_counted_not_kept():
    state = AgentState(question="what files are in my home folder?")
    state.record_execution(tc=ToolCall(id="1", name="workspace_browse", arguments={"path": "/u/home"}),
                           result=_listing(11215))
    state.record_execution(tc=ToolCall(id="2", name="workspace_browse", arguments={"path": "/u/home/f"}),
                           result=_listing(3))
    state.record_execution(tc=ToolCall(id="3", name="get_file_metadata", arguments={"path": "/u/home/a.tsv"}),
                           result={"metadata": {"name": "a.tsv", "size": 10}})
    result = state.to_result()
    dumped = result.model_dump()
    assert result.files_found == 11218
    assert "items" not in dumped and "ui_grids" not in dumped
    assert dumped["metadata"] == [{"name": "a.tsv", "size": 10}]
    # The raw tool executions (tool_trace) stay in memory for the link filter
    # and step outputs; agent_dispatch sends on only a summary of them.
    assert len(result.model_dump_json(exclude={"tool_trace"})) < 5_000
