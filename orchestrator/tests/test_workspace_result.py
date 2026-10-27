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
    return {"count": n, "items": items}


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


def test_a_trimmed_flat_listing_keeps_its_count_total_and_path():
    # workspace_browse results are flat since 2026-10-27; message trimming
    # kept the path only for the old nested shape.
    import json
    from shared.agent_utils import trim_messages_to_fit

    listing = json.dumps({"items": [{"name": f"f{i}.fastq", "path": f"/u/home/reads/f{i}.fastq"} for i in range(400)],
                          "count": 50, "total": 812, "path": "/u/home/reads"})
    call = lambda i: {"role": "assistant", "content": None, "tool_calls": [
        {"id": f"tc{i}", "type": "function", "function": {"name": "workspace_browse", "arguments": "{}"}}]}
    messages = [{"role": "system", "content": "test"}, {"role": "user", "content": "list"},
                call(1), {"role": "tool", "tool_call_id": "tc1", "content": listing},
                call(2), {"role": "tool", "tool_call_id": "tc2", "content": '{"items": []}'},
                {"role": "assistant", "content": "done"}]
    trimmed = trim_messages_to_fit(messages, None, max_tokens=500)
    old = [m["content"] for m in trimmed if m.get("tool_call_id") == "tc1"]
    assert old and old[0].startswith("[Previous tool result compressed]")
    assert "count=50" in old[0] and "total=812" in old[0] and "path=/u/home/reads" in old[0]
