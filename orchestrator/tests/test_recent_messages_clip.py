"""Prior turns keep their start AND end.

Every message used to be cut to its first 500 characters (agents and the
router), so the plan, "Ready to submit" inputs and next steps at the end of a
long reply were lost: the service agent asked nbowers to re-list jobs 2-4,
"let's do the next steps" got "please clarify" (marcnguy 6623208c), and the
router said "I don't see a list" (nbowers 6c721969).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.agent_utils import clip_middle, format_recent_messages  # noqa: E402

PLAN_REPLY = (
    "Here is what I found about your samples.\n"
    + "Details about each sample. " * 200
    + "\n\nNext steps:\n1. Assemble SRR1\n2. Assemble SRR2\n3. Annotate both"
)


def test_clip_middle_keeps_both_ends():
    out = clip_middle("A" * 100 + "B" * 100, 10, 5)
    assert out == "A" * 10 + "\n…[middle omitted]…\n" + "B" * 5


def test_clip_middle_leaves_short_text_alone():
    assert clip_middle("short", 10, 10) == "short"


def test_last_assistant_reply_keeps_its_next_steps():
    out = format_recent_messages([
        {"role": "user", "content": "check my samples"},
        {"role": "assistant", "content": PLAN_REPLY},
        {"role": "user", "content": "ok do the next steps"},
    ])
    assert "3. Annotate both" in out
    assert out.startswith("user: check my samples\nassistant: Here is what I found")


def test_older_assistant_replies_get_the_smaller_budget():
    older = "O" * 3000
    newer = "N" * 3000
    out = format_recent_messages([
        {"role": "assistant", "content": older},
        {"role": "assistant", "content": newer},
    ])
    older_line, newer_line = out.split("\nassistant: ")
    assert "[middle omitted]" in older_line          # 3000 > 1500
    assert "[middle omitted]" not in newer_line      # 3000 <= 4000


def test_window_is_the_last_five():
    msgs = [{"role": "user", "content": f"m{i}"} for i in range(8)]
    assert format_recent_messages(msgs) == "\n".join(f"user: m{i}" for i in range(3, 8))


def test_router_context_keeps_the_end_of_the_last_reply():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import patch

    from orchestrator.models import OrchestratorRequest
    from orchestrator.router import router as router_mod

    captured = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after prompt build")

    registry = SimpleNamespace(healthy_agents=["data"], agents={}, agent_keys=[],
                               catalog=lambda: [])
    req = OrchestratorRequest(query="ok do the next steps", recent_messages=[
        {"role": "assistant", "content": PLAN_REPLY},
    ])
    with patch.object(router_mod, "build_routing_prompt", fake_build):
        try:
            asyncio.run(router_mod.route(req, registry, llm=None))
        except RuntimeError:
            pass
    assert "3. Annotate both" in (captured.get("conversation_context") or "")
