"""Progress note: one short sentence next to every tool call.

Injected into every agent's system prompt (shared loop, data, analysis,
planning, service populate).  The sentence is shown to the user as the
step's label while the tool runs (``shared.progress.intent_line``); the
code-built description becomes the small detail under it.  Testing session
2026-10-16, #5: users wanted to know what is actually happening during a
1–3 minute turn.
"""

PROGRESS_SKILL_PROMPT = """=== PROGRESS NOTE ===
When you call a tool, also write one short sentence (under 15 words) as your message text saying, in plain words for the user, what you are about to look up or do. It is shown to the user while the tool runs; it is not your answer.
- Form: "Looking up <what> in <where>", "Checking <file> for <what>", "Preparing <service> for <input>".
- Describe the thing, not the mechanism: no tool names, field names, ids, paths or markdown.
- One sentence only; leave the rest of your reply for when you have the results."""
