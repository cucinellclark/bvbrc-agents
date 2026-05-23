"""LLM-facing messages injected into agent conversations during execution.

These are NOT system prompts -- they are mid-conversation instructions
inserted when the agent loop detects problems (duplicates, stuck loops,
max iterations). They are evolution targets because their wording
directly affects LLM behavior.

Human-facing progress messages (e.g., "Analyzing your question...") are
NOT included here -- they don't affect LLM behavior and live inline in
the agent code (or in shared/progress_messages.py if extracted later).
"""

# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

DUPLICATE_CALL_WARNING = (
    "DUPLICATE CALL DETECTED: You already executed this exact "
    "query and received results above. Do NOT repeat it. "
    "Use the results you already have to provide your final "
    "answer to the user's question."
)

DUPLICATE_CALL_WARNING_SHORT = (
    "DUPLICATE CALL DETECTED. Use the results you "
    "already have. Do NOT repeat this call."
)

DUPLICATE_CALL_WARNING_SERVICE = (
    "DUPLICATE CALL DETECTED: You already called this "
    "tool with these exact arguments. Use the results "
    "you already have. Do NOT repeat this call."
)

# ---------------------------------------------------------------------------
# Max iterations forced synthesis
# ---------------------------------------------------------------------------

MAX_ITERATIONS_SYNTHESIS = (
    "You have reached the maximum number of tool call iterations. "
    "You MUST now provide your final answer based on the data you "
    "have already collected. Do NOT request any more tool calls. "
    "Summarize what you found and answer the user's question."
)

# ---------------------------------------------------------------------------
# Stuck in loop recovery
# ---------------------------------------------------------------------------

STUCK_IN_LOOP_DECOMPOSE = (
    "You are stuck in a loop. STOP calling tools. Instead, "
    "provide your plan as a text response, explaining what "
    "services are needed and their dependencies. If you need "
    "information from the user, ask a clear question."
)

STUCK_IN_LOOP_BUILD = (
    "You are stuck in a loop. STOP calling tools. If you need "
    "information from the user to complete this step, ask a "
    "clear question in your text response."
)

# ---------------------------------------------------------------------------
# Max iterations fallback (human-facing, but included here for completeness
# since they're returned as the agent's final answer)
# ---------------------------------------------------------------------------

MAX_ITERATIONS_FALLBACK = "Reached maximum iterations. Executed {n} tool calls."
MAX_PLANNING_ITERATIONS_FALLBACK = (
    "Reached maximum planning iterations. Planned {n} tool calls."
)
