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
# Empty final answer after tools ran
# ---------------------------------------------------------------------------

# Thinking models sometimes spend the whole completion on reasoning and
# return content=None after their tools succeeded; without a nudge the
# user got "unable to produce a textual summary" over data that was found.
EMPTY_ANSWER_NUDGE = (
    "Your last reply was empty. You have already run your tools and their "
    "results are above. Now write your answer to the user's question in "
    "plain text, using those results (include the key numbers and links). "
    "Keep your reasoning short. Do not call more tools unless something "
    "essential is missing."
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

# Shared loop (data / workspace / helpdesk / analysis) and the planning loop:
# the model kept repeating a call it had already made.  Sent as a user
# message (vLLM/Qwen reject mid-conversation system messages), then one
# final call with tool_choice="none".
STUCK_ANSWER_NOW = (
    "[SYSTEM CHECK — not from the user. Do not acknowledge or quote this note.] "
    "You have repeated a tool call you already made, and repeating it will not "
    "give a different result. Do not call any more tools. Answer the user now "
    "with what the tool results above show; if something is still missing, say "
    "exactly what and suggest the next request."
)

# Shown when even the forced answer comes back empty.
STUCK_FALLBACK = (
    "I kept repeating the same lookup and stopped. Here is what I ran:\n\n{steps}\n\n"
    "Please ask again more narrowly, for example with a count first or a smaller set."
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
