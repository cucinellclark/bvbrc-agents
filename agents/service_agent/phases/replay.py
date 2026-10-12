"""Replay a prepared job or group exactly as the user approved it.

The "Submit this job" button on an ``execution_blocked`` card sends back the
arguments of the call that plan mode refused (``blocked_actions``).  Re-running
the populate loop instead made the model rebuild the inputs from a truncated
conversation, and it submitted different parameters than the card showed
(2026-10-09: SILVA / no_host became Standard / homo_sapiens).  This phase runs
those calls verbatim -- no LLM -- and writes a templated answer.

The gateway only forwards actions that match a card stored in the session, but
this module still treats them as untrusted: only execute-only tools run, and
``config`` / ``headers`` keys are dropped so a request cannot override the
agent's own config (``execute_tool`` injects config only when the key is
absent).
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from service_agent.models import AgentConfig, AgentState, ToolCall
from shared.tools import (
    EXECUTE_ONLY_TOOLS,
    execute_tool,
    is_already_submitted_result,
    is_blocked_result,
)
from shared.tools.registry import TOOL_DISPATCH

logger = logging.getLogger(__name__)

MAX_REPLAY_ACTIONS = 10

_COMPLETION_NOTE = (
    "When it finishes, a completion message is added to this chat (reopen or "
    "refresh the chat to see it), or check the Jobs list."
)


def valid_replay_actions(actions: Any) -> list[dict[str, Any]] | None:
    """Return the actions in a runnable shape, or ``None`` if any is invalid."""
    if not isinstance(actions, list) or not actions or len(actions) > MAX_REPLAY_ACTIONS:
        return None
    cleaned: list[dict[str, Any]] = []
    for action in actions:
        if not isinstance(action, dict):
            return None
        tool = action.get("tool")
        arguments = action.get("arguments")
        if tool not in EXECUTE_ONLY_TOOLS or not isinstance(arguments, dict):
            return None
        cleaned.append({
            "tool": tool,
            "arguments": {
                k: copy.deepcopy(v)
                for k, v in arguments.items()
                if k not in ("config", "headers")
            },
            "summary": str(action.get("summary") or tool),
        })
    return cleaned


def _submit_line(summary: str, result: dict[str, Any]) -> str:
    out = result.get("output_path")
    where = f" Results will be saved to `{out}`." if out else ""
    return f"Submitted **{summary}**.{where}"


def _group_line(summary: str, result: dict[str, Any]) -> str:
    name = result.get("name") or summary
    count = result.get("count")
    path = result.get("path")
    verb = "Updated" if result.get("action") in ("append", "replace") else "Created"
    size = f" ({count} items)" if count is not None else ""
    where = f" at `{path}`" if path else ""
    return f"{verb} group **{name}**{size}{where}."


async def replay_blocked_actions(
    actions: list[dict[str, Any]],
    config: AgentConfig,
    state: AgentState,
) -> AgentState:
    """Run each prepared call verbatim, in order; stop at the first failure."""
    headers = (
        {"Authorization": config.bvbrc_auth_token}
        if config.bvbrc_auth_token else None
    )
    lines: list[str] = []
    succeeded = 0
    failed = False

    for i, action in enumerate(actions):
        tool, summary = action["tool"], action["summary"]
        tc = ToolCall(id=f"replay_{i}", name=tool, arguments=action["arguments"])
        result = await execute_tool(
            tool_name=tool,
            arguments=dict(action["arguments"]),
            dispatch_table=TOOL_DISPATCH,
            timeout_seconds=config.tool_timeout_seconds,
            config=config,
            headers=headers,
        )
        error = result.get("error") if isinstance(result, dict) else "no result"
        state.record_execution(tc=tc, result=result, error=error, iteration=0)

        if is_already_submitted_result(result):
            # A repeat click (or a second turn) for a job that already went
            # out: not a failure, and nothing new was submitted.
            earlier = result.get("duplicate_of")
            lines.append(
                f"**{summary}** was already submitted from this chat in the last "
                f"few minutes{f' ({earlier})' if earlier else ''}, so it was not "
                f"submitted again. Check the Jobs list for its progress."
            )
            succeeded += 1
            continue
        if is_blocked_result(result):
            lines.append(
                f"Not run: **{summary}**. This chat is in Plan mode, so nothing "
                f"was submitted or created."
            )
            failed = True
            break
        if error:
            logger.warning("Replay of %s failed: %s", tool, error)
            lines.append(f"Could not run **{summary}**: {error}")
            failed = True
            break

        succeeded += 1
        if tool == "submit_gowe_job" and result.get("submission_id"):
            sub_id = result["submission_id"]
            state.workflow_id = result.get("workflow_id")
            state.submission_id = sub_id
            state.submission_ids.append(sub_id)
            state.auto_submitted = True
            state.persisted = True
            lines.append(_submit_line(summary, result))
        elif tool == "create_group":
            lines.append(_group_line(summary, result))
        else:
            lines.append(f"Done: **{summary}**.")

    skipped = actions[len(state.tool_executions):]
    if failed and skipped:
        lines.append(
            "Not attempted after that: "
            + "; ".join(f"**{a['summary']}**" for a in skipped) + "."
        )
    if state.submission_ids:
        lines.append(_COMPLETION_NOTE)

    message = "\n\n".join(lines)
    state.current_phase = "done"
    if not succeeded:
        state.status = "error"
        state.error_message = message
    else:
        state.status = "completed"
        state.operation_message = message
    return state
