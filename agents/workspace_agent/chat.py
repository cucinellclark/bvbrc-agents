#!/usr/bin/env python3
"""
Interactive chat for testing the BV-BRC Workspace Agent.

Runs the full agent loop (plan -> execute tools -> evaluate -> repeat)
against the real BV-BRC Workspace API and shows what the agent does at
each step.

Usage (from Workspace/ directory):
    python chat.py

Commands:
    /config         - Show current configuration
    /verbose        - Toggle verbose output (tool traces)
    /help           - Show commands
    quit / exit     - Exit
"""

from __future__ import annotations

import asyncio
import json
import os

from workspace_agent.agent import run_agent
from workspace_agent.models import AgentConfig, AgentResult


# ---------------------------------------------------------------------------
# Auth token loading
# ---------------------------------------------------------------------------

def _load_auth_token() -> str | None:
    """Load auth token from auth_token.txt in the Workspace directory."""
    token_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "auth_token.txt"
    )
    if os.path.exists(token_path):
        with open(token_path, "r") as f:
            token = f.read().strip()
        if token:
            return token
    return None


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _dim(text: str) -> str:
    return f"\033[2m{text}\033[0m"

def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m"

def _green(text: str) -> str:
    return f"\033[32m{text}\033[0m"

def _yellow(text: str) -> str:
    return f"\033[33m{text}\033[0m"

def _red(text: str) -> str:
    return f"\033[31m{text}\033[0m"

def _cyan(text: str) -> str:
    return f"\033[36m{text}\033[0m"


def print_result(result: AgentResult, verbose: bool = False) -> None:
    """Print the agent result in a readable format."""

    # Status line
    status_color = _green if result.status == "completed" else _yellow
    print(f"\n{_dim('Status:')} {status_color(result.status)}  "
          f"{_dim('|')}  {_dim('Iterations:')} {result.iterations_used}  "
          f"{_dim('|')}  {_dim('Time:')} {result.elapsed_seconds}s")

    if result.paths_explored:
        print(f"{_dim('Paths explored:')} {', '.join(result.paths_explored)}")

    if result.items:
        print(f"{_dim('Files found:')} {len(result.items)}")

    if result.metadata:
        print(f"{_dim('Files inspected:')} {len(result.metadata)}")

    if result.previews:
        print(f"{_dim('Files previewed:')} {len(result.previews)}")

    # Tool trace (verbose mode)
    if result.tool_trace and verbose:
        print(f"\n{_bold('--- Tool Executions ---')}")
        for i, ex in enumerate(result.tool_trace, 1):
            tc = ex.tool_call
            duration = f" ({ex.duration_ms:.0f}ms)" if ex.duration_ms else ""
            print(f"\n  {_cyan(f'Step {i}:')} {_bold(tc.name)}{duration}")

            args_str = json.dumps(tc.arguments, default=str)
            if len(args_str) > 120:
                args_str = json.dumps(tc.arguments, indent=4, default=str)
            print(f"    {_dim('Args:')} {args_str}")

            if ex.error:
                print(f"    {_red('Error:')} {ex.error}")
            elif ex.result is not None:
                result_str = json.dumps(ex.result, default=str)
                if len(result_str) > 300:
                    r = ex.result
                    if isinstance(r, dict):
                        summary_parts = []
                        inner = r.get("result", r)
                        if "count" in inner:
                            summary_parts.append(f"count={inner['count']}")
                        if "items" in inner and isinstance(inner["items"], list):
                            summary_parts.append(
                                f"items=[{len(inner['items'])} files]"
                            )
                        if "metadata" in inner:
                            summary_parts.append("metadata=<present>")
                        if "data" in inner:
                            data_preview = str(inner["data"])[:80]
                            summary_parts.append(f"data={data_preview}...")
                        if summary_parts:
                            result_str = "{" + ", ".join(summary_parts) + "}"
                        else:
                            result_str = result_str[:300] + "..."
                print(f"    {_dim('Result:')} {result_str}")
    elif result.tool_trace and not verbose:
        print(f"\n{_dim(f'  [{len(result.tool_trace)} tool call(s) executed -- use /verbose to see details]')}")

    # Final answer
    if result.answer:
        print(f"\n{_bold('--- Answer ---')}")
        print(result.answer)

    print()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def main() -> None:
    # Load config from agent_config.yaml (or WORKSPACE_AGENT_CONFIG env var)
    token = _load_auth_token()
    config = AgentConfig.from_yaml(bvbrc_auth_token=token)

    verbose = False

    config_source = os.environ.get("WORKSPACE_AGENT_CONFIG") or "agent_config.yaml"
    print(_bold("BV-BRC Workspace Agent - Interactive Chat"))
    print(f"{_dim('Config:')} {config_source}")
    print(f"{_dim('Model:')} {config.llm_model}")
    print(f"{_dim('Endpoint:')} {config.llm_base_url}")
    print(f"{_dim('Context limit:')} {config.max_context_tokens} tokens")
    print(f"{_dim('Auth token:')} {'loaded' if config.bvbrc_auth_token else 'not set'}")
    print(f"\n{_dim('Type a question about your workspace, or /help for commands.')}")
    print()

    while True:
        try:
            query = input(_bold("you> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{_dim('Bye.')}")
            break

        if not query:
            continue

        # --- Commands ---
        if query.lower() in ("quit", "exit", "q"):
            print(_dim("Bye."))
            break

        if query.lower() == "/help":
            print(f"""
{_bold('Commands:')}
  /config         Show current configuration
  /verbose        Toggle verbose output (show tool execution details)
  /help           Show this help
  quit            Exit
""")
            continue

        if query.lower() == "/config":
            print(f"\n{_bold('Configuration:')}")
            print(f"  {_dim('LLM')}")
            print(f"    Model:              {config.llm_model}")
            print(f"    Endpoint:           {config.llm_base_url}")
            print(f"    Temperature:        {config.temperature}")
            print(f"    Max output tokens:  {config.max_tokens}")
            print(f"  {_dim('Context')}")
            print(f"    Max context tokens: {config.max_context_tokens}")
            print(f"    Max tool result:    {config.max_tool_result_chars} chars")
            print(f"  {_dim('Agent')}")
            print(f"    Max iterations:     {config.max_iterations}")
            print(f"    Tool timeout:       {config.tool_timeout_seconds}s")
            print(f"  {_dim('Workspace')}")
            print(f"    API:                {config.bvbrc_workspace_url}")
            print(f"    Auth token:         {'set' if config.bvbrc_auth_token else 'not set'}")
            print(f"  {_dim('Runtime')}")
            print(f"    Verbose:            {verbose}")
            print()
            continue

        if query.lower() == "/verbose":
            verbose = not verbose
            print(f"{_dim('Verbose output:')} {'ON' if verbose else 'OFF'}\n")
            continue

        # --- Full agent run ---
        print(f"\n{_dim('Exploring workspace...')}")
        try:
            result = await run_agent(query, config)
            print_result(result, verbose=verbose)
        except Exception as e:
            print(f"{_red('Error:')} {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
