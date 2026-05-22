#!/usr/bin/env python3
"""
Interactive chat for testing the BV-BRC Data Agent.

Runs the full agent loop (plan -> execute tools -> evaluate -> repeat)
against the real BV-BRC Solr API and shows what the agent does at each
step.

Usage (from Data/ directory):
    source data_agent_env/bin/activate
    python chat.py

Commands:
    /plan <query>   - Plan-only mode (no tool execution)
    /config         - Show current configuration
    /verbose        - Toggle verbose output (tool traces)
    /help           - Show commands
    quit / exit     - Exit
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

from data_agent.agent import plan_only, run_agent
from data_agent.models import AgentConfig, AgentResult


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

    if result.sources:
        print(f"{_dim('Collections:')} {', '.join(result.sources)}")

    # Tool trace (verbose mode or plan-only)
    if result.tool_trace and verbose:
        print(f"\n{_bold('--- Tool Executions ---')}")
        for i, ex in enumerate(result.tool_trace, 1):
            tc = ex.tool_call
            duration = f" ({ex.duration_ms:.0f}ms)" if ex.duration_ms else ""
            print(f"\n  {_cyan(f'Step {i}:')} {_bold(tc.name)}{duration}")

            # Compact argument display
            args_str = json.dumps(tc.arguments, default=str)
            if len(args_str) > 120:
                args_str = json.dumps(tc.arguments, indent=4, default=str)
            print(f"    {_dim('Args:')} {args_str}")

            if ex.error:
                print(f"    {_red('Error:')} {ex.error}")
            elif ex.result is not None:
                result_str = json.dumps(ex.result, default=str)
                if len(result_str) > 300:
                    # Show key summary fields only
                    r = ex.result
                    if isinstance(r, dict):
                        summary_parts = []
                        if "numFound" in r:
                            summary_parts.append(f"numFound={r['numFound']}")
                        if "count" in r:
                            summary_parts.append(f"count={r['count']}")
                        if "results" in r and isinstance(r["results"], list):
                            summary_parts.append(f"results=[{len(r['results'])} records]")
                        if "facets" in r and isinstance(r["facets"], dict):
                            facet_info = {k: len(v) for k, v in r["facets"].items()
                                         if isinstance(v, list)}
                            summary_parts.append(f"facets={facet_info}")
                        if "error" in r:
                            summary_parts.append(f"error={r['error']}")
                        if summary_parts:
                            result_str = "{" + ", ".join(summary_parts) + "}"
                        else:
                            result_str = result_str[:300] + "..."
                print(f"    {_dim('Result:')} {result_str}")
    elif result.tool_trace and not verbose:
        # Compact trace
        print(f"\n{_dim(f'  [{len(result.tool_trace)} tool call(s) executed — use /verbose to see details]')}")

    # Plan-only calls
    if result.planned_tool_calls and not result.tool_trace:
        print(f"\n{_bold('--- Planned Tool Calls ---')}")
        for i, tc in enumerate(result.planned_tool_calls, 1):
            print(f"\n  {_cyan(f'Step {i}:')} {_bold(tc['name'])}")
            args_str = json.dumps(tc["arguments"], indent=4, default=str)
            print(f"    {_dim('Args:')} {args_str}")

    # Final answer
    if result.answer:
        print(f"\n{_bold('--- Answer ---')}")
        print(result.answer)

    print()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def main() -> None:
    config = AgentConfig()
    verbose = False

    print(_bold("BV-BRC Data Agent - Interactive Chat"))
    print(f"{_dim('Model:')} {config.llm_model}")
    print(f"{_dim('Endpoint:')} {config.llm_base_url}")
    print(f"{_dim('API:')} {config.bvbrc_api_url}")
    print(f"{_dim('Max iterations:')} {config.max_iterations}")
    print(f"\n{_dim('Type a question about BV-BRC data, or /help for commands.')}")
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
  /plan <query>   Plan-only mode (shows what the agent would do, no execution)
  /config         Show current configuration
  /verbose        Toggle verbose output (show tool execution details)
  /help           Show this help
  quit            Exit
""")
            continue

        if query.lower() == "/config":
            print(f"\n{_bold('Configuration:')}")
            print(f"  Model:          {config.llm_model}")
            print(f"  Endpoint:       {config.llm_base_url}")
            print(f"  API:            {config.bvbrc_api_url}")
            print(f"  Max iterations: {config.max_iterations}")
            print(f"  Temperature:    {config.temperature}")
            print(f"  Max tokens:     {config.max_tokens}")
            print(f"  Tool timeout:   {config.tool_timeout_seconds}s")
            print(f"  Verbose:        {verbose}")
            print(f"  Auth token:     {'set' if config.bvbrc_auth_token else 'not set'}")
            print()
            continue

        if query.lower() == "/verbose":
            verbose = not verbose
            print(f"{_dim('Verbose output:')} {'ON' if verbose else 'OFF'}\n")
            continue

        # --- Plan-only mode ---
        if query.lower().startswith("/plan "):
            plan_query = query[6:].strip()
            if not plan_query:
                print(_red("Usage: /plan <your question>"))
                continue

            print(f"\n{_dim('Planning (no execution)...')}")
            try:
                result = await plan_only(plan_query, config)
                print_result(result, verbose=True)  # always verbose for plan
            except Exception as e:
                print(f"{_red('Error:')} {type(e).__name__}: {e}\n")
            continue

        # --- Full agent run ---
        print(f"\n{_dim('Thinking...')}")
        try:
            result = await run_agent(query, config)
            print_result(result, verbose=verbose)
        except Exception as e:
            print(f"{_red('Error:')} {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
