#!/usr/bin/env python3
"""
Interactive chat for testing the BV-BRC Service Agent v2.

Runs the full three-phase agent loop (Decompose -> Build -> Compose)
against the real BV-BRC APIs and shows what the agent does at each step.

Usage (from Service2/ directory):
    python chat.py

Commands:
    <query>         - Run the full three-phase agent
    /config         - Show current configuration
    /verbose        - Toggle verbose output (tool traces)
    /help           - Show commands
    quit / exit     - Exit
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from service_agent.agent import run_agent
from service_agent.models import AgentConfig, AgentResult


# ---------------------------------------------------------------------------
# Auth token loading
# ---------------------------------------------------------------------------

def _load_auth_token() -> str | None:
    """Load auth token from auth_token.txt in the Service2 directory."""
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
    status_colors = {
        "completed": _green,
        "needs_input": _yellow,
        "error": _red,
    }
    color_fn = status_colors.get(result.status, _yellow)
    print(f"\n{_dim('Status:')} {color_fn(result.status)}  "
          f"{_dim('|')}  {_dim('Time:')} {result.elapsed_seconds}s")

    if result.sources:
        print(f"{_dim('Services:')} {', '.join(result.sources)}")

    # Question (needs_input)
    if result.question:
        print(f"\n{_bold('--- Question for User ---')}")
        print(result.question)

    # Error
    if result.error_message:
        print(f"\n{_red('--- Error ---')}")
        print(result.error_message)

    # Workflow plan
    if result.workflow_plan:
        plan = result.workflow_plan
        print(f"\n{_bold('--- Workflow Plan ---')}")
        print(f"  {_dim('Name:')} {plan.get('workflow_name', 'unnamed')}")
        print(f"  {_dim('Description:')} {plan.get('description', '')}")
        steps = plan.get("steps", [])
        for s in steps:
            deps = s.get("depends_on", [])
            dep_str = f" -> [{', '.join(deps)}]" if deps else ""
            print(f"  {_cyan(s.get('step_id', '?'))}: "
                  f"{s.get('service_name', '?')} -- {s.get('intent', '')}{dep_str}")
        topo = plan.get("topological_order", [])
        if topo:
            print(f"  {_dim('Build order:')} {' -> '.join(topo)}")

    # Completed steps
    if result.completed_steps:
        print(f"\n{_bold('--- Validated Steps ---')}")
        for sid, step_data in result.completed_steps.items():
            svc = step_data.get("service_name", "?")
            api = step_data.get("api_name", "?")
            print(f"\n  {_cyan(f'[{sid}]')} {_bold(svc)} ({api})")
            if step_data.get("auto_corrections"):
                for ac in step_data["auto_corrections"]:
                    print(f"    {_yellow('auto-corrected:')} {ac}")
            if step_data.get("warnings"):
                for w in step_data["warnings"]:
                    print(f"    {_yellow('warning:')} {w}")
            params = step_data.get("params", {})
            params_str = json.dumps(params, indent=6, default=str)
            if len(params_str) > 500:
                params_str = params_str[:500] + "\n      ... [truncated]"
            print(f"    {_dim('Params:')} {params_str}")

    # Manifest
    if result.manifest:
        print(f"\n{_bold('--- Workflow Manifest ---')}")
        manifest_str = json.dumps(result.manifest, indent=2, default=str)
        if len(manifest_str) > 3000:
            manifest_str = manifest_str[:3000] + "\n... [truncated]"
        print(manifest_str)

    # Tool trace
    if result.tool_trace and verbose:
        print(f"\n{_bold(f'--- Tool Executions ({len(result.tool_trace)}) ---')}")
        for i, ex in enumerate(result.tool_trace, 1):
            tc = ex.tool_call
            duration = f" ({ex.duration_ms:.0f}ms)" if ex.duration_ms else ""
            print(f"\n  {_cyan(f'{i}.')} {_bold(tc.name)}{duration}")

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
                        for k in ("status", "valid", "service_name", "workflow_id",
                                  "step_count", "count", "error"):
                            if k in r:
                                summary_parts.append(f"{k}={r[k]}")
                        if summary_parts:
                            result_str = "{" + ", ".join(summary_parts) + "}"
                        else:
                            result_str = result_str[:300] + "..."
                    else:
                        result_str = result_str[:300] + "..."
                print(f"    {_dim('Result:')} {result_str}")
    elif result.tool_trace and not verbose:
        n = len(result.tool_trace)
        print(f"\n{_dim(f'  [{n} tool call(s) executed -- use /verbose to see details]')}")

    print()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def main() -> None:
    # Load auth token
    token = _load_auth_token()
    config = AgentConfig()
    if token:
        config.bvbrc_auth_token = token

    verbose = False

    print(_bold("BV-BRC Service Agent v2 - Interactive Chat"))
    print(f"{_dim('Architecture:')} Three-phase (Decompose -> Build -> Compose)")
    print(f"{_dim('Model:')} {config.llm_model}")
    print(f"{_dim('Endpoint:')} {config.llm_base_url}")
    print(f"{_dim('Workspace:')} {config.bvbrc_workspace_url}")
    print(f"{_dim('Max iterations:')} {config.max_iterations} (per phase)")
    print(f"{_dim('Auth token:')} {'set' if config.bvbrc_auth_token else 'not set'}")
    print(f"\n{_dim('Type a service request, or /help for commands.')}")
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
  <query>         Run the full three-phase agent (Decompose -> Build -> Compose)
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
            print(f"  Workspace:      {config.bvbrc_workspace_url}")
            print(f"  MCP server:     {config.mcp_server_path}")
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

        # --- Full agent run ---
        print(f"\n{_dim('Phase 1: Decomposing request...')}")
        try:
            result = await run_agent(query, config)
            print_result(result, verbose=verbose)
        except Exception as e:
            print(f"{_red('Error:')} {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
