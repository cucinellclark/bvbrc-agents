#!/usr/bin/env python3
"""Interactive REPL for testing the orchestrator.

Phase 1 commands: agent discovery, health checks, direct tool invocation.
Phase 2 commands: full orchestration loop (ask <query>).

Usage:
    python scripts/chat.py
    python scripts/chat.py --config config/agents.yaml
    python scripts/chat.py --agent data       # Connect to a single agent
    python scripts/chat.py --token "un=..."   # Provide auth token
    BV_BRC_AUTH_TOKEN="un=..." python scripts/chat.py  # Auth via env var
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich import print as rprint

from orchestrator.config import OrchestratorConfig
from orchestrator.registry import AgentRegistry, AgentHandle
from orchestrator.events.events import Event, EventType
from orchestrator.llm import LLMClient, LLMConfig
from orchestrator.orchestrate import orchestrate
from orchestrator.models import OrchestratorRequest

console = Console()


def print_event(event: Event) -> None:
    """Print an event with formatting."""
    color_map = {
        EventType.DISCOVERY_START: "cyan",
        EventType.DISCOVERY_AGENT: "green",
        EventType.DISCOVERY_DONE: "cyan",
        EventType.HEALTH_CHECK: "yellow",
        EventType.ORCHESTRATOR_ERROR: "red",
    }
    color = color_map.get(event.type, "white")
    console.print(f"  [{color}]{event.type.value}[/]: {json.dumps(event.data, default=str)}")


def print_agent_table(registry: AgentRegistry) -> None:
    """Print a table of all registered agents."""
    table = Table(title="Registered Agents", show_lines=True)
    table.add_column("Key", style="cyan", width=12)
    table.add_column("Name", style="white", width=20)
    table.add_column("Endpoint", style="dim", width=30)
    table.add_column("Status", width=10)
    table.add_column("Tools", width=8)
    table.add_column("Latency", width=10)

    for key, agent in registry.agents.items():
        status = "[green]healthy[/]" if agent.is_healthy else "[red]unhealthy[/]"
        latency = f"{agent._last_latency_ms:.0f}ms" if agent._last_latency_ms > 0 else "-"
        table.add_row(
            key,
            agent.name,
            agent.endpoint,
            status,
            str(len(agent.tools)),
            latency,
        )
    console.print(table)


def print_tools(agent: AgentHandle) -> None:
    """Print tools for an agent."""
    if not agent.tools:
        console.print(f"  [dim]No tools discovered for {agent.key}[/]")
        return

    table = Table(title=f"Tools: {agent.name}", show_lines=True)
    table.add_column("Name", style="cyan", width=30)
    table.add_column("Description", width=60)
    table.add_column("Params", width=8)

    for tool in agent.tools:
        param_count = len(tool.inputSchema.get("properties", {})) if tool.inputSchema else 0
        desc = (tool.description or "")[:60]
        if len(tool.description or "") > 60:
            desc += "..."
        table.add_row(tool.name, desc, str(param_count))

    console.print(table)


def print_tool_detail(agent: AgentHandle, tool_name: str) -> None:
    """Print detailed info about a specific tool."""
    tool = next((t for t in agent.tools if t.name == tool_name), None)
    if not tool:
        console.print(f"  [red]Tool '{tool_name}' not found on agent '{agent.key}'[/]")
        return

    console.print(Panel(
        f"[cyan]{tool.name}[/]\n\n"
        f"[white]{tool.description or '(no description)'}[/]\n\n"
        f"[dim]Input Schema:[/]\n"
        f"{json.dumps(tool.inputSchema, indent=2)}",
        title=f"Tool Detail: {tool.name}",
    ))


def print_help() -> None:
    """Print available commands."""
    help_text = """
[cyan]Available Commands:[/]

  [bold green]ask <query>[/]         — Run the full orchestration loop (route → execute → synthesize)

  [green]agents[/]              — List all registered agents and their status
  [green]tools <agent>[/]       — List tools for an agent (e.g., 'tools data')
  [green]tool <agent> <name>[/] — Show detailed tool schema
  [green]call <agent> <tool>[/] — Call a tool interactively (prompts for JSON args)
  [green]health[/]              — Run health checks on all agents
  [green]catalog[/]             — Print the agent catalog (what the routing LLM sees)
  [green]history[/]              — Show conversation history for this session
  [green]clear[/]               — Clear conversation history
  [green]rediscover[/]          — Re-run agent discovery
  [green]ping <agent>[/]        — Ping a specific agent
  [green]help[/]                — Show this help
  [green]quit / exit[/]         — Exit
"""
    console.print(help_text)


async def cmd_call_tool(registry: AgentRegistry, parts: list[str]) -> None:
    """Handle the 'call' command."""
    if len(parts) < 3:
        console.print("  [red]Usage: call <agent_key> <tool_name>[/]")
        return

    agent_key = parts[1]
    tool_name = parts[2]

    try:
        agent = registry.get(agent_key)
    except KeyError as e:
        console.print(f"  [red]{e}[/]")
        return

    # Check tool exists
    if tool_name not in agent.tool_names:
        console.print(f"  [red]Tool '{tool_name}' not found on agent '{agent_key}'[/]")
        console.print(f"  Available: {', '.join(agent.tool_names)}")
        return

    # Prompt for arguments
    console.print(f"  Enter arguments as JSON (or empty for {{}}): ", end="")
    try:
        args_raw = input().strip()
        args = json.loads(args_raw) if args_raw else {}
    except json.JSONDecodeError as e:
        console.print(f"  [red]Invalid JSON: {e}[/]")
        return

    console.print(f"  [dim]Calling {agent_key}.{tool_name}({json.dumps(args)})...[/]")

    try:
        result = await agent.call_tool(tool_name, args)

        # Format result
        if result.content:
            for block in result.content:
                if hasattr(block, "text"):
                    try:
                        parsed = json.loads(block.text)
                        formatted = json.dumps(parsed, indent=2)
                        if len(formatted) > 2000:
                            formatted = formatted[:2000] + "\n... (truncated)"
                        console.print(Syntax(formatted, "json", theme="monokai"))
                    except (json.JSONDecodeError, TypeError):
                        console.print(block.text[:2000])
                else:
                    console.print(f"  [dim]{block}[/]")

            is_error = getattr(result, "isError", False)
            if is_error:
                console.print("  [red]Tool returned an error[/]")
            else:
                console.print("  [green]Tool call succeeded[/]")
        else:
            console.print("  [dim](empty result)[/]")

    except Exception as e:
        console.print(f"  [red]Error: {e}[/]")


async def cmd_ask(
    query: str,
    registry: AgentRegistry,
    llm_client: LLMClient,
    auth_token: str | None = None,
    conversation: list[dict[str, str]] | None = None,
) -> None:
    """Handle the 'ask' command — run the full orchestration loop."""
    if not query.strip():
        console.print("  [red]Usage: ask <your question>[/]")
        return

    if conversation is None:
        conversation = []

    request = OrchestratorRequest(
        query=query,
        auth_token=auth_token,
        recent_messages=conversation.copy(),
    )

    event_color_map = {
        EventType.ORCHESTRATOR_START: "cyan",
        EventType.ROUTING_START: "dim",
        EventType.ROUTING_DECISION: "yellow",
        EventType.AGENT_START: "blue",
        EventType.AGENT_TOOL_CALL: "dim",
        EventType.AGENT_TOOL_RESULT: "dim",
        EventType.AGENT_RESULT: "green",
        EventType.AGENT_ERROR: "red",
        EventType.SYNTHESIS_START: "dim",
        EventType.SYNTHESIS_DONE: "magenta",
        EventType.ORCHESTRATOR_DONE: "cyan bold",
        EventType.ORCHESTRATOR_ERROR: "red",
    }

    async for event in orchestrate(request, registry, llm_client):
        color = event_color_map.get(event.type, "white")

        if event.type == EventType.ROUTING_DECISION:
            decision = event.data.get("decision", "?")
            agent_key = event.data.get("agent_key", "")
            reasoning = event.data.get("reasoning", "")
            steps = event.data.get("steps", [])

            if decision == "pipeline" and steps:
                console.print(f"  [{color}]routing → pipeline ({len(steps)} steps)[/]")
                for j, s in enumerate(steps):
                    deps = s.get("depends_on", [])
                    dep_str = f" (after step {', '.join(str(d) for d in deps)})" if deps else ""
                    console.print(
                        f"  [{color}]  step {j}: [cyan]{s['agent_key']}[/]{dep_str}[/] — {s['task'][:70]}"
                    )
            else:
                console.print(
                    f"  [{color}]routing → {decision}[/]"
                    + (f" → agent=[cyan]{agent_key}[/]" if agent_key else "")
                )
            if reasoning:
                console.print(f"  [dim]  reason: {reasoning}[/]")

        elif event.type == EventType.AGENT_PROGRESS:
            msg = event.data.get("message", "") or event.data.get("warning", "")
            skipped = event.data.get("skipped", False)
            if skipped:
                console.print(f"  [red]  {msg}[/]")
            elif msg:
                console.print(f"  [dim]  {msg}[/]")

        elif event.type == EventType.AGENT_START:
            step_label = f"step {event.step_index}" if event.step_index is not None else ""
            console.print(
                f"  [{color}]agent_start[/] [{event.agent_name}] "
                + (f"({step_label}) " if step_label else "")
                + f"task: {event.data.get('task', '')[:80]}"
            )

        elif event.type == EventType.AGENT_TOOL_CALL:
            tool = event.data.get("tool", "?")
            console.print(f"  [{color}]  calling {tool}...[/]")

        elif event.type == EventType.AGENT_TOOL_RESULT:
            elapsed = event.data.get("elapsed_ms", 0)
            status = event.data.get("status", "?")
            console.print(
                f"  [{color}]  result: status={status}, "
                f"elapsed={elapsed:.0f}ms[/]"
            )

        elif event.type == EventType.AGENT_RESULT:
            result_ui = event.data.get("result_for_ui", {})
            iters = result_ui.get("iterations_used")
            secs = result_ui.get("elapsed_seconds", "?")
            # Show iterations only if the agent reports them (Data agent
            # does; Service2 uses a three-phase model without this field)
            iter_str = f"iterations={iters}, " if iters else ""
            console.print(
                f"  [{color}]agent_result[/] [{event.agent_name}] "
                f"{iter_str}elapsed={secs}s"
            )

        elif event.type == EventType.ORCHESTRATOR_DONE:
            elapsed = event.data.get("elapsed_ms", 0)
            console.print(f"\n  [{color}]done[/] (total: {elapsed:.0f}ms)")
            response_text = event.data.get("response_text", "")
            if response_text:
                console.print(
                    Panel(
                        response_text,
                        title="Response",
                        border_style="green",
                        padding=(1, 2),
                    )
                )

            # If any agent result contains a manifest and it wasn't
            # already included in the response_text, show it separately
            result_for_ui = event.data.get("result_for_ui", {})
            # Check both single-agent and multi-agent result shapes
            manifests = []
            if "manifest" in result_for_ui:
                manifests.append(result_for_ui["manifest"])
            for ar in result_for_ui.get("agent_results", []):
                if "manifest" in ar:
                    manifests.append(ar["manifest"])
            for manifest in manifests:
                if manifest and "```json" not in response_text:
                    manifest_json = json.dumps(manifest, indent=2, default=str)
                    console.print(
                        Panel(
                            Syntax(manifest_json, "json", theme="monokai"),
                            title="Workflow Manifest",
                            border_style="yellow",
                            padding=(1, 2),
                        )
                    )

            # Track conversation history
            conversation.append({"role": "user", "content": query})
            if response_text:
                conversation.append({"role": "assistant", "content": response_text})

        elif event.type == EventType.ORCHESTRATOR_ERROR:
            console.print(
                f"  [{color}]ERROR: {event.data.get('error', '?')}[/]"
            )


async def main():
    parser = argparse.ArgumentParser(description="Orchestrator REPL")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent.parent / "config" / "agents.yaml"),
        help="Path to agents.yaml config file",
    )
    parser.add_argument(
        "--agent",
        help="Connect to a single agent by key (skip others)",
    )
    parser.add_argument(
        "--token", "-t",
        help="BV-BRC auth token (PATRIC token). Overrides BV_BRC_AUTH_TOKEN env var.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Load config
    console.print(Panel("[bold]BV-BRC Orchestrator — Phase 1 REPL[/]", style="cyan"))
    console.print(f"  Config: {args.config}")

    try:
        config = OrchestratorConfig.from_yaml(args.config)
    except FileNotFoundError as e:
        console.print(f"  [red]{e}[/]")
        return

    # Apply auth token override (CLI flag > env var > auth_token.txt > config)
    token = args.token or os.environ.get("BV_BRC_AUTH_TOKEN")
    if not token:
        # Try reading from auth_token.txt
        token_file = Path(__file__).parent.parent / "auth_token.txt"
        if token_file.exists():
            token = token_file.read_text().strip()
    if token:
        for agent_config in config.agents.values():
            if not agent_config.auth_token:
                agent_config.auth_token = token
        source = (
            "--token flag" if args.token
            else "BV_BRC_AUTH_TOKEN env" if os.environ.get("BV_BRC_AUTH_TOKEN")
            else "auth_token.txt"
        )
        console.print(f"  Auth: token provided ({source})")
    else:
        has_any_token = any(a.auth_token for a in config.agents.values())
        if not has_any_token:
            console.print(
                "  [yellow]Warning: No auth token provided. BV-BRC MCP servers "
                "require authentication.[/]\n"
                "  [dim]Use --token or set BV_BRC_AUTH_TOKEN env var.[/]"
            )

    # Filter to single agent if requested
    if args.agent:
        if args.agent not in config.agents:
            console.print(
                f"  [red]Agent '{args.agent}' not in config. "
                f"Available: {', '.join(config.agents.keys())}[/]"
            )
            return
        config.agents = {args.agent: config.agents[args.agent]}
        console.print(f"  Connecting to agent: {args.agent}")

    # Initialize LLM client for routing and synthesis
    llm_config = LLMConfig(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        model=config.llm_model,
        temperature=config.llm_temperature,
        max_tokens=config.llm_max_tokens,
        timeout_seconds=config.llm_timeout_seconds,
    )
    llm_client = LLMClient(llm_config)
    console.print(f"  LLM: {llm_config.model} @ {llm_config.base_url}")

    # Discover agents
    console.print("\n[cyan]Discovering agents...[/]")
    registry = AgentRegistry(config)

    async for event in registry.discover_all():
        print_event(event)

    console.print()
    print_agent_table(registry)
    console.print()
    print_help()

    # In-memory conversation history for the duration of this session
    conversation: list[dict[str, str]] = []

    # REPL loop
    while True:
        try:
            console.print("[bold cyan]orchestrator>[/] ", end="")
            line = input().strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/]")
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()

        if cmd in ("quit", "exit", "q"):
            break

        elif cmd == "ask":
            query = " ".join(parts[1:]) if len(parts) > 1 else ""
            await cmd_ask(query, registry, llm_client, auth_token=token, conversation=conversation)

        elif cmd == "history":
            if not conversation:
                console.print("  [dim](no conversation history)[/]")
            else:
                for i, msg in enumerate(conversation):
                    role_color = "cyan" if msg["role"] == "user" else "green"
                    text = msg["content"][:120]
                    if len(msg["content"]) > 120:
                        text += "..."
                    console.print(f"  [{role_color}]{msg['role']:>9}[/]: {text}")
                console.print(f"  [dim]({len(conversation)} messages)[/]")

        elif cmd == "clear":
            conversation.clear()
            console.print("  [dim]Conversation history cleared.[/]")

        elif cmd == "help":
            print_help()

        elif cmd == "agents":
            print_agent_table(registry)

        elif cmd == "tools":
            if len(parts) < 2:
                console.print("  [red]Usage: tools <agent_key>[/]")
                continue
            try:
                agent = registry.get(parts[1])
                print_tools(agent)
            except KeyError as e:
                console.print(f"  [red]{e}[/]")

        elif cmd == "tool":
            if len(parts) < 3:
                console.print("  [red]Usage: tool <agent_key> <tool_name>[/]")
                continue
            try:
                agent = registry.get(parts[1])
                print_tool_detail(agent, parts[2])
            except KeyError as e:
                console.print(f"  [red]{e}[/]")

        elif cmd == "call":
            await cmd_call_tool(registry, parts)

        elif cmd == "health":
            console.print("  [dim]Running health checks...[/]")
            events = await registry.health_check_all()
            for event in events:
                print_event(event)

        elif cmd == "catalog":
            console.print(Panel(registry.catalog(), title="Agent Catalog"))

        elif cmd == "rediscover":
            console.print("  [dim]Re-discovering agents...[/]")
            await registry.shutdown()
            async for event in registry.discover_all():
                print_event(event)
            print_agent_table(registry)

        elif cmd == "ping":
            if len(parts) < 2:
                console.print("  [red]Usage: ping <agent_key>[/]")
                continue
            try:
                agent = registry.get(parts[1])
                event = await agent.health_check()
                print_event(event)
            except KeyError as e:
                console.print(f"  [red]{e}[/]")

        else:
            console.print(f"  [red]Unknown command: {cmd}. Type 'help' for commands.[/]")

    # Cleanup
    await llm_client.close()
    await registry.shutdown()
    console.print("[dim]Orchestrator shut down.[/]")


if __name__ == "__main__":
    asyncio.run(main())
