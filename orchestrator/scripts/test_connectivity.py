#!/usr/bin/env python3
"""Quick connectivity test — verifies MCP connections to all configured agents.

This is a non-interactive smoke test. Run it to verify your agents are
reachable before using the full REPL.

Usage:
    python scripts/test_connectivity.py
    python scripts/test_connectivity.py --token "un=..."
    BV_BRC_AUTH_TOKEN="un=..." python scripts/test_connectivity.py
    python scripts/test_connectivity.py --verbose
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from orchestrator.config import OrchestratorConfig
from orchestrator.registry import AgentRegistry
from orchestrator.events.events import EventType

console = Console()


async def main():
    parser = argparse.ArgumentParser(description="Test MCP connectivity")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent.parent / "config" / "agents.yaml"),
    )
    parser.add_argument(
        "--token", "-t",
        help="BV-BRC auth token (PATRIC token). Overrides BV_BRC_AUTH_TOKEN env var.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    import logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    console.print("[bold cyan]BV-BRC Orchestrator — Connectivity Test[/]\n")

    try:
        config = OrchestratorConfig.from_yaml(args.config)
    except FileNotFoundError as e:
        console.print(f"[red]Config not found: {e}[/]")
        sys.exit(1)

    # Apply auth token
    token = args.token or os.environ.get("BV_BRC_AUTH_TOKEN")
    if token:
        for agent_config in config.agents.values():
            if not agent_config.auth_token:
                agent_config.auth_token = token
        console.print(f"Auth: token provided")
    else:
        console.print("[yellow]Warning: No auth token. Use --token or BV_BRC_AUTH_TOKEN env var.[/]")

    console.print(f"Testing {len(config.agents)} agent(s)...\n")
    registry = AgentRegistry(config)

    results = {"pass": 0, "fail": 0}

    async for event in registry.discover_all():
        if event.type == EventType.DISCOVERY_AGENT:
            agent_name = event.data["agent"]
            tool_count = event.data["tool_count"]
            console.print(
                f"  [green]PASS[/] {agent_name}: "
                f"connected, {tool_count} tools discovered"
            )
            results["pass"] += 1

        elif event.type == EventType.ORCHESTRATOR_ERROR:
            console.print(f"  [red]FAIL[/] {event.data.get('error', 'unknown error')}")
            results["fail"] += 1

        elif event.type == EventType.DISCOVERY_DONE:
            if args.verbose:
                for key, info in event.data.get("agents", {}).items():
                    console.print(f"\n  [dim]{key}:[/]")
                    for k, v in info.items():
                        console.print(f"    {k}: {v}")

    console.print(
        f"\n[bold]Results:[/] "
        f"[green]{results['pass']} passed[/], "
        f"[red]{results['fail']} failed[/] "
        f"/ {results['pass'] + results['fail']} total"
    )

    await registry.shutdown()

    if results["fail"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
