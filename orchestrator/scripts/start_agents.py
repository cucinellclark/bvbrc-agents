#!/usr/bin/env python3
"""Start all agent MCP servers and tear them down on exit.

Launches the Data, Service2, and Workspace agent MCP HTTP servers as
subprocesses.  All output is prefixed with the agent name for easy
reading.  When this script is interrupted (Ctrl-C) or terminated, it
sends SIGTERM to every child, waits briefly, then SIGKILL if needed.

Usage:
    python scripts/start_agents.py
    python scripts/start_agents.py --agents data service2
    python scripts/start_agents.py --verbose

Prerequisites:
    Each agent needs a Python virtualenv at <agent>/bvbrc-mcp-server/mcp_env/.
    If missing, run the agent's install.sh first:
        cd <agent>/bvbrc-mcp-server && bash install.sh
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path

# ── Agent definitions ─────────────────────────────────────────────────

AGENTS_DIR = Path(__file__).resolve().parent.parent.parent  # .../Agents/

@dataclass
class AgentDef:
    """Definition for an agent MCP server to launch."""
    key: str               # Short key used in --agents filter
    label: str             # Display name
    workdir: Path          # Working directory (bvbrc-mcp-server/)
    venv_python: Path      # Python binary inside the venv
    script: str            # Entry-point script name
    port: int              # Port the server should bind to
    port_override: bool    # Whether to set PORT env var (overrides config.json)

AGENT_DEFS: list[AgentDef] = [
    AgentDef(
        key="data",
        label="Data",
        workdir=AGENTS_DIR / "Data" / "bvbrc-mcp-server",
        venv_python=AGENTS_DIR / "Data" / "bvbrc-mcp-server" / "mcp_env" / "bin" / "python3",
        script="http_server.py",
        port=12009,
        port_override=False,  # config.json already has port=12009
    ),
    AgentDef(
        key="service2",
        label="Service2",
        workdir=AGENTS_DIR / "Service2" / "bvbrc-mcp-server",
        venv_python=AGENTS_DIR / "Service2" / "bvbrc-mcp-server" / "mcp_env" / "bin" / "python3",
        script="http_server.py",
        port=8055,
        port_override=True,  # config.json has port=8053; override to 8055
    ),
    AgentDef(
        key="workspace",
        label="Workspace",
        workdir=AGENTS_DIR / "Workspace" / "bvbrc-mcp-server",
        venv_python=AGENTS_DIR / "Workspace" / "bvbrc-mcp-server" / "mcp_env" / "bin" / "python3",
        script="http_server.py",
        port=8054,
        port_override=True,  # config.json says 8053; override to 8054 to avoid conflict with Service2
    ),
]

# ── Colours for log prefixes ─────────────────────────────────────────

COLOURS = {
    "data":      "\033[36m",   # cyan
    "service2":  "\033[33m",   # yellow
    "workspace": "\033[35m",   # magenta
}
RESET = "\033[0m"
BOLD  = "\033[1m"
RED   = "\033[31m"
GREEN = "\033[32m"
DIM   = "\033[2m"

# ── Subprocess management ────────────────────────────────────────────

async def stream_output(
    stream: asyncio.StreamReader,
    prefix: str,
    colour: str,
    is_stderr: bool = False,
) -> None:
    """Read lines from a subprocess stream and print them with a prefix."""
    stream_tag = "err" if is_stderr else "out"
    while True:
        line = await stream.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="replace").rstrip()
        if is_stderr:
            print(f"{colour}{BOLD}[{prefix}|{stream_tag}]{RESET} {text}")
        else:
            print(f"{colour}[{prefix}]{RESET} {text}")


async def launch_agent(
    agent: AgentDef,
    verbose: bool = False,
) -> asyncio.subprocess.Process | None:
    """Launch a single agent MCP server as a subprocess."""
    colour = COLOURS.get(agent.key, "")
    prefix = f"{agent.label:>10}"

    # ── Pre-flight checks ──
    if not agent.workdir.is_dir():
        print(f"{RED}[{prefix}] ERROR: working directory not found: {agent.workdir}{RESET}")
        return None

    if not agent.venv_python.exists():
        print(
            f"{RED}[{prefix}] ERROR: virtualenv not found: {agent.venv_python}{RESET}\n"
            f"{DIM}[{prefix}]   Run:  cd {agent.workdir} && bash install.sh{RESET}"
        )
        return None

    script_path = agent.workdir / agent.script
    if not script_path.exists():
        print(f"{RED}[{prefix}] ERROR: entry script not found: {script_path}{RESET}")
        return None

    # ── Build environment ──
    env = os.environ.copy()
    if agent.port_override:
        env["PORT"] = str(agent.port)

    # ── Launch ──
    if verbose:
        print(f"{colour}[{prefix}]{RESET} {DIM}cmd: {agent.venv_python} {agent.script}{RESET}")
        print(f"{colour}[{prefix}]{RESET} {DIM}cwd: {agent.workdir}{RESET}")
        print(f"{colour}[{prefix}]{RESET} {DIM}port: {agent.port}{RESET}")

    proc = await asyncio.create_subprocess_exec(
        str(agent.venv_python),
        agent.script,
        cwd=str(agent.workdir),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    print(f"{colour}[{prefix}]{RESET} {GREEN}started{RESET}  (pid={proc.pid}, port={agent.port})")

    # Stream stdout and stderr concurrently
    asyncio.create_task(stream_output(proc.stdout, prefix, colour, is_stderr=False))
    asyncio.create_task(stream_output(proc.stderr, prefix, colour, is_stderr=True))

    return proc


async def shutdown(processes: dict[str, asyncio.subprocess.Process]) -> None:
    """Gracefully shut down all child processes."""
    if not processes:
        return

    print(f"\n{BOLD}Shutting down agents...{RESET}")

    # Send SIGTERM to all
    for key, proc in processes.items():
        if proc.returncode is None:
            colour = COLOURS.get(key, "")
            print(f"{colour}[{key:>10}]{RESET} sending SIGTERM (pid={proc.pid})")
            try:
                proc.terminate()
            except ProcessLookupError:
                pass

    # Wait up to 5 seconds for graceful exit
    for key, proc in processes.items():
        if proc.returncode is None:
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                colour = COLOURS.get(key, "")
                print(f"{colour}[{key:>10}]{RESET} {RED}still running, sending SIGKILL{RESET}")
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass

    # Final wait
    for proc in processes.values():
        if proc.returncode is None:
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass

    print(f"{BOLD}All agents stopped.{RESET}")


# ── Main ──────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start all (or selected) BV-BRC agent MCP servers.",
    )
    parser.add_argument(
        "--agents", "-a",
        nargs="+",
        choices=[a.key for a in AGENT_DEFS],
        default=None,
        help="Only start specific agents (default: all). E.g. --agents data service2",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print extra launch details",
    )
    args = parser.parse_args()

    selected_keys = set(args.agents) if args.agents else {a.key for a in AGENT_DEFS}
    agents_to_start = [a for a in AGENT_DEFS if a.key in selected_keys]

    print(f"{BOLD}BV-BRC Agent Launcher{RESET}")
    print(f"Starting {len(agents_to_start)} agent(s): {', '.join(a.label for a in agents_to_start)}")
    print(f"Press Ctrl-C to stop all agents.\n")

    # Launch all agents
    processes: dict[str, asyncio.subprocess.Process] = {}
    for agent in agents_to_start:
        proc = await launch_agent(agent, verbose=args.verbose)
        if proc is not None:
            processes[agent.key] = proc

    if not processes:
        print(f"{RED}No agents started. Exiting.{RESET}")
        return

    print(f"\n{GREEN}{BOLD}{len(processes)} agent(s) running.{RESET} Waiting...\n")

    # Install signal handlers for clean shutdown
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def on_signal() -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, on_signal)

    # Wait for either a signal or any process to exit unexpectedly
    wait_tasks = {
        key: asyncio.create_task(proc.wait())
        for key, proc in processes.items()
    }
    stop_task = asyncio.create_task(stop_event.wait())

    done, _ = await asyncio.wait(
        [stop_task, *wait_tasks.values()],
        return_when=asyncio.FIRST_COMPLETED,
    )

    # If a process exited (not from our signal), report it
    if stop_task not in done:
        for key, task in wait_tasks.items():
            if task in done:
                rc = task.result()
                colour = COLOURS.get(key, "")
                print(
                    f"\n{colour}[{key:>10}]{RESET} "
                    f"{RED}exited unexpectedly (code={rc}){RESET}"
                )

    # Clean shutdown
    await shutdown(processes)


if __name__ == "__main__":
    asyncio.run(main())
