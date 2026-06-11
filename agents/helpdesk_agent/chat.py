"""
Interactive chat REPL for the Helpdesk Agent.

Provides a simple command-line interface for testing the helpdesk agent
with multi-turn conversation support.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from helpdesk_agent.agent import run_agent
from helpdesk_agent.models import AgentConfig


async def chat_loop(config: AgentConfig | None = None) -> None:
    """Run an interactive chat loop."""
    cfg = config or AgentConfig()

    print("BV-BRC Helpdesk Agent - Interactive Chat")
    print(f"Model: {cfg.llm_model} @ {cfg.llm_base_url}")
    print("Commands: 'quit' to exit, 'json' to toggle JSON output")
    print("-" * 50)

    json_mode = False

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break
        if query.lower() == "json":
            json_mode = not json_mode
            print(f"JSON output: {'ON' if json_mode else 'OFF'}")
            continue

        try:
            result = await run_agent(query, cfg)

            if json_mode:
                print(json.dumps(result.model_dump(), indent=2, default=str))
            else:
                print(f"\nHelpdesk: {result.answer}")
                if result.sources:
                    print(f"\n  Sources: {', '.join(result.sources)}")
                print(f"  ({result.iterations_used} iterations, {result.elapsed_seconds}s)")

        except Exception as e:
            print(f"\nERROR: {type(e).__name__}: {e}", file=sys.stderr)


def main():
    """Entry point for the chat REPL."""
    asyncio.run(chat_loop())


if __name__ == "__main__":
    main()
