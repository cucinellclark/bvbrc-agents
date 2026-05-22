"""
CLI runner for the BV-BRC Workspace Exploration Agent.

Usage:
    python -m workspace_agent "Show me my reads files"
    python -m workspace_agent --json "What files are in my home directory?"
    python -m workspace_agent --interactive
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from workspace_agent.agent import run_agent
from workspace_agent.models import AgentConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BV-BRC Workspace Exploration Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python -m workspace_agent "Show me my reads files"\n'
            '  python -m workspace_agent --json "What files are in my Assembly folder?"\n'
            "  python -m workspace_agent --interactive\n"
        ),
    )

    parser.add_argument(
        "query",
        nargs="?",
        help="Natural language workspace exploration question.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="LLM API base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="LLM API key.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model name.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="BV-BRC auth token.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Max agent iterations (default: 8).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output raw JSON instead of pretty-printed summary.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive REPL mode.",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AgentConfig:
    """Build AgentConfig from CLI arguments."""
    overrides: dict = {}
    if args.base_url:
        overrides["llm_base_url"] = args.base_url
    if args.api_key:
        overrides["llm_api_key"] = args.api_key
    if args.model:
        overrides["llm_model"] = args.model
    if args.token:
        overrides["bvbrc_auth_token"] = args.token
    if args.max_iterations is not None:
        overrides["max_iterations"] = args.max_iterations
    return AgentConfig(**overrides)


async def run_query(query: str, config: AgentConfig, json_output: bool) -> None:
    """Run a single query and display the result."""
    print(f"\n{'=' * 60}")
    print(f"Query: {query}")
    print(f"Model: {config.llm_model}")
    print(f"Endpoint: {config.llm_base_url}")
    print(f"Max iterations: {config.max_iterations}")
    print(f"{'=' * 60}\n")

    try:
        result = await run_agent(query, config)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    if json_output:
        print(json.dumps(result.model_dump(), indent=2, default=str))
    else:
        print(result.pretty())


async def interactive_loop(config: AgentConfig, json_output: bool) -> None:
    """Interactive REPL for testing multiple queries."""
    print("BV-BRC Workspace Agent - Interactive Mode")
    print(f"Model: {config.llm_model} @ {config.llm_base_url}")
    print("Type 'quit' or 'exit' to stop. Type 'json' to toggle JSON output.\n")

    use_json = json_output

    while True:
        try:
            query = input("workspace> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break
        if query.lower() == "json":
            use_json = not use_json
            print(f"JSON output: {'ON' if use_json else 'OFF'}")
            continue

        await run_query(query, config, use_json)
        print()


async def main() -> None:
    args = parse_args()
    config = build_config(args)

    if args.interactive:
        await interactive_loop(config, args.json_output)
    elif args.query:
        await run_query(args.query, config, args.json_output)
    else:
        print("Error: provide a query or use --interactive mode.", file=sys.stderr)
        print("Usage: python -m workspace_agent 'your query here'", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
