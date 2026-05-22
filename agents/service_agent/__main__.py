"""
CLI runner for the BV-BRC Service Agent v2.

Usage:
    python -m service_agent "Assemble and annotate E. coli from SRR12345678"
    python -m service_agent --submit "Assemble and annotate E. coli from SRR12345678"
    python -m service_agent --interactive
    python -m service_agent --json "query"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from service_agent.agent import run_agent
from service_agent.models import AgentConfig
from service_agent.submission import (
    submit_workflow,
    validate_workflow,
    validate_workflow_json,
    check_engine_health,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BV-BRC Service Agent v2 - Three-Phase Workflow Construction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python -m service_agent "Assemble genome from SRR12345678"\n'
            '  python -m service_agent --validate "Assemble and annotate E. coli"\n'
            "  python -m service_agent --validate-file workflow.json\n"
            '  python -m service_agent --submit "Assemble genome from SRR12345678"\n'
            '  python -m service_agent --json "query"  # machine-readable output\n'
            "  python -m service_agent --interactive     # REPL mode\n"
        ),
    )

    parser.add_argument(
        "query",
        nargs="?",
        help="Natural language service request.",
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
        "--temperature",
        type=float,
        default=None,
        help="LLM temperature (default: 0.0).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Max iterations per phase sub-loop (default: 10).",
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
    parser.add_argument(
        "--auth-token-file",
        default=None,
        help="Path to auth token file (default: Service2/auth_token.txt).",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit the planned workflow to the workflow engine after planning.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the planned workflow against the workflow engine (no submission).",
    )
    parser.add_argument(
        "--validate-file",
        default=None,
        metavar="FILE",
        help="Validate a workflow JSON file against the engine (no agent run).",
    )
    parser.add_argument(
        "--engine-url",
        default=None,
        help="Workflow engine API URL (default: http://140.221.78.67:12008/api/v1).",
    )

    return parser.parse_args()


def _load_auth_token(token_file: str | None = None) -> str | None:
    """Load auth token from file."""
    candidates = []
    if token_file:
        candidates.append(token_file)
    # Default: look for auth_token.txt in Service2 directory
    service_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates.append(os.path.join(service_dir, "auth_token.txt"))

    for path in candidates:
        if os.path.exists(path):
            with open(path, "r") as f:
                token = f.read().strip()
            if token:
                return token
    return None


def build_config(args: argparse.Namespace) -> AgentConfig:
    """Build AgentConfig from CLI arguments."""
    overrides: dict = {}
    if args.base_url:
        overrides["llm_base_url"] = args.base_url
    if args.api_key:
        overrides["llm_api_key"] = args.api_key
    if args.model:
        overrides["llm_model"] = args.model
    if args.temperature is not None:
        overrides["temperature"] = args.temperature
    if args.max_iterations is not None:
        overrides["max_iterations"] = args.max_iterations

    token = _load_auth_token(args.auth_token_file)
    if token:
        overrides["bvbrc_auth_token"] = token

    if args.engine_url:
        overrides["workflow_engine_url"] = args.engine_url

    return AgentConfig(**overrides)


async def run_query(
    query: str,
    config: AgentConfig,
    json_output: bool,
    do_submit: bool = False,
    do_validate: bool = False,
) -> None:
    """Run a single query and optionally validate/submit the result."""
    print(f"\n{'=' * 60}")
    print(f"Query: {query}")
    print(f"Model: {config.llm_model}")
    print(f"Endpoint: {config.llm_base_url}")
    if do_submit or do_validate:
        print(f"Engine: {config.workflow_engine_url}")
    print(f"{'=' * 60}\n")

    try:
        result = await run_agent(query, config)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    if result.status == "completed" and result.manifest:
        if do_validate:
            print("\n--- Validating workflow against engine ---", file=sys.stderr)
            validation = await validate_workflow(result, config)
            _print_validation_result(validation, json_output)
        if do_submit:
            # TODO: enable real submission when ready
            # submission = await submit_workflow(result, config)
            # result.submission = submission
            from service_agent.submission import extract_workflow_definition
            definition = extract_workflow_definition(result)
            print("\n--- Would submit to engine (dry run) ---")
            print(json.dumps(definition, indent=2, default=str))

    if json_output:
        print(json.dumps(result.model_dump(), indent=2, default=str))
    else:
        print(result.pretty())


def _print_validation_result(validation: dict, json_output: bool) -> None:
    """Pretty-print or dump the engine validation response."""
    if json_output:
        print(json.dumps(validation, indent=2, default=str))
        return

    is_valid = validation.get("valid", False)
    print(f"\n{'=' * 60}")
    print(f"  Validation: {'VALID' if is_valid else 'INVALID'}")

    if validation.get("warnings"):
        for w in validation["warnings"]:
            print(f"  Warning: {w}")
    if validation.get("auto_fixes"):
        for f in validation["auto_fixes"]:
            print(f"  Auto-fix: {f}")
    if validation.get("error"):
        print(f"  Error: {validation['error']}")
    if validation.get("message"):
        print(f"  Message: {validation['message']}")
    print(f"{'=' * 60}")


async def validate_file(path: str, config: AgentConfig, json_output: bool) -> None:
    """Load a workflow JSON file and validate it against the engine."""
    try:
        with open(path, "r") as fh:
            workflow_json = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"ERROR: Cannot load {path}: {exc}", file=sys.stderr)
        sys.exit(1)

    # If the file is a full AgentResult dump, unwrap the manifest
    if isinstance(workflow_json, dict) and "manifest" in workflow_json:
        inner = workflow_json["manifest"]
        if isinstance(inner, dict) and "manifest" in inner:
            workflow_json = inner["manifest"]
        elif isinstance(inner, dict) and "steps" in inner:
            workflow_json = inner

    print(f"Validating {path} against {config.workflow_engine_url} ...")
    validation = await validate_workflow_json(workflow_json, config)
    _print_validation_result(validation, json_output)


async def interactive_loop(
    config: AgentConfig,
    json_output: bool,
    do_submit: bool = False,
) -> None:
    """Interactive REPL for testing queries."""
    print("BV-BRC Service Agent v2 (interactive mode)")
    print(f"Model: {config.llm_model} @ {config.llm_base_url}")
    if do_submit:
        print(f"Auto-submit: ON  (engine: {config.workflow_engine_url})")
    print("Type 'quit' or 'exit' to stop. Type 'json' to toggle JSON output.")
    print("Type 'submit' to toggle auto-submission.\n")

    use_json = json_output
    auto_submit = do_submit

    while True:
        try:
            query = input("query> ").strip()
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
        if query.lower() == "submit":
            auto_submit = not auto_submit
            print(f"Auto-submit: {'ON' if auto_submit else 'OFF'}")
            continue

        await run_query(query, config, use_json, do_submit=auto_submit)
        print()


async def main() -> None:
    args = parse_args()
    config = build_config(args)

    if args.validate_file:
        await validate_file(args.validate_file, config, args.json_output)
    elif args.interactive:
        await interactive_loop(config, args.json_output, do_submit=args.submit)
    elif args.query:
        await run_query(
            args.query, config, args.json_output,
            do_submit=args.submit, do_validate=args.validate,
        )
    else:
        print("Error: provide a query or use --interactive mode.", file=sys.stderr)
        print("Usage: python -m service_agent 'your query here'", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
