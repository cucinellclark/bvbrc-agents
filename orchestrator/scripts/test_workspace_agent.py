#!/usr/bin/env python3
"""Test workspace agent via MCP agent_chat — end-to-end diagnostic.

Calls the MCP server's agent_chat tool with agent_type="workspace" to
verify the full workspace browsing pipeline works. Prints detailed logs
at every stage so you can identify exactly where failures occur.

Usage:
    python scripts/test_workspace_agent.py
    python scripts/test_workspace_agent.py --token "un=..."
    python scripts/test_workspace_agent.py --query "list my genome groups"
    python scripts/test_workspace_agent.py --verbose
    python scripts/test_workspace_agent.py --endpoint http://localhost:8053/mcp
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add orchestrator root to sys.path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger("test_workspace")


# ---------------------------------------------------------------------------
# Default test queries — exercise different workspace agent capabilities
# ---------------------------------------------------------------------------

DEFAULT_QUERIES = [
    "What's in my home workspace?",
    "What's in my genome groups folder?",
    "Show me my recent files",
]


# ---------------------------------------------------------------------------
# Auth token resolution
# ---------------------------------------------------------------------------


def load_auth_token(explicit_token: str | None) -> str | None:
    """Resolve auth token: CLI arg > env var > auth_token.txt."""
    if explicit_token:
        return explicit_token

    token = os.environ.get("BV_BRC_AUTH_TOKEN")
    if token:
        return token

    token_file = Path(__file__).parent.parent / "auth_token.txt"
    if token_file.exists():
        token = token_file.read_text().strip()
        if token:
            return token

    return None


# ---------------------------------------------------------------------------
# MCP call via fastmcp Client
# ---------------------------------------------------------------------------


async def call_workspace_agent(
    endpoint: str,
    auth_token: str,
    query: str,
    llm_override: dict | None = None,
    timeout: int = 120,
) -> dict:
    """Call agent_chat with agent_type=workspace via the MCP server.

    Returns the parsed result dict from the agent.
    """
    from fastmcp import Client
    from fastmcp.client.transports import StreamableHttpTransport

    transport = StreamableHttpTransport(
        url=endpoint,
        headers={"Authorization": f"Bearer {auth_token}"},
    )

    client = Client(transport, name="workspace-test", timeout=timeout)

    # Build context
    ctx: dict = {}
    if llm_override:
        ctx["llm_override"] = llm_override

    arguments: dict = {
        "query": query,
        "agent_type": "workspace",
        "token": auth_token,
    }
    if ctx:
        arguments["context"] = json.dumps(ctx)

    logger.info("Connecting to MCP server at %s", endpoint)

    async with client:
        logger.info("Connected. Calling agent_chat(agent_type='workspace')")
        logger.info("Query: %r", query)

        start = time.monotonic()
        result = await client.call_tool_mcp(
            name="agent_chat",
            arguments=arguments,
        )
        elapsed = time.monotonic() - start

    logger.info("agent_chat returned in %.1fs", elapsed)
    return result, elapsed


# ---------------------------------------------------------------------------
# Result parsing and display
# ---------------------------------------------------------------------------


def parse_mcp_result(result) -> dict | None:
    """Extract the JSON dict from an MCP CallToolResult."""
    if not result or not hasattr(result, "content"):
        logger.error("MCP result has no content attribute: %r", result)
        return None

    for block in result.content:
        if hasattr(block, "text"):
            try:
                return json.loads(block.text)
            except (json.JSONDecodeError, TypeError) as e:
                logger.error("Failed to parse result JSON: %s", e)
                logger.error("Raw text: %s", block.text[:2000])
                return None

    logger.error("No text content blocks in MCP result")
    return None


def print_result(data: dict, elapsed: float) -> None:
    """Print a structured summary of the workspace agent result."""
    status = data.get("status", "unknown")
    answer = data.get("answer", "")
    iterations = data.get("iterations_used", "?")
    agent_elapsed = data.get("elapsed_seconds", "?")
    items = data.get("items", [])
    tool_trace = data.get("tool_trace", [])
    paths = data.get("paths_explored", [])

    status_color = {
        "completed": "\033[92m",  # green
        "max_iterations": "\033[93m",  # yellow
        "error": "\033[91m",  # red
        "needs_input": "\033[94m",  # blue
    }.get(status, "\033[0m")

    print(f"\n{'=' * 70}")
    print(f"  Status:      {status_color}{status}\033[0m")
    print(f"  Elapsed:     {elapsed:.1f}s (agent internal: {agent_elapsed}s)")
    print(f"  Iterations:  {iterations}")
    print(f"  Items found: {len(items)}")
    print(f"  Paths:       {paths}")

    if tool_trace:
        print(f"\n  Tool Trace ({len(tool_trace)} calls):")
        for i, call in enumerate(tool_trace):
            tool = call.get("tool", "?")
            args = call.get("arguments", {})
            error = call.get("error")
            duration = call.get("duration_ms")

            # Summarize arguments
            arg_summary = {}
            if args.get("path"):
                arg_summary["path"] = args["path"]
            if args.get("workspace_types"):
                arg_summary["types"] = args["workspace_types"]
            if args.get("name_contains"):
                arg_summary["name_contains"] = args["name_contains"]
            if args.get("num_results"):
                arg_summary["num_results"] = args["num_results"]

            status_str = (
                f"\033[91mERROR: {error}\033[0m" if error else "\033[92mOK\033[0m"
            )
            duration_str = f" ({duration}ms)" if duration else ""
            print(f"    [{i + 1}] {tool}{duration_str} -> {status_str}")
            if arg_summary:
                print(f"        args: {json.dumps(arg_summary)}")

    if items:
        print(f"\n  Items (first 10):")
        for item in items[:10]:
            name = item.get("name", "?")
            item_type = item.get("type", "?")
            size = item.get("size", "")
            size_str = f" ({size} bytes)" if size else ""
            print(f"    - {name} [{item_type}]{size_str}")
        if len(items) > 10:
            print(f"    ... and {len(items) - 10} more")

    print(f"\n  Answer:")
    # Indent the answer for readability
    for line in answer.split("\n"):
        print(f"    {line}")
    print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    parser = argparse.ArgumentParser(
        description="Test workspace agent via MCP agent_chat"
    )
    parser.add_argument(
        "--endpoint",
        default="http://140.221.78.15:8153/mcp",
        help="MCP server endpoint (default: http://140.221.78.15:8153/mcp)",
    )
    parser.add_argument(
        "--token",
        "-t",
        help="BV-BRC auth token. Overrides BV_BRC_AUTH_TOKEN env var.",
    )
    parser.add_argument(
        "--query",
        "-q",
        help="Custom query to test. If not provided, runs default test queries.",
    )
    parser.add_argument(
        "--llm-model",
        help="Override LLM model name (passed as llm_override.model).",
    )
    parser.add_argument(
        "--llm-base-url",
        help="Override LLM base URL (passed as llm_override.base_url).",
    )
    parser.add_argument(
        "--llm-api-key",
        help="Override LLM API key (passed as llm_override.api_key).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout in seconds for MCP call (default: 120).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    # Quiet down noisy libraries unless verbose
    if not args.verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("fastmcp").setLevel(logging.WARNING)

    # Resolve auth token
    auth_token = load_auth_token(args.token)
    if not auth_token:
        logger.error(
            "No auth token. Provide via --token, BV_BRC_AUTH_TOKEN env var, "
            "or orchestrator/auth_token.txt"
        )
        sys.exit(1)

    # Extract user from token for display
    user = "unknown"
    for part in auth_token.split("|"):
        if part.startswith("un="):
            user = part[3:]
            break
    logger.info("Auth token loaded for user: %s", user)

    # Build LLM override if any flags provided
    llm_override = None
    if args.llm_model or args.llm_base_url or args.llm_api_key:
        llm_override = {}
        if args.llm_model:
            llm_override["model"] = args.llm_model
        if args.llm_base_url:
            llm_override["base_url"] = args.llm_base_url
        if args.llm_api_key:
            llm_override["api_key"] = args.llm_api_key
        logger.info("LLM override: %s", llm_override)

    # Determine queries to run
    queries = [args.query] if args.query else DEFAULT_QUERIES

    print(f"\n{'#' * 70}")
    print(f"  Workspace Agent Test")
    print(f"  Endpoint: {args.endpoint}")
    print(f"  User:     {user}")
    print(f"  Queries:  {len(queries)}")
    print(f"{'#' * 70}\n")

    passed = 0
    failed = 0

    for i, query in enumerate(queries, 1):
        print(f"\n--- Test {i}/{len(queries)}: {query!r} ---\n")

        try:
            result, elapsed = await call_workspace_agent(
                endpoint=args.endpoint,
                auth_token=auth_token,
                query=query,
                llm_override=llm_override,
                timeout=args.timeout,
            )

            data = parse_mcp_result(result)
            if data is None:
                logger.error("Failed to parse MCP result")
                failed += 1
                continue

            print_result(data, elapsed)

            status = data.get("status", "error")
            if status in ("completed", "max_iterations"):
                passed += 1
            else:
                failed += 1
                logger.error("Agent returned status=%s", status)

        except Exception as e:
            logger.error("Test failed with exception: %s", e, exc_info=True)
            failed += 1

    # Summary
    total = passed + failed
    print(f"\n{'#' * 70}")
    print(f"  Results: {passed}/{total} passed, {failed}/{total} failed")
    print(f"{'#' * 70}\n")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    asyncio.run(main())
