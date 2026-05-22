"""Entry point for running the orchestrator HTTP server.

Usage:
    python -m orchestrator.server
    python -m orchestrator.server --port 9000
    python -m orchestrator.server --config config/agents.yaml --host 0.0.0.0

Environment variables:
    BV_BRC_AUTH_TOKEN  — Default auth token for agent connections
    ORCHESTRATOR_PORT  — Default port (overridden by --port)
    ORCHESTRATOR_HOST  — Default host (overridden by --host)
"""

import argparse
import logging
import sys
from pathlib import Path

import uvicorn

from orchestrator.server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BV-BRC Copilot Orchestrator HTTP Server"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="Port to listen on (default: 9000)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to agents.yaml config file (default: config/agents.yaml)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: info)",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Resolve config path
    config_path = args.config
    if config_path is None:
        default = Path(__file__).parent.parent / "config" / "agents.yaml"
        if default.exists():
            config_path = str(default)

    # Create and run
    app = create_app(config_path=config_path)

    print(
        f"Starting BV-BRC Orchestrator on {args.host}:{args.port}\n"
        f"  Config: {config_path or '(defaults)'}\n"
        f"  Docs:   http://{args.host}:{args.port}/docs\n"
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
