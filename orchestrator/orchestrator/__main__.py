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
import logging.handlers
import sys
from pathlib import Path

import uvicorn

from orchestrator.server import create_app

# Centralized log directory for all agents
_LOG_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "DevEnvironment" / "logs" / "agents"


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

    # Configure logging — console + rotating file in DevEnvironment/logs/agents/
    log_level = getattr(logging, args.log_level.upper())
    log_format = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    formatter = logging.Formatter(log_format)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Console handler at requested level
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Rotating file handler at DEBUG level
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        _LOG_DIR / "orchestrator.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

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
