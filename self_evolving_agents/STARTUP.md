# Basic System Startup

How to start the core Copilot system for local testing. This covers the two essential processes only — no workflow engine, no RAG API, no PM2, no Node.js gateway.

All paths below are relative to the `bvbrc-agents/` repo root.

## Prerequisites

- Python 3.11 venvs exist for each component (see AGENTS.md for paths)
- `BV_BRC_AUTH_TOKEN` is set in your environment, or `auth_token.txt` files are in place
- Nothing else is already bound to ports 8053 or 9000

## Startup order

Each layer depends on the one before it. Start in two separate terminals.

### 1. MCP Server (port 8053)

```bash
cd mcp_server
source mcp_env/bin/activate
python3 http_server.py
```

Wait for `Uvicorn running on http://0.0.0.0:8053` before continuing. This single process serves all three agents (data, service, workspace) via the unified `agent_chat` tool.

### 2. Orchestrator (port 9000)

```bash
cd orchestrator
./scripts/start_orchestrator.sh
```

Connects to the MCP server at `localhost:8053`, discovers tools, and starts accepting requests. Wait for the startup log to confirm all agents are discovered.

## Smoke tests

```bash
# MCP server responding
curl -s http://localhost:8053/mcp | head

# Orchestrator health
curl -s http://localhost:9000/health | python3 -m json.tool
```

## Shutdown

Kill processes in reverse order (orchestrator, MCP server) with Ctrl-C in each terminal.
