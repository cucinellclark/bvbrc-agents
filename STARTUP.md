# Basic System Startup

How to start the core Copilot system for local testing. This covers the two essential processes only — no workflow engine, no RAG API, no PM2, no Node.js gateway.

All paths below are relative to the `bvbrc-agents/` repo root.

## First-time setup

```bash
git clone git@github.com:cucinellclark/bvbrc-agents.git
cd bvbrc-agents
./setup.sh
```

This clones the MCP server repo into `mcp_server/`, clones `bvbrc-python-api`, and creates both virtual environments with all dependencies installed.

You also need to provide a BV-BRC auth token. Either:
- Set `BV_BRC_AUTH_TOKEN` in your environment, or
- Place an `auth_token.txt` file in `orchestrator/` and/or `mcp_server/`

You must also create `mcp_server/config/config.json` -- copy from `mcp_server/config/mcp_example.json` and fill in your endpoints.

## Startup order

Each layer depends on the one before it. Start in two separate terminals from the repo root.

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
source orchestrator_env/bin/activate
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
