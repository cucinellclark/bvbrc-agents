# Basic System Startup

How to start the core Copilot system for local testing. Only the orchestrator is needed for Copilot chat — agents run in-process (no separate MCP server required).

All paths below are relative to the `bvbrc-agents/` repo root.

## First-time setup

```bash
git clone git@github.com:cucinellclark/bvbrc-agents.git
cd bvbrc-agents
./setup.sh
```

This clones the MCP server repo into `mcp_server/`, clones `bvbrc-python-api`, and creates virtual environments with all dependencies installed.

You also need to provide a BV-BRC auth token. Either:
- Set `BV_BRC_AUTH_TOKEN` in your environment, or
- Place an `auth_token.txt` file in `orchestrator/`

## Startup

Only one process is needed. All 6 agents (data, service, workspace, helpdesk, analysis, planning) run in-process inside the orchestrator via `shared.agent_dispatch.dispatch_agent()`.

### Orchestrator (port 9100)

```bash
cd orchestrator
source orchestrator_env/bin/activate
./scripts/start_orchestrator.sh --port 9100
```

Wait for the startup log to confirm all agents are discovered. The orchestrator imports agent code directly — no MCP connection needed.

### LLM Admission Proxy (optional, port 8005)

Limits concurrent LLM requests to vLLM on mango. Sits on holly between the orchestrator/agents and the upstream vLLM. Without the proxy, concurrent chat turns can stampede the GPU and cause 500s.

```bash
# Start the proxy (uses the orchestrator venv)
cd llm_admission
./start_admission_proxy.sh --background --port 8005
```

The proxy reads its config from environment variables:

| Env var | Default | Purpose |
|---|---|---|
| `LLM_UPSTREAM_URL` | `http://mango.cels.anl.gov:8004/v1` | Real vLLM endpoint |
| `ADMISSION_MAX_IN_FLIGHT` | `4` | Max concurrent requests forwarded to upstream |
| `ADMISSION_MAX_WAITING` | `16` | Max queued requests before 429 rejection |
| `ADMISSION_PORT` | `8005` | Proxy listen port |

**Wire the orchestrator to the proxy:** Set `LLM_ADMISSION_URL` before starting the orchestrator:

```bash
export LLM_ADMISSION_URL=http://127.0.0.1:8005/v1
cd orchestrator
./scripts/start_orchestrator.sh --port 9100
```

When `LLM_ADMISSION_URL` is set, all LLM traffic (routing, agents, classifier) is rewritten to go through the proxy. The proxy forwards to its configured upstream. Unset the env var and restart the orchestrator to bypass the proxy.

**Rollback:** `unset LLM_ADMISSION_URL` + restart orchestrator. No code changes needed.

### MCP Server (optional, port 8153)

Only needed for external MCP clients (e.g., Claude Desktop). Not needed for Copilot chat.

```bash
cd mcp_server
source mcp_env/bin/activate
python3 http_server.py
```

## Smoke tests

```bash
# Orchestrator health
curl -s http://localhost:9100/health | python3 -m json.tool

# Admission proxy health (if running)
curl -s http://localhost:8005/health | python3 -m json.tool
```

## Shutdown

Kill the orchestrator with Ctrl-C. If the admission proxy is running in background mode:

```bash
cd llm_admission
./start_admission_proxy.sh --stop
```
