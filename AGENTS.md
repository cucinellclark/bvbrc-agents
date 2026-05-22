# AGENTS.md

## What this is

Multi-agent AI copilot for BV-BRC (Bacterial-Viral Bioinformatics Resource Center). An LLM-powered orchestrator routes user questions to specialized Python agents (Data, Service, Workspace) that query BV-BRC APIs via MCP protocol.

## Architecture & request flow

```
Orchestrator (Python/FastAPI, port 9000)
     → Consolidated MCP Server (port 8053)  [separate repo]
          → agent_chat(agent_type="data")      → Data Agent
          → agent_chat(agent_type="service")   → Service Agent
          → agent_chat(agent_type="workspace") → Workspace Agent
```

The orchestrator uses a cheap routing LLM (`gpt41mini`) to classify requests, then delegates to the appropriate agent via the unified `agent_chat` MCP tool with an `agent_type` parameter. The single MCP server imports and runs each agent's internal LLM loop directly.

## Repository layout

```
bvbrc-agents/                    (this repo)
├── agents/
│   ├── data_agent/              Data retrieval agent (Solr queries)
│   ├── service_agent/           Service agent v2 — 3-phase workflow builder
│   └── workspace_agent/         Workspace browsing agent (read-only)
├── orchestrator/
│   ├── orchestrator/            Python FastAPI orchestrator package
│   ├── config/                  agents.yaml
│   ├── tests/                   pytest suite (117 tests)
│   ├── scripts/                 start_orchestrator.sh
│   └── pyproject.toml
├── config/                      Shared LLM config (llm.yaml + llm_config.py)
├── self_evolving_agents/        Plans and docs for the self-evolving feature
├── setup.sh                     One-step setup (clones MCP server, installs deps)
└── AGENTS.md
```

After running `./setup.sh`, the following is cloned into this directory (gitignored):

```
├── mcp_server/                  Consolidated FastMCP HTTP server
│   ├── tools/                   MCP tool registrations + agent_chat_tool.py
│   ├── functions/               Solr query, service plan, workspace functions
│   ├── common/                  Shared utilities (auth, config, LLM client)
│   ├── config/                  config.json
│   └── bvbrc-python-api/        BV-BRC Solr Python API (also cloned)
```

The MCP server has its own repo: `git@github.com:cucinellclark/bvbrc-mcp-server.git`

## Setup

```bash
git clone git@github.com:cucinellclark/bvbrc-agents.git
cd bvbrc-agents
./setup.sh
```

This clones the MCP server, creates virtual environments, and installs all dependencies.

## Shared config: `config/llm.yaml`

Single source of truth for LLM settings. All agents import `config/llm_config.py` via `sys.path` manipulation — every agent's `models.py` does:
```python
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent.parent / "config")
sys.path.insert(0, _CONFIG_DIR)
from llm_config import load_llm_defaults
```

The config loader has model-specific quirks in `MODEL_PARAM_EXCLUSIONS` — certain models (gpt5, o3, o4-mini) need parameters stripped or renamed (e.g., `max_tokens` → `max_completion_tokens`). Check `llm_config.py` before adding new models.

## Port mapping

| Component | Port |
|---|---|
| MCP Server | 8053 |
| Orchestrator | 9000 |

The orchestrator's `orchestrator/config/agents.yaml` points all three agent entries at `http://localhost:8053` and uses `chat_tool_params.agent_type` to distinguish between agents.

## Starting the system

See `self_evolving_agents/STARTUP.md` for full instructions.

```bash
# 1. MCP server
cd mcp_server && source mcp_env/bin/activate && python3 http_server.py

# 2. Orchestrator
cd orchestrator && ./scripts/start_orchestrator.sh
```

## Environment variables

| Variable | Purpose |
|---|---|
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | Override `config/llm.yaml` for all agents |
| `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `LLM_TIMEOUT_SECONDS` | Additional LLM overrides |
| `BV_BRC_AUTH_TOKEN` | BV-BRC auth token (also read from `auth_token.txt` files per component) |
| `ORCH_AUTO_SUBMIT` | Set `"true"` to skip user confirmation on workflow submission |

## Running tests

```bash
# Orchestrator — formal pytest suite (117 tests, async auto-mode)
cd orchestrator && source orchestrator_env/bin/activate && pytest
```

## Python environments

| Component | Venv path |
|---|---|
| Orchestrator | `orchestrator/orchestrator_env/` |
| MCP server | `mcp_server/mcp_env/` |

## Non-obvious conventions

- **`agent_chat` is the bridge**: The orchestrator never calls agent Python code directly. It calls the `agent_chat` MCP tool on the consolidated server with an `agent_type` parameter, which internally imports and runs the appropriate agent's `run_agent()` function.
- **`mcp_server_path` in agent configs**: Each agent's `AgentConfig.mcp_server_path` is computed dynamically via `Path(__file__)` to point at `mcp_server/`. This lets agents import `functions.*` modules from the MCP server.
- **Workspace agent returns dual output**: Both natural language text and structured data (items, grids, metadata, previews) for rich UI rendering.
- **No .env files**: All config is YAML/JSON files + optional env var overrides. No dotenv pattern.
- **LLM backend is Argo Gateway API** (Argonne National Lab) — not direct OpenAI/Anthropic. Base URL: `https://apps.inside.anl.gov/argoapi/v1`.
