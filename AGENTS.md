# AGENTS.md — Agent A: Backend Core

## Your Role

You are **Agent A: Backend Core**. You own the Python agents (data, service, workspace), the orchestrator, and the MCP server. This is the brain of the BV-BRC Copilot.

## Multi-Agent Context

You are one of 4 parallel OpenCode sessions working on the BV-BRC Copilot:

| Agent | Scope | Directory |
|---|---|---|
| **A (you)** | Agents + Orchestrator + MCP Server | `bvbrc-agents/` |
| **B** | Workflow Engine | `bvbrc-workflow-engine/` |
| **C** | Node.js API Gateway | `BV-BRC-Copilot-API/` |
| **D** | Frontend UI (Dojo 1.x) | `bvbrc_website/` |

### Before you start any task:
1. Read `/home/ac.cucinell/bvbrc-dev/Copilot/Agents/INTERFACE_SPEC.md` — the cross-component contract
2. Read `/home/ac.cucinell/bvbrc-dev/Copilot/Agents/DECISIONS.md` — check for decisions that affect you
3. If you change an interface that other agents depend on, update INTERFACE_SPEC.md and DECISIONS.md FIRST

### What you own (can modify freely):
- `agents/data_agent/`, `agents/service_agent/`, `agents/workspace_agent/`
- `orchestrator/`
- `mcp_server/` (tools, functions, common)
- `config/`, `shared/`

### What you must NOT modify:
- `bvbrc-workflow-engine/` — Agent B's territory
- `BV-BRC-Copilot-API/` — Agent C's territory
- `bvbrc_website/` — Agent D's territory

### Your interfaces with other agents:
- **You → Agent B**: You consume the Workflow Engine REST API via `mcp_server/common/workflow_engine_client.py`. See INTERFACE_SPEC.md §4.
- **You → Agent C**: You expose the orchestrator's `/orchestrate/stream` SSE endpoint. See INTERFACE_SPEC.md §2.
- **Agent C → You**: The Gateway calls your orchestrator. You define the contract.
- **Agent D → You** (indirect): Frontend events flow through Agent C to you.

---

## Architecture

```
Orchestrator (Python/FastAPI, port 9000)
     → Consolidated MCP Server (port 8053)
          → agent_chat(agent_type="data")      → Data Agent
          → agent_chat(agent_type="service")   → Service Agent
          → agent_chat(agent_type="workspace") → Workspace Agent
```

The orchestrator uses a routing LLM (`gpt41mini`) to classify requests, then delegates to the appropriate agent via the unified `agent_chat` MCP tool with an `agent_type` parameter.

## Repository layout

```
bvbrc-agents/                    (this repo)
├── agents/
│   ├── data_agent/              Data retrieval agent (Solr queries)
│   ├── service_agent/           Service agent — 3-phase workflow builder (Decompose→Build→Compose)
│   └── workspace_agent/         Workspace browsing agent (read-only, dual output)
├── orchestrator/
│   ├── orchestrator/            Python FastAPI orchestrator package
│   │   ├── server.py            FastAPI HTTP server (port 9000)
│   │   ├── orchestrate.py       Core pipeline: route → execute → synthesize
│   │   ├── router/              LLM-powered routing + keyword fallback
│   │   ├── executor/            Plan executor (sequential/parallel via depends_on)
│   │   ├── synthesizer/         Response synthesis
│   │   ├── mcp/                 MCP client (FastMCP HTTP)
│   │   ├── registry/            Agent discovery + health checks
│   │   ├── llm/                 LLM client wrapper
│   │   ├── events/              Event types for SSE streaming
│   │   ├── session/             (placeholder — not yet implemented)
│   │   └── models.py            OrchestratorRequest/Response
│   ├── config/agents.yaml       Agent registry (all 3 agents @ localhost:8053)
│   ├── tests/                   pytest suite (11 test files)
│   └── scripts/                 start_orchestrator.sh
├── mcp_server/                  Consolidated FastMCP HTTP server (cloned via setup.sh, gitignored)
│   ├── http_server.py           Main server entry point
│   ├── tools/
│   │   ├── agent_chat_tool.py   THE BRIDGE: dispatches to agent run_agent() by agent_type
│   │   ├── data_tools.py        Data MCP tools
│   │   ├── service_tools.py     Service MCP tools
│   │   ├── workspace_tools.py   Workspace MCP tools
│   │   └── ...                  group, SRA, RAG tools
│   ├── functions/
│   │   ├── data_functions.py          Solr query/facet backend
│   │   ├── service_functions.py       Service API
│   │   ├── workflow_functions.py      Workflow engine interaction
│   │   ├── workflow_composition_functions.py  Manifest composition
│   │   ├── workspace_functions.py     Workspace JSON-RPC
│   │   └── ...
│   ├── common/
│   │   ├── workflow_engine_client.py  REST client for workflow engine (port 12008)
│   │   ├── auth.py, config.py, json_rpc.py, token_provider.py
│   │   └── ...
│   └── config/
│       ├── config.json, service_mapping.json, manifest_template.json
│       └── ...
├── config/
│   ├── llm.yaml                 Single source of truth for LLM settings
│   └── llm_config.py            Python loader with model-specific quirks
├── shared/
│   ├── agent_utils.py           Tool-call parsing, argument normalization, fingerprinting
│   └── agent_messages.py        LLM-facing injection messages
├── self_evolving_agents/        Plan-only (PLAN.md + evolution_config.yaml, no code yet)
├── setup.sh                     One-step setup (clones MCP server, installs deps)
└── STARTUP.md                   How to start the system
```

## Key conventions

- **`agent_chat` is the bridge**: The orchestrator calls the `agent_chat` MCP tool with `agent_type` parameter. The MCP server's `agent_chat_tool.py` imports and runs the agent's `run_agent()`.
- **Shared LLM config**: All agents import `config/llm_config.py` via `sys.path` manipulation in their `models.py`.
- **Model quirks**: `MODEL_PARAM_EXCLUSIONS` in `llm_config.py` — gpt5/o3/o4-mini need `max_tokens` and `temperature` stripped. Check before adding models.
- **`mcp_server_path`**: Each agent's `AgentConfig.mcp_server_path` is computed via `Path(__file__)` to import MCP server functions.
- **Workspace agent dual output**: Returns both text and structured data (items, grids, metadata, previews).
- **Service agent 3 phases**: Decompose (LLM → DAG plan) → Build (LLM per-step → validated params) → Compose (programmatic → manifest JSON).
- **LLM backend**: Argo Gateway API at `https://apps.inside.anl.gov/argoapi/v1` (OpenAI-compatible).
- **No .env files**: YAML/JSON + optional env var overrides only.

## Service Agent Workflow Flow (your most complex component)

```
User request
  → classifier.py (gpt41mini): classify intent → plan|submit|status|cancel
  → if plan:
      Phase 1 (decompose.py): LLM loop → WorkflowPlan DAG
      Phase 2 (build.py): Per-step LLM loop → ValidatedStep params
      Phase 3 (compose.py): Programmatic → workflow manifest JSON
      → workflow_engine_client.plan_workflow(manifest) → workflow_id
  → if submit/status/cancel:
      handlers/ → direct workflow engine API calls
```

## Starting the system

```bash
# 1. MCP server
cd mcp_server && source mcp_env/bin/activate && python3 http_server.py

# 2. Orchestrator
cd orchestrator && ./scripts/start_orchestrator.sh
```

## Running tests

```bash
cd orchestrator && source orchestrator_env/bin/activate && pytest
```

## Ports

| Component | Port |
|---|---|
| MCP Server | 8053 |
| Orchestrator | 9000 |
