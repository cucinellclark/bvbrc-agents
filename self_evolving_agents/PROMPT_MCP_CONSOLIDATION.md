# MCP Server Consolidation — Implementation Prompt

## Task

Consolidate the three per-agent MCP servers into the single canonical server at `mcp-server/bvbrc-mcp-server/`. When finished, there will be one MCP server process serving all three agents (data, service, workspace), the orchestrator will point all agent entries at `http://localhost:8053`, and the old per-agent MCP directories will be deleted.

Read `PLAN_MCP_CONSOLIDATION.md` in the project root for the full architectural plan. Execute all 11 phases in order. Below are the specifics.

## Project Root

`/home/ac.cucinell/bvbrc-dev/Copilot/Agents`

## Constraints

- Do NOT create new files unless the plan calls for them. Prefer editing existing files.
- Do NOT change any agent behavior — this is a structural consolidation only.
- Do NOT modify the agent Python packages themselves (`Data/data_agent/`, `Service2/service_agent/`, `Workspace/workspace_agent/`). Only MCP server code and orchestrator config/executor are touched.
- Preserve all per-agent response formatting differences in the unified `agent_chat_tool.py` (Service2 returns `manifest`, `workflow_plan`, `question`, `workflow_id`, `persisted`; Workspace returns `items`, `metadata`, `ui_grids`, `previews`, `paths_explored`).
- Use the existing `sys.path.insert` pattern for cross-directory imports (same as `config/llm_config.py`).

## Phase 1: Merge Agent-Specific Changes into Canonical

All edits target files in `mcp-server/bvbrc-mcp-server/`.

### 1a: `common/llm_client.py`

Replace with the version from `Service2/bvbrc-mcp-server/common/llm_client.py` (or `Workspace/` — they're identical). This adds:
- `sys.path` import of `config/llm_config.py`
- `get_excluded_params()` usage to conditionally strip `temperature`/`max_tokens` for models that reject them (gpt5, o3, o4-mini)
- `uses_max_completion_tokens()` support
- `get_temperature_override()` support

### 1b: `common/workflow_engine_client.py`

Replace with the version from `Service2/bvbrc-mcp-server/common/workflow_engine_client.py`. This migrates from aiohttp to httpx. The business logic is identical — same methods, same endpoints, same sanitization. Only the HTTP library differs. Service2 migrated because aiohttp's async DNS resolver deadlocks when sharing an event loop with the OpenAI SDK.

### 1c: `functions/service_functions.py`

Migrate the remaining aiohttp usage to httpx. The file has:
- Line 9: `import aiohttp`
- Line ~467: `async with aiohttp.ClientSession() as session:` used for fetching task stdout/stderr logs

Change to:
- `import httpx`
- `async with httpx.AsyncClient() as client:`
- `response.status` -> `response.status_code`
- `await response.json()` -> `response.json()`
- `await response.text()` -> `response.text`

### 1d: `tools/data_tools.py`

Merge additions from `Data/bvbrc-mcp-server/tools/data_tools.py`:
1. Add the `normalize_select` import (wherever the Data version imports it from)
2. Add the `_auto_quote_query()` helper function (~47 lines)
3. Add the `solr_query` and `solr_facet_query` MCP tools (~150 lines)

These are additive — no existing code changes. Diff the two files to see exactly what Data added.

### 1e: `tools/service_tools.py`

Find the commented-out `submit_workflow` tool registration (line ~776):
```python
# @mcp.tool(name="submit_workflow")
```
Uncomment it so the tool is registered.

### 1f: `functions/` — Add Service2's extra files

Copy from `Service2/bvbrc-mcp-server/functions/` into `mcp-server/bvbrc-mcp-server/functions/`:
- `service_validation_functions.py`
- `workflow_composition_functions.py`

### 1g: `functions/service_plan_functions.py`

Find the `genome_size` parameter default and change it from `"5M"` (string) to `5000000` (int). The Service2 copy has this fix.

### 1h: `config/config.json`

Change `mcp_url` from `"140.221.78.67"` to `"0.0.0.0"`.

## Phase 2: Create Unified `tools/agent_chat_tool.py`

Create a NEW file: `mcp-server/bvbrc-mcp-server/tools/agent_chat_tool.py`

This is a dispatcher that registers one `agent_chat` MCP tool accepting an `agent_type` parameter. Use the three existing implementations as source:
- `Data/bvbrc-mcp-server/tools/agent_chat_tool.py` (160 lines) — data agent path
- `Service2/bvbrc-mcp-server/tools/agent_chat_tool.py` (192 lines) — service agent path
- `Workspace/bvbrc-mcp-server/tools/agent_chat_tool.py` (150 lines) — workspace agent path

Structure:

```python
def register_agent_chat_tool(mcp, token_provider):
    @mcp.tool(name="agent_chat")
    async def agent_chat(query, agent_type="data", context="", token="", mcp_ctx=None):
        if agent_type == "data":
            # ... data agent import + call + response formatting
        elif agent_type == "service":
            # ... service agent import + call + response formatting
        elif agent_type == "workspace":
            # ... workspace agent import + call + response formatting
```

The `sys.path` setup at the top of the file must add:
- `Agents/Data/` — for `from data_agent.agent import run_agent`
- `Agents/Service2/` — for `from service_agent.agent import run_agent`
- `Agents/Workspace/` — for `from workspace_agent.agent import run_agent`
- `Agents/config/` — for `from llm_config import load_llm_defaults`

Use `Path(__file__).resolve()` to compute these relative to the file location.

Preserve ALL per-agent response formatting:
- **Data:** Returns `answer`, `status`, `sources`, `iterations_used`, `elapsed_seconds`, `tool_trace`
- **Service:** Returns all of the above plus `manifest`, `workflow_plan`, `question`, `workflow_id`, `persisted`. Uses `result.pretty()` for answer text. Handles `needs_input` status with `result.question`.
- **Workspace:** Returns all standard fields plus `items`, `metadata`, `ui_grids`, `previews`, `paths_explored`. Sources is always `[]`.

Read all three source files carefully and merge their logic. The common parts (config construction, LLM override handling, progress callback setup) should be shared. The agent-specific parts (import, call, response formatting) should be in separate helper functions or if/elif branches.

## Phase 3: Update `http_server.py`

Edit `mcp-server/bvbrc-mcp-server/http_server.py`:
1. Add import: `from tools.agent_chat_tool import register_agent_chat_tool`
2. After the existing tool registration block (after the `register_sra_tools(...)` call around line 75), add:
   ```python
   register_agent_chat_tool(mcp, token_provider)
   ```

## Phase 4: Add `bvbrc-python-api`

```bash
cd /home/ac.cucinell/bvbrc-dev/Copilot/Agents/mcp-server/bvbrc-mcp-server
git clone git@github.com:cucinellclark/bvbrc-python-api.git
```

## Phase 5: Update `tools/__init__.py`

Edit `mcp-server/bvbrc-mcp-server/tools/__init__.py`:
Add `from tools.agent_chat_tool import register_agent_chat_tool` and include it in `__all__` if that exists.

## Phase 6: Update Orchestrator `config/agents.yaml`

Edit `Orchestrator/config/agents.yaml`. Change ALL three agent entries to:
- `endpoint: "http://localhost:8053"` (same for all three)
- Add `chat_tool_params:` with the appropriate `agent_type` for each

Full target content for the agents section:

```yaml
agents:
  data:
    name: "Data Agent"
    description: >
      Retrieves biological data from BV-BRC. Translates natural language
      questions into Solr queries against genome, feature, AMR, pathway,
      epitope, and other BV-BRC collections. Returns structured data results
      including counts, facets, and tabular data.
    endpoint: "http://localhost:8053"
    protocol: "mcp"
    capabilities:
      - data_retrieval
      - solr_query
      - faceted_search
      - collection_browsing
    max_iterations: 5
    timeout_seconds: 120
    auth_token_env: "BV_BRC_AUTH_TOKEN"
    chat_tool: "agent_chat"
    mcp_server_name: "bvbrc_server"
    chat_tool_params:
      agent_type: "data"

  service2:
    name: "Service Agent"
    description: >
      Constructs BV-BRC bioinformatics service workflows. Takes natural
      language descriptions of analyses and builds validated workflow plans
      for genome assembly, annotation, BLAST, phylogenetics, and 30+ other
      BV-BRC services. Returns workflow configurations for user review.
      Handles multi-step pipelines with dependencies between services.
    endpoint: "http://localhost:8053"
    protocol: "mcp"
    capabilities:
      - workflow_planning
      - service_configuration
      - parameter_validation
      - multi_step_pipelines
    max_iterations: 10
    timeout_seconds: 300
    auth_token_env: "BV_BRC_AUTH_TOKEN"
    chat_tool: "agent_chat"
    mcp_server_name: "bvbrc_server"
    chat_tool_params:
      agent_type: "service"

  workspace:
    name: "Workspace Agent"
    description: >
      Explores the user's BV-BRC cloud workspace (personal file system).
      Browses directories, searches for files by name/type/extension,
      retrieves file metadata, and previews file contents. Returns both
      a natural language summary and structured data (file listings,
      metadata, UI grid payloads) for rich rendering. Read-only -- does
      not create, modify, or delete files.
    endpoint: "http://localhost:8053"
    protocol: "mcp"
    capabilities:
      - workspace_browsing
      - file_search
      - file_metadata
      - file_preview
    max_iterations: 8
    timeout_seconds: 120
    auth_token_env: "BV_BRC_AUTH_TOKEN"
    chat_tool: "agent_chat"
    mcp_server_name: "bvbrc_server"
    chat_tool_params:
      agent_type: "workspace"
```

Keep the `orchestrator:` section unchanged.

## Phase 7: Orchestrator Code Changes

### 7a: `Orchestrator/orchestrator/config.py`

Add `chat_tool_params` field to the `AgentConfig` class (around line 46). Add `Any` to the typing import if needed:

```python
class AgentConfig(BaseModel):
    name: str
    description: str
    endpoint: str
    protocol: str = "mcp"
    capabilities: list[str] = Field(default_factory=list)
    max_iterations: int = 5
    timeout_seconds: int = 120
    auth_token: str | None = None
    chat_tool: str = "agent_chat"
    mcp_server_name: str | None = None
    chat_tool_params: dict[str, Any] = Field(default_factory=dict)
```

### 7b: `Orchestrator/orchestrator/executor/agent_executor.py`

In `execute_agent_step()`, after line 64 (`arguments: dict[str, Any] = {"query": step.task}`), add:

```python
    # Inject agent-specific chat tool params (e.g., agent_type)
    if agent.config.chat_tool_params:
        arguments.update(agent.config.chat_tool_params)
```

This must come BEFORE the context and token are added to arguments (before line 67).

## Phase 8: Set Up Consolidated Venv

```bash
cd /home/ac.cucinell/bvbrc-dev/Copilot/Agents/mcp-server/bvbrc-mcp-server
python3.11 -m venv mcp_env
source mcp_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install openai>=1.0.0
pip install -e bvbrc-python-api/
```

Also install the utilities/distllm dependencies if the RAG tools need them:
```bash
pip install -r utilities/requirements.txt
```

Then verify agent packages are importable by running Python with the sys.path entries that `agent_chat_tool.py` will use:
```python
import sys
from pathlib import Path
agents_root = Path(".").resolve().parent.parent  # Agents/
sys.path.insert(0, str(agents_root / "Data"))
sys.path.insert(0, str(agents_root / "Service2"))
sys.path.insert(0, str(agents_root / "Workspace"))
sys.path.insert(0, str(agents_root / "config"))

from data_agent.agent import run_agent as data_run
from service_agent.agent import run_agent as service_run
from workspace_agent.agent import run_agent as workspace_run
print("All agent imports OK")
```

If any import fails, install its missing dependencies into the mcp_env venv.

## Phase 9: Test Server Startup

```bash
cd /home/ac.cucinell/bvbrc-dev/Copilot/Agents/mcp-server/bvbrc-mcp-server
source mcp_env/bin/activate
PORT=8053 python3 http_server.py
```

Verify:
- No import errors on startup
- Server binds to `0.0.0.0:8053`
- The startup log shows `agent_chat` among the registered tools

## Phase 10: Run Orchestrator Tests

```bash
cd /home/ac.cucinell/bvbrc-dev/Copilot/Agents/Orchestrator
source orchestrator_env/bin/activate
pytest tests/ -v
```

All 117 tests should pass. The orchestrator tests use mocks and don't need a live MCP server. They validate that:
- `AgentConfig` correctly loads `chat_tool_params` from YAML
- The executor correctly merges `chat_tool_params` into arguments
- The rest of the orchestrator pipeline is unaffected

If any tests fail, fix the issue before proceeding.

## Phase 11: Delete Old MCP Server Directories

```bash
rm -rf /home/ac.cucinell/bvbrc-dev/Copilot/Agents/Data/bvbrc-mcp-server
rm -rf /home/ac.cucinell/bvbrc-dev/Copilot/Agents/Service2/bvbrc-mcp-server
rm -rf /home/ac.cucinell/bvbrc-dev/Copilot/Agents/Workspace/bvbrc-mcp-server
rm -rf /home/ac.cucinell/bvbrc-dev/Copilot/Agents/Service/bvbrc-mcp-server
```

## Phase 12: Update AGENTS.md

Update the `AGENTS.md` file in the project root to reflect:
- Port mapping table: single entry `mcp-server` on port 8053 instead of three separate ports
- Starting the system: one MCP server command instead of three
- Key directories: remove `*/bvbrc-mcp-server/` references, add `mcp-server/bvbrc-mcp-server/`
- Environment variables: `PORT` is no longer needed for individual agents

## Verification Checklist

After all phases, confirm:
- [ ] `mcp-server/bvbrc-mcp-server/http_server.py` starts without errors
- [ ] `tools/list` MCP call returns all tools including `agent_chat`
- [ ] `Orchestrator/tests/` all pass with `pytest -v`
- [ ] `Data/bvbrc-mcp-server/` does not exist
- [ ] `Service2/bvbrc-mcp-server/` does not exist
- [ ] `Workspace/bvbrc-mcp-server/` does not exist
- [ ] `AGENTS.md` reflects the new single-server setup
