# Analysis Agent — Design Plan

## Purpose

The Analysis Agent inspects the outputs of completed BV-BRC service jobs and produces a human-readable summary with structured data for UI rendering. It is triggered in two ways:

1. **Automatically** — when the workflow engine fires its completion webhook and the orchestrator resumes the conversation, the analysis agent reads the job outputs and gives the user a rich "here's what your job produced" message.
2. **On demand** — a user can ask "analyze the results of my last assembly" and the orchestrator routes to this agent.

The agent does **shallow content inspection**: it browses output directories, reads key output files (reports, TSVs, summary stats), and extracts service-specific metrics (N50, gene counts, BLAST hits, etc.) without deep/long-running computation.

---

## Architecture

### Position in the system

```
Orchestrator
  → agent_chat(agent_type="analysis")
    → MCP Server dispatches to analysis_agent.agent.run_agent()
      → LLM loop with tools (browse, read, metadata)
      → returns AgentResult with text + structured data
```

The analysis agent is a peer of the four existing agents (data, service, workspace, helpdesk). It is registered in `agents.yaml`, dispatched via `agent_chat_tool.py`, and routed to by the orchestrator like any other agent.

### Agent type identifier

```
agent_type: "analysis"
```

### Directory layout

```
bvbrc-agents/agents/analysis_agent/
├── __init__.py
├── agent.py              # run_agent() entry point — iterative plan-execute-evaluate loop
├── models.py             # AgentConfig, AgentResult, AgentState (shared pattern)
├── prompts.py            # System prompt with service-specific output knowledge
├── tool_registry.py      # Tool definitions and schemas
├── output_knowledge.py   # Service output patterns + metric extraction hints
└── tools/
    ├── __init__.py
    ├── browse.py          # workspace_browse, get_file_metadata (reused from workspace_agent)
    └── read.py            # read_output_file (wrapper around workspace read with analysis focus)
```

---

## Trigger Mechanism: Workflow Completion → Analysis

### Current flow (no analysis agent)

```
Workflow Engine  →  POST /workflow-complete (webhook)  →  Gateway
Gateway writes a static completion message to the chat session:
  "Your workflow X completed successfully. Output files: [paths]"
```

### Proposed flow (with analysis agent)

```
Workflow Engine  →  POST /workflow-complete (webhook)  →  Gateway
Gateway sends a synthetic orchestrator request:
  POST /orchestrate/stream
  {
    query: <constructed analysis prompt>,
    target_agent: "analysis",
    session_id: <from webhook>,
    auth_token: <from webhook>,
    workflow_context: {                    // NEW field on OrchestratorRequest
      workflow_id, workflow_name, status,
      steps: [{step_name, app_name, status, task_id, output_path, output_file}],
      output_paths: [string]
    }
  }
```

The `workflow_context` provides the orchestrator and agent with structured metadata about what just completed so it doesn't have to discover it. The orchestrator's routing LLM (or keyword fallback) decides which agent handles the request based on the workflow status and query content — it will route succeeded workflows to the analysis agent, and may handle failures differently (direct response, helpdesk, etc.).

### Changes required per component

| Component | Change |
|---|---|
| **OrchestratorRequest** (Agent A) | Add optional `workflow_context: dict` field |
| **agent_executor.py** (Agent A) | Pass `workflow_context` into the agent's `context` dict |
| **Gateway** (Agent C) | On webhook with `status: "succeeded"`, POST to orchestrator instead of writing static message. On `"failed"`/`"cancelled"`, keep current behavior |
| **agents.yaml** (Agent A) | Register analysis agent entry |
| **agent_chat_tool.py** (Agent A) | Add `elif agent_type == "analysis":` dispatch |
| **router prompts** (Agent A) | Add analysis agent to the routing LLM's agent catalog so on-demand requests route correctly |

---

## Agent Internals

### Loop architecture

The analysis agent uses the same **iterative plan-execute-evaluate loop** as the data, workspace, and helpdesk agents:

```
1. Build system prompt (with service output knowledge) + user query
2. If workflow_context is provided, inject it as a structured context message
3. For each iteration (up to max_iterations):
   a. Call LLM with messages + tool schemas
   b. If no tool_calls → final answer, break
   c. Execute each tool_call, record results
   d. Feed results back into conversation
4. If max iterations hit → force synthesis with tool_choice="none"
5. Return AgentResult via state.to_result()
```

### Entry point

```python
async def run_agent(
    query: str,
    config: AgentConfig | None = None,
    context: dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AgentResult
```

Follows the identical signature as all other agents.

### Config (models.py)

```python
class AgentConfig(BaseModel):
    # Standard fields (from llm_config.py defaults)
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    temperature: float
    max_tokens: int
    bvbrc_auth_token: str | None = None
    mcp_server_path: str  # computed via Path(__file__)

    # Analysis-specific
    max_iterations: int = 6
    max_tool_result_chars: int = 8000
    max_preview_bytes: int = 8192        # 8KB, same as workspace agent
```

### AgentResult (models.py)

```python
class AgentResult(BaseModel):
    answer: str                           # Natural language summary
    status: str                           # "completed" | "max_iterations" | "error"
    sources: list[str]                    # Service names analyzed (e.g. ["GenomeAssembly2"])
    tool_trace: list[ToolExecution]       # Full execution history
    iterations_used: int
    elapsed_seconds: float

    # Structured output for UI (like workspace agent)
    output_files: list[dict]              # [{path, name, type, size, service}]
    metrics: list[dict]                   # [{service, metric_name, value, unit}]
    previews: list[dict]                  # [{path, data, bytes_read, total_size}]
    report_links: list[dict]             # [{path, label, service}] — HTML reports to link
    step_summaries: list[dict]            # [{step_name, app_name, status, summary, metrics}]
```

The MCP server's `agent_chat_tool.py` will map this to the standardized response dict:

```python
{
    "answer": result.answer,
    "status": result.status,
    "sources": result.sources,
    "elapsed_seconds": result.elapsed_seconds,
    "tool_trace": [...],
    # Analysis-specific fields
    "output_files": result.output_files,
    "metrics": result.metrics,
    "previews": result.previews,
    "report_links": result.report_links,
    "step_summaries": result.step_summaries,
}
```

---

## Tools

The analysis agent gets **4 tools**, three reused from the workspace agent and one new:

### 1. `workspace_browse` (reused)

Imported from `workspace_agent.tools.browse`. Browse/search workspace directories with filters.

**Why**: The agent needs to navigate output directories, especially the hidden `.{output_file}/` directories where BV-BRC stores results.

### 2. `get_file_metadata` (reused)

Imported from `workspace_agent.tools.browse`. Get detailed metadata for a file.

**Why**: Needed to check file sizes, types, and existence of expected output files.

### 3. `read_file_preview` (reused)

Imported from `workspace_agent.tools.read`. Read first N bytes of a file (default 8KB).

**Why**: The core analysis capability. Reads TSV reports, text summaries, JSON outputs, and extracts metrics from them.

### 4. `get_expected_outputs` (new)

A local tool (no API call) that looks up the expected output file patterns for a given service type using the embedded `service_outputs.json` knowledge.

```python
def get_expected_outputs(service_name: str) -> dict:
    """
    Returns the expected output file patterns for a BV-BRC service.

    Args:
        service_name: The BV-BRC service app name (e.g. "GenomeAssembly2")

    Returns:
        Dict mapping output_key -> file path template
        e.g. {"contigs_fasta": "${output_path}/.${output_file}/${output_file}_contigs.fasta",
              "assembly_report": "..."}
    """
```

**Why**: Gives the LLM explicit knowledge of which files to look for per service, avoiding blind browsing. The LLM can resolve the path templates using the `output_path` and `output_file` from workflow context, then directly read the important files.

### Tool import strategy

The three workspace tools are imported directly from the workspace agent's tool modules. The analysis agent's `tool_registry.py` imports the functions and re-registers them with the same schemas:

```python
from workspace_agent.tools.browse import workspace_browse, get_file_metadata
from workspace_agent.tools.read import read_file_preview
```

This ensures both agents share the same implementation. If the workspace agent's tools change, the analysis agent inherits the changes.

---

## Service-Specific Output Knowledge

The system prompt includes embedded knowledge about what each service produces and what metrics to extract. This is stored in `output_knowledge.py` and injected into the prompt.

### Supported services and their key metrics

| Service | Key Output Files | Metrics to Extract |
|---|---|---|
| **GenomeAssembly2** | `*_contigs.fasta`, `*_AssemblyReport.html` | N50, total contigs, total length, largest contig, GC% |
| **GenomeAnnotation** | `*.genome`, `GenomeReport.html` | CDS count, gene count, tRNA count, rRNA count, genome size |
| **ComprehensiveGenomeAnalysis** | `FullGenomeReport.html`, `annotated.genome` | Combined assembly + annotation stats |
| **Homology (BLAST)** | `blast_out.txt`, `blast_out.json` | Number of hits, top hit organism, best E-value, percent identity range |
| **CodonTree** | `*_tree.nwk`, `*_report.html` | Number of taxa, number of genes used, tree format |
| **GeneTree** | `*_raxml_rell_tree.nwk`, `*_aligned.fa` | Number of taxa, alignment length, model used |
| **Variation** | `all.var.tsv` | Total variants, SNP count, indel count, variant distribution by type |
| **RNASeq** | `multiqc_report.html` | Read mapping rate, total reads, differential expression gene count |
| **TaxonomicClassification** | `*_multiqc_report.html` | Classification rate, top taxa, read distribution |
| **MetagenomeBinning** | `BinningReport.html` | Number of bins, completeness, contamination |
| **ViralAssembly** | `*.contigs.fasta`, `*.assembly_report.txt` | Contig count, coverage, reference match |
| **FastqUtils** | `*_processed`, `*.qc_report.txt` | Reads before/after, trimmed %, quality scores |

For services not in this list, the agent falls back to dynamic discovery: browse the output directory and characterize what it finds.

---

## System Prompt Design

The system prompt has three sections:

### 1. Role and behavior

```
You are the BV-BRC Analysis Agent. Your job is to examine the output files
from completed bioinformatics service jobs and provide the user with a clear,
informative summary of what was produced.

You should:
- Identify what output files were created
- Read key output files to extract important metrics and statistics
- Summarize findings in clear, accessible language
- Highlight any notable results or potential issues
- Provide links/paths to HTML reports the user can view

You should NOT:
- Modify any files
- Re-run or submit any jobs
- Make definitive biological conclusions (present data, let the user interpret)
```

### 2. Service output knowledge

Injected from `output_knowledge.py`. Tells the LLM which files to look for per service and which metrics to extract from each.

### 3. Workflow context handling

```
When workflow_context is provided, use it to:
1. Call get_expected_outputs for each step's app_name
2. Resolve output file paths using the step's output_path and output_file params
3. Browse the output directory to confirm files exist
4. Read key files (TSV, text reports, FASTA headers) to extract metrics
5. Produce a per-step summary, then an overall workflow summary

For multi-step workflows, present results in execution order, noting how
outputs from earlier steps fed into later steps.
```

---

## Multi-Step Workflow Handling

For workflows with multiple steps (e.g., Assembly → Annotation):

1. The agent receives all steps in `workflow_context.steps`
2. It analyzes **every step** with `status: "succeeded"`
3. It presents results in topological/execution order
4. For each step, it provides:
   - Service name and what it does (one sentence)
   - Output files found
   - Key metrics extracted
   - Links to HTML reports
5. An overall summary at the end ties the steps together

Example output for an Assembly → Annotation workflow:

```
## Workflow: genome-analysis-ecoli

### Step 1: Genome Assembly (GenomeAssembly2)
Assembled raw reads into contiguous sequences.

**Assembly Statistics:**
- Total contigs: 47
- N50: 234,891 bp
- Total length: 4,832,105 bp
- Largest contig: 512,340 bp

**Output files:** 3 files in /.assembly_output/
📄 Report: /user/home/CopilotWorkflows/.assembly_output/assembly_output_AssemblyReport.html

### Step 2: Genome Annotation (GenomeAnnotation)
Annotated the assembled genome, identifying genes and functional elements.

**Annotation Statistics:**
- CDS: 4,412
- tRNA: 86
- rRNA: 22
- Hypothetical proteins: 1,203 (27%)

**Output files:** 5 files in /.annotation_output/
📄 Report: /user/home/CopilotWorkflows/.annotation_output/GenomeReport.html

### Summary
Successfully assembled and annotated an E. coli-sized genome (4.8 Mbp)
with 4,412 predicted coding sequences.
```

---

## On-Demand Usage

When a user asks to analyze outputs (not triggered by webhook), the orchestrator routes to the analysis agent. The agent receives a natural language query without `workflow_context`.

In this case, the agent must:

1. Parse the user's intent (which job/service/output to analyze)
2. Use conversation context (recent messages may reference a workflow_id or output path)
3. Browse the workspace to find the relevant output directory
4. Proceed with the same analysis loop

The routing LLM should classify requests like these to the analysis agent:

- "Analyze my assembly results"
- "What did my BLAST job find?"
- "Summarize the outputs from workflow wf_abc123"
- "Look at the results in /user/home/CopilotWorkflows/assembly_output"

---

## Registration

### agents.yaml entry

```yaml
analysis:
  name: "Analysis Agent"
  description: >
    Analyzes output files from completed BV-BRC service jobs. Browses
    job output directories, reads key result files (reports, tables,
    sequences), and extracts service-specific metrics (assembly N50,
    gene counts, BLAST hits, variant counts, etc.). Returns both a
    natural language summary and structured data (file listings,
    metrics, report links) for rich UI rendering.
  endpoint: "http://140.221.78.15:8153"
  protocol: "mcp"
  capabilities:
    - output_analysis
    - metric_extraction
    - result_summarization
    - workspace_browsing
  max_iterations: 6
  timeout_seconds: 120
  auth_token_env: "BV_BRC_AUTH_TOKEN"
  chat_tool: "agent_chat"
  mcp_server_name: "bvbrc_server"
  chat_tool_params:
    agent_type: "analysis"
```

### MCP server dispatch (agent_chat_tool.py)

```python
elif agent_type == "analysis":
    return await _run_analysis_agent(query, config_kwargs, ctx, progress_callback)
```

With `_run_analysis_agent()` following the same pattern as the other four `_run_*_agent()` helpers.

### Router prompt update

Add the analysis agent to the routing LLM's agent catalog in `orchestrator/orchestrator/router/prompts.py`:

```
- analysis: Analyzes output files from completed BV-BRC service jobs.
  Routes here when the user asks about job results, output analysis,
  or service output interpretation. Also receives workflow completion
  triggers. Do NOT route here for running/submitting jobs (use service)
  or for general file browsing (use workspace).
```

---

## SSE Event Flow (Webhook → Orchestrator → Frontend)

The gateway always forwards workflow completion to the orchestrator, regardless
of status. The orchestrator's routing logic decides what to do:

```
1. Workflow Engine fires POST /workflow-complete to Gateway
2. Gateway receives webhook:
   - Constructs a request with workflow_context (steps, output_paths, status)
   - POST /orchestrate/stream (no target_agent — let the orchestrator route)
   - The query describes what happened: "Workflow X completed with status Y"
3. Orchestrator routes based on status:
   a. status == "succeeded" → routes to analysis agent
      - Analysis agent browses outputs, extracts metrics, returns summary
   b. status == "failed" / "cancelled" → orchestrator decides
      - May produce a direct response (simple failure message)
      - May route to helpdesk for troubleshooting guidance
      - The analysis agent is NOT invoked for failures
4. Gateway streams SSE events to frontend and persists the message
```

This keeps the routing decision in the orchestrator where it belongs, rather
than hardcoding status-based branching in the gateway.

### New SSE data in `synthesis_complete` / `final_response`

The `result_for_ui` object in the synthesis event will include:

```json
{
  "agent_used": "analysis",
  "output_files": [...],
  "metrics": [...],
  "report_links": [...],
  "step_summaries": [...],
  "previews": [...]
}
```

The frontend can use this to render rich output cards, metric tables, and report links alongside the natural language summary.

---

## Implementation Checklist

### Agent A (Backend Core) — this repo

- [ ] Create `bvbrc-agents/agents/analysis_agent/` package
- [ ] Implement `models.py` (AgentConfig, AgentResult, AgentState)
- [ ] Implement `output_knowledge.py` (service output patterns from service_outputs.json)
- [ ] Implement `tool_registry.py` (reuse 3 workspace tools + new get_expected_outputs)
- [ ] Implement `prompts.py` (system prompt with service knowledge)
- [ ] Implement `agent.py` (run_agent loop)
- [ ] Add `_run_analysis_agent()` to `mcp_server/tools/agent_chat_tool.py`
- [ ] Register in `orchestrator/config/agents.yaml`
- [ ] Update router prompts in `orchestrator/orchestrator/router/prompts.py`
- [ ] Add `workflow_context` to `OrchestratorRequest` model
- [ ] Thread `workflow_context` through `agent_executor.py` into agent context
- [ ] Write unit tests in `orchestrator/tests/`

### Agent C (Gateway) — separate repo

- [ ] Modify `workflow-complete` webhook handler: forward ALL completions to orchestrator instead of writing static message
- [ ] Pass `workflow_context` (steps, output_paths, app names, status) in orchestrator request
- [ ] Handle analysis agent's `result_for_ui` in SSE event mapping

### Agent D (Frontend) — separate repo

- [ ] Handle `result_for_ui` with `agent_used: "analysis"` for rich rendering
- [ ] Render output file listings, metric tables, report links from structured data

---

## Resolved Design Decisions

1. **Timeout**: Keep 120s default. Adjust later if multi-step workflows with many output files consistently hit the limit.

2. **Workflow completion routing**: The gateway sends ALL workflow completions to the orchestrator (not just successes). The orchestrator's routing logic decides what to do — routes to analysis agent for successes, may use direct response or helpdesk for failures. This keeps routing decisions in the orchestrator where they belong.

3. **HTML report parsing**: Start with raw 8KB preview. GPT-4.1 handles HTML fragments well enough, and BV-BRC reports typically front-load summary statistics. If the LLM consistently misses metrics buried in large HTML files, add a lightweight `strip_html` tool in a future iteration.

4. **Caching**: Deferred. Always re-read files for now. Optimize with conversation-context-based caching later if re-analysis latency becomes a concern.
