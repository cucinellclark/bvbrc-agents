# Self-Evolving Agents: Implementation Plan

## Overview

A self-evolution system for the BV-BRC Copilot that improves **both prompts and code** across the orchestrator, data agent, service agent, and workspace agent. Uses LLM-as-judge scoring, experience accumulation, and human-in-the-loop approval.

All model choices are configurable via `evolution_config.yaml`.

---

## Architecture

```
                         ┌──────────────────────────────────┐
                         │     evolution_config.yaml         │
                         │  (models, targets, scoring, etc.) │
                         └──────────┬───────────────────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         │                          │                          │
    ┌────▼─────┐          ┌────────▼────────┐         ┌───────▼────────┐
    │  Trace   │          │   LLM Judge     │         │   Evolvers     │
    │ Collector│─traces──▶│  (score each    │─scored─▶│  (prompt +     │
    │          │          │   interaction)  │ traces  │   code)        │
    └────┬─────┘          └────────┬────────┘         └───────┬────────┘
         │                         │                          │
         │                ┌────────▼────────┐         ┌───────▼────────┐
         │                │ Experience Bank  │         │   Proposals    │
         │                │ (successes,      │         │  (pending/     │
         │                │  failures, by    │         │   approved/    │
         │                │  category)       │         │   rejected)    │
         │                └─────────────────┘         └───────┬────────┘
         │                                                    │
         │                                            ┌───────▼────────┐
         │                                            │  Human Review  │
         │                                            │  (approve/     │
         │                                            │   reject)      │
         │                                            └───────┬────────┘
         │                                                    │
         │                                            ┌───────▼────────┐
    Orchestrator                                      │  Apply + Re-   │
    Event Stream ◄────────────────────────────────────│  benchmark     │
    (production)                                      └────────────────┘
```

---

## Phases

### Phase 1: Trace Collection

**Goal:** Capture structured interaction traces without modifying agent behavior.

**What to build:**
- A `TraceCollector` class that wraps `orchestrate()` as a decorator or middleware
- Persists each interaction as a JSON file under `self_evolving_agents/traces/`
- Hooks into `orchestrate_to_response()` at line 290 of `orchestrate.py` -- the `execution_trace` is already built there but only held in memory

**Key integration point:** `orchestrator/orchestrator/orchestrate.py`
- The `orchestrate_to_response()` function already collects every event into `execution_trace` (lines 290-299)
- Option A: Add a post-hook that persists `execution_trace` after the response is built
- Option B: Add a middleware layer in `server.py` that wraps the `/orchestrate` endpoint

**What each trace should contain:**
```json
{
  "trace_id": "uuid",
  "timestamp": "ISO8601",
  "query": "user's question",
  "conversation_context": "...",
  "routing_decision": {"decision": "agent", "agent_key": "data", ...},
  "agent_results": [...],
  "tool_calls": [...],
  "response_text": "final answer",
  "elapsed_ms": 1234,
  "events": [...]
}
```

**Considerations:**
- Trace files can get large if you capture full event streams. Consider a compact mode that omits `AGENT_PROGRESS` events.
- Auth tokens flow through the request. **Strip `auth_token` from traces** before persisting.
- The benchmark runner (`bruce_questions/run_benchmark.py`, located outside the bvbrc-agents repo) already saves full traces to JSON. You can retroactively judge existing benchmark results without building the collector first. The benchmark path is configurable via `evolution_config.yaml` or the `EVOLUTION_BENCHMARK_PATH` env var.

---

### Phase 2: LLM Judge

**Goal:** Score each trace on multiple quality dimensions using an LLM.

**What to build:**
- A `Judge` class that takes a trace dict and returns dimension scores + explanations
- Uses the `judge` model from `evolution_config.yaml`
- Outputs a `JudgmentResult` with per-dimension scores and an overall weighted score

**Key design decisions:**

1. **Judge prompt construction.** The judge needs to see:
   - The user's question
   - What routing decision was made
   - What tool calls the agent made (with arguments)
   - The final response
   - It does NOT need the full event stream -- summarize it

2. **Calibration.** The judge's scores are only useful if they're consistent. Consider:
   - Including 2-3 reference-scored examples in the judge prompt (few-shot calibration)
   - Running the judge on the same trace twice and checking agreement
   - Starting with a manual scoring pass on 20-30 traces to establish ground truth

3. **Cost.** Every trace gets judged = 1 LLM call per interaction. With `gpt41mini` this is cheap, but monitor it.

**Considerations:**
- The judge model is different from the agent model. If the agent uses `gpt41` and the judge uses `gpt41mini`, the judge may not understand domain-specific correctness well. You may need the `gpt41` judge for the `answer_correctness` dimension specifically.
- For routing accuracy, you might not need an LLM at all -- a rule-based check ("did data keywords go to data agent?") could be more reliable. The `router_judge` model in the config exists for this purpose but consider rules first.
- The scoring rubric in `evolution_config.yaml` has weights that sum to 1.0. These weights are themselves evolvable -- but manually at first.

---

### Phase 3: Experience Bank

**Goal:** Categorize and store judged traces for evolution context.

**What to build:**
- A `ExperienceBank` class that reads/writes JSONL files organized by category
- Methods: `add_success()`, `add_failure()`, `get_failures(category, limit)`, `get_successes(category, limit)`
- Auto-categorization based on which agent handled the query

**Structure:**
```
self_evolving_agents/experience_bank/
  routing/
    successes.jsonl
    failures.jsonl
  data_agent/
    successes.jsonl
    failures.jsonl
  service_agent/
    successes.jsonl
    failures.jsonl
  workspace_agent/
    successes.jsonl
    failures.jsonl
  synthesis/
    successes.jsonl
    failures.jsonl
```

**Considerations:**
- JSONL (one JSON object per line) is better than a single JSON array for append-only workloads. No need to parse the whole file to add an entry.
- Implement rotation early. Without it, failure files will grow unbounded. The config has `max_entries_per_category: 500` and `retention_days: 90`.
- The experience bank is a training data store, not a database. Don't over-engineer it. A filesystem + JSONL is fine at your scale.
- Consider indexing by failure pattern (e.g., "zero results", "wrong agent", "timeout") so the evolver can focus on specific failure modes rather than bulk analysis.

---

### Phase 4: Prompt Evolver

> **Implementation note (decided 2026-05-22):** Phases 4 and 5 will use
> [GEPA](https://github.com/gepa-ai/gepa) (`pip install gepa`) instead of
> a hand-rolled evolutionary loop. GEPA's `optimize_anything` API provides
> reflective mutation, Pareto-efficient candidate selection, and merge
> operations out of the box -- replacing what would otherwise be a custom
> `PromptEvolver` / `CodeEvolver` class.
>
> The Phase 2 LLM Judge becomes GEPA's **evaluator function**: given a
> candidate prompt/code, inject it into the agent, run benchmark traces,
> score them with the judge, and return `(score, side_info_dict)`. GEPA
> handles the reflect-mutate-select loop.
>
> **Why GEPA over custom code:**
> - Pareto-aware selection across per-question scores (avoids regressions)
> - Reflective mutation reads full execution traces ("Actionable Side
>   Information") to diagnose *why* a candidate failed, not just *that* it
>   failed
> - Proven at scale (Databricks, Shopify, OpenAI Cookbook's self-evolving
>   agents reference implementation)
> - 100-500 evaluations to converge vs. thousands for RL
>
> **Cost consideration:** Each GEPA evaluation = one full benchmark run
> (~20 questions x ~120s). Budget ~150 evaluations for a prompt target.
> This is an offline batch process, not real-time.
>
> **What Phases 1-3 must provide for GEPA compatibility:**
> - Trace format that can be summarized into ASI (Actionable Side
>   Information) for the reflection LLM
> - Judge scores as a float in [0, 1] with per-dimension breakdowns
> - Experience bank organized so GEPA's evaluator can pull failures for
>   context
>
> The interfaces built in Phases 1-2 (trace collector, judge, experience
> bank) are designed to plug directly into a GEPA evaluator wrapper.

**Goal:** Analyze failure patterns and propose targeted prompt edits.

**What to build:**
- A `PromptEvolver` class (or GEPA evaluator wrapper) that:
  1. Reads recent failures from the experience bank
  2. Reads the current prompt file
  3. Asks the `prompt_evolver` model to propose a specific edit
  4. Writes a proposal file to `self_evolving_agents/proposals/pending/`

**Evolution targets** (from `evolution_config.yaml`):

| Target | File | Impact | Notes |
|---|---|---|---|
| `_STRATEGY` | `agents/data_agent/prompts/system.py:143-177` | Highest | Controls query planning logic |
| `_PROBE_STRATEGY` | `agents/data_agent/prompts/system.py:179-203` | High | When/how to do data recon |
| `_EFFICIENCY` | `agents/data_agent/prompts/system.py:205-231` | High | Iteration budget management |
| `ROUTING_SYSTEM_PROMPT` | `orchestrator/.../router/prompts.py:11-87` | High | Agent selection logic |
| `build_phase1_prompt` | `agents/.../prompts/phase1.py:10-113` | High | Workflow decomposition |
| `SYNTHESIS_SYSTEM_PROMPT` | `orchestrator/.../synthesizer/prompts.py:13-38` | Medium | Final answer formatting |

**Key design decisions:**

1. **Scope of edits.** The evolver should propose MINIMAL, TARGETED edits -- not rewrite entire prompts. A diff format is ideal:
   ```yaml
   target: data_agent_system_prompt
   section: _STRATEGY
   edit_type: append   # or: replace, insert_after, delete
   context_lines: "5. REFINE ITERATIVELY:..."
   proposed_change: |
     Add after "NEVER report 0 results after trying only one query formulation":
     "   e. CHECK ALTERNATE COLLECTIONS: If the primary collection has no data,
          check related collections (e.g., genome_amr instead of genome for
          resistance data)."
   ```

2. **Prompt versioning.** Before the evolver modifies any file, snapshot the current version. Options:
   - Git commit before each change (if `git_commit_on_apply: true`)
   - Copy to `proposals/history/<target>_v<N>.py`
   - Inline version comment in the prompt file

3. **One proposal per target at a time.** Don't let the evolver propose 5 changes to `_STRATEGY` simultaneously -- they could conflict.

**Considerations:**
- Prompt evolution can cause regressions. A change that fixes one failure pattern can break a working pattern. This is why `require_benchmark_improvement: true` exists in the config.
- The `_QUERY_SYNTAX` section is mostly factual Solr documentation. Evolving it is risky -- a wrong syntax example could cause widespread failures. Mark it `priority_override: low` (already done in config).
- The `COLLECTION_REFERENCE` (at `agents/data_agent/prompts/collection_reference.py`) is auto-generated from `data_types.xlsx`. The evolver should NEVER touch it directly -- instead, if it identifies field naming issues, it should propose changes to the generator or the Excel source.
- Prompt changes may interact with model choice. A prompt optimized for `gpt41` may not work as well if you switch to a different model. Track which model was active when each proposal was generated.

---

### Phase 5: Code Evolver

> **Implementation note:** Like Phase 4, this will use GEPA's
> `optimize_anything` API. The seed candidate is the full source file
> (or a targeted section), and the evaluator runs the test suite +
> benchmark. See the GEPA note in Phase 4 for details.

**Goal:** Propose code modifications to agent logic, routing, tool handling, etc.

**What to build:**
- A `CodeEvolver` class (or GEPA evaluator wrapper) similar to PromptEvolver but:
  - Reads the full source file (not just a section)
  - Proposes a unified diff
  - Includes a test plan (what to verify the change doesn't break)

**Code targets** (from `evolution_config.yaml`):

| Target | File | What It Controls |
|---|---|---|
| `data_agent_loop` | `agents/data_agent/agent.py` | Iteration logic, duplicate detection, forced synthesis |
| `router_logic` | `orchestrator/.../router/router.py` | JSON parsing, fallback keywords, fuzzy matching |
| `service_decompose` | `agents/.../phases/decompose.py` | Anti-loop, tool fingerprinting, plan extraction |
| `service_build` | `agents/.../phases/build.py` | Per-step parameter resolution, validation |
| `data_tool_registry` | `agents/data_agent/tool_registry.py` | Tool schemas (search_data, facet_query, etc.) |
| `mcp_agent_chat_tool` | `mcp_server/tools/agent_chat_tool.py` | Unified agent dispatcher (separate repo) |
| `mcp_data_tools` | `mcp_server/tools/data_tools.py` | MCP tool definitions for data agent (separate repo) |
| `mcp_service_tools` | `mcp_server/tools/service_tools.py` | MCP tool definitions for service agent (separate repo) |
| `mcp_workspace_tools` | `mcp_server/tools/workspace_tools.py` | MCP tool definitions for workspace agent (separate repo) |
| `mcp_solr_functions` | `mcp_server/functions/` | Solr query, service plan, workspace functions (separate repo) |
| `mcp_common` | `mcp_server/common/` | Auth, config, LLM client utilities (separate repo) |

**Key design decisions:**

1. **Code changes are higher risk than prompt changes.** Consider:
   - Always require manual approval for code changes (even in `auto_prompt` mode)
   - Run the full test suite (not just benchmark) after code changes
   - Limit code diffs to < 50 lines changed per proposal

2. **What kind of code changes make sense for LLM evolution?**
   - Adding/improving fallback logic (e.g., better keyword lists in `_fallback_routing`)
   - Tuning parameters (duplicate detection thresholds, max iterations, timeout values)
   - Adding new error handling branches
   - Improving tool result truncation/formatting
   - **NOT:** Architectural changes, dependency changes, or new tool implementations

3. **The evolver needs the full file context** to propose valid code. Unlike prompts (which are self-contained strings), code has imports, type dependencies, and calling conventions. The `code_evolver` model needs `max_tokens: 8192` minimum.

**Considerations:**
- Code evolution is where things can go wrong fast. A prompt change that says "try harder" is low risk. A code change that modifies the tool execution loop can break the entire agent.
- The orchestrator tests (`orchestrator/tests/`, 117 tests) are your safety net for code changes to orchestrator files. Run them with `cd orchestrator && source orchestrator_env/bin/activate && pytest`. Data agent has integration tests only. You may need to write unit tests for code targets that don't have them.
- Tool schemas (`TOOL_SCHEMAS` in `agents/data_agent/tool_registry.py`) are a hybrid -- they're code but they contain LLM-facing descriptions. Evolving the `description` field of a tool schema is really prompt evolution wearing a code disguise. Consider treating tool schema descriptions as prompt targets.
- The `frozen: true` flag on `orchestrate_pipeline` (`orchestrator/orchestrator/orchestrate.py`) is important. The orchestration loop is pure plumbing -- there's almost nothing in there that an LLM evolver should touch. The risk/reward ratio is terrible.
- The MCP server (`mcp_server/`) lives in a separate repo (`bvbrc-mcp-server`). It IS eligible for evolution, but proposals that target MCP server files are committed to that repo, not `bvbrc-agents`. MCP targets use `repo: "mcp_server"` in the config to distinguish them.

---

### Phase 6: Proposal Manager + Human Review

**Goal:** Stage proposals for review and track approval/rejection.

**What to build:**
- Proposal file format (YAML):
  ```yaml
  id: "prop_20260521_001"
  target: "data_agent_system_prompt"
  section: "_STRATEGY"
  type: "prompt"   # or "code"
  status: "pending"  # pending, approved, rejected, applied, reverted
  created: "2026-05-21T14:00:00Z"
  model_used: "gpt41"
  evidence:
    failure_count: 15
    success_count: 8
    failure_patterns:
      - "Agent tries search_data before probe_data on unfamiliar organisms"
      - "Agent exceeds iteration budget on multi-collection queries"
  current_content: |
    ... (snapshot of current text/code)
  proposed_content: |
    ... (proposed replacement)
  diff: |
    ... (unified diff)
  benchmark_before: null  # Filled after validation
  benchmark_after: null
  reviewer_notes: ""
  ```
- A CLI or simple script for: `list`, `show <id>`, `approve <id>`, `reject <id> --reason "..."`, `apply <id>`, `revert <id>`

**Considerations:**
- Keep the proposal format flat and readable. You'll be reviewing these manually, so clarity matters more than schema elegance.
- Track the model that was used when the proposal was generated. If you switch models later, old proposals may be less relevant.
- `revert` needs to work. Store enough information in the proposal to undo the change (i.e., the full `current_content` before the change was applied).
- Consider a simple approval flow:
  1. `pending` -> `approved` (human reviews and approves)
  2. `approved` -> Run benchmark in `--dry-run` mode with the proposed change
  3. If benchmark improves -> `applied` (write the change to the file)
  4. If benchmark regresses -> `rejected` with benchmark results as reason
  5. `applied` -> `reverted` (if you discover issues later)

---

### Phase 7: Evolution Runner (CLI)

**Goal:** A single CLI entry point that orchestrates the full evolution loop.

**Commands:**
```bash
# Run the judge on recent traces (or existing benchmark results)
python -m self_evolving_agents judge --traces ./traces/ --limit 50

# Analyze experience bank and generate evolution proposals
python -m self_evolving_agents evolve --target data_agent_system_prompt

# Evolve code targets
python -m self_evolving_agents evolve --target router_logic --type code

# List/review proposals
python -m self_evolving_agents proposals list
python -m self_evolving_agents proposals show prop_20260521_001
python -m self_evolving_agents proposals approve prop_20260521_001

# Apply an approved proposal and re-benchmark
python -m self_evolving_agents apply prop_20260521_001

# Run the full loop: judge -> analyze -> evolve -> propose
python -m self_evolving_agents run --full
```

---

## Things to Consider

### 1. The Argo Gateway is Your Bottleneck

Every evolution component calls the Argo API. The judge alone adds 1 LLM call per interaction. If you're judging 100 traces + running the evolver + re-benchmarking, that's potentially hundreds of API calls. Plan for:
- Rate limiting / backoff in the LLM client
- Batching traces for judgment (send 5 traces in one prompt instead of 5 separate calls)
- Running evolution offline / during low-traffic periods
- Cost tracking per evolution run

### 2. Model-Specific Parameter Quirks

The shared `config/llm_config.py` (at the bvbrc-agents repo root) has `MODEL_PARAM_EXCLUSIONS` and `MODEL_USE_MAX_COMPLETION_TOKENS` with special handling for gpt5, o3, o4-mini, gpt41. The evolution system's LLM client needs to respect these same rules. Options:
- Import and reuse `llm_config.py` (via the same `sys.path.insert` pattern)
- Or copy the exclusion logic into the evolution system's own client
- The first option is better -- single source of truth

### 3. Prompt vs Code: The Blurry Line

Some things that look like code are really prompts:
- Tool schema `description` fields in `agents/data_agent/tool_registry.py` -- these are LLM-facing text
- Agent `description` fields in `orchestrator/config/agents.yaml` -- the router LLM reads these
- Error messages injected into conversation history (e.g., the "DUPLICATE CALL DETECTED" message in `agents/data_agent/agent.py:385-390`)
- The `_build_simulated_result()` note strings in `agents/data_agent/agent.py:124-193`

Treat these as prompt targets even though they live in code files. The config already supports this -- a `type: code` target can have prompt-like evolution if the evolver is instructed to focus on string constants.

### 4. The Evaluation Chicken-and-Egg Problem

To judge quality, you need traces. To get traces, you need the system running with real user interactions. To justify the evolution effort, you need evidence of improvement. Bootstrap the cycle:
1. **Start with benchmark results.** The benchmark runner (`bruce_questions/run_benchmark.py`, external to this repo) already captures full traces. Judge those first. Configure the benchmark path in `evolution_config.yaml` or via `EVOLUTION_BENCHMARK_PATH`.
2. **Manually score 20-30 traces** to calibrate the LLM judge and establish ground truth.
3. **Run the benchmark twice** with the same questions to establish variance. If the same question gets different scores on two runs, your benchmark has a noise problem.

### 5. Evolution Can Cause Cascading Regressions

A change to the routing prompt might fix misrouted service queries but break data queries that were working. Mitigations:
- **Always re-benchmark the full question set**, not just the questions that were failing
- **Track per-question scores**, not just overall averages. A proposal that improves average score by 5% but regresses 3 previously-working questions is suspicious.
- **Keep the change scope small.** One section of one prompt per proposal. Never evolve multiple targets simultaneously.
- **Maintain a "golden set"** of 10-15 questions that must always pass. Any proposal that regresses a golden question is auto-rejected.

### 6. Code Evolution Safety

Code changes need stronger guardrails than prompt changes:
- **Run the test suite** (`cd orchestrator && source orchestrator_env/bin/activate && pytest -v`) before and after every code change
- **Never auto-approve code changes.** Even in `auto_prompt` mode, code stays manual.
- **Limit the diff size.** If the evolver wants to change > 50 lines, reject and ask it to propose a smaller change.
- **Don't let the evolver add new dependencies.** If the proposed code imports a new library, reject it.
- **Don't let the evolver change function signatures.** Internal refactors are fine, but changing the API contract of `run_agent()`, `orchestrate()`, or `route()` would break callers.

### 7. The Experience Bank Needs Curation

Raw failure traces contain noise. Not every low-scoring interaction represents a fixable problem:
- User asked an off-topic question (not a BV-BRC question at all)
- BV-BRC API was down or slow (infrastructure, not agent)
- User's question was genuinely ambiguous
- The LLM judge scored incorrectly (judge miscalibration)

Consider adding a `classification` field to experience entries: `fixable`, `infrastructure`, `ambiguous`, `out_of_scope`, `judge_error`. This can be done manually at first, automated later.

### 8. Benchmark Coverage Gaps

The benchmark suite (`bruce_questions/`, external to this repo) has question sets, but they may not cover all scenarios the evolution system targets. Gaps to watch for:
- **Pipeline questions** (multi-agent): Do any benchmark questions trigger pipeline routing? If not, you can't benchmark routing improvements for pipelines.
- **Workspace agent**: Are there workspace-specific questions? If not, workspace evolution has no validation signal.
- **Error recovery**: Do any questions intentionally test failure modes (bad organism names, impossible queries)?
- **Service agent**: Are there questions that test the full 3-phase decompose-build-compose workflow?

You may need to expand the benchmark question set as you identify coverage gaps. Consider generating synthetic questions from the experience bank's failure patterns.

### 9. Multi-Model Consistency

If you evolve a prompt while using `gpt41` and then switch to a different model (e.g., Llama-4 on the local endpoint), the evolved prompt may not work as well. Each model has different instruction-following characteristics. Options:
- Track which model was used when each proposal was generated/validated
- Re-validate proposals when switching models
- Consider model-specific prompt variants (though this adds complexity)

### 10. When to Stop Evolving

Evolution should converge, not run forever. Signs that a target is "done":
- Benchmark score plateaus (< 1% improvement over 3 consecutive proposals)
- The experience bank has < 5 new failures in the last 30 days for that category
- Proposals start getting rejected because the evolver can't find meaningful improvements

At that point, mark the target `frozen: true` in the config and focus evolution effort elsewhere.

---

## File Structure

The `self_evolving_agents/` directory lives inside the `bvbrc-agents` repo.
All target paths in `evolution_config.yaml` are relative to the `bvbrc-agents/` root.

```
bvbrc-agents/                          # Repo: git@github.com:cucinellclark/bvbrc-agents.git
├── agents/
│   ├── data_agent/                    # Data retrieval agent
│   ├── service_agent/                 # Service agent (3-phase workflow)
│   └── workspace_agent/               # Workspace browsing agent
├── orchestrator/
│   ├── orchestrator/                  # Orchestrator Python package
│   ├── config/agents.yaml             # Agent registry
│   └── tests/                         # pytest suite (117 tests)
├── config/                            # Shared LLM config (llm.yaml + llm_config.py)
├── mcp_server/                        # GITIGNORED — cloned from bvbrc-mcp-server repo
├── setup.sh                           # Clones MCP server, installs deps
│
└── self_evolving_agents/              # ← This plan lives here
    ├── PLAN.md                        # This file
    ├── evolution_config.yaml          # All configuration
    ├── venv/                          # Python 3.11 venv
    │
    ├── evolve/                        # Core Python package
    │   ├── __init__.py
    │   ├── __main__.py                # CLI entry point
    │   ├── config.py                  # Load evolution_config.yaml + env overrides
    │   ├── llm_client.py              # LLM client (reuses config/llm_config.py patterns)
    │   ├── trace_collector.py         # Trace persistence
    │   ├── judge.py                   # LLM-as-judge scoring
    │   ├── experience_bank.py         # Categorized trace storage
    │   ├── prompt_evolver.py          # Prompt evolution proposals
    │   ├── code_evolver.py            # Code evolution proposals
    │   ├── proposal_manager.py        # Proposal CRUD + approval workflow
    │   └── benchmark_runner.py        # Wrapper around external bruce_questions/run_benchmark.py
    │
    ├── traces/                        # Persisted interaction traces
    │   └── 2026-05-21/
    │       ├── trace_abc123.json
    │       └── ...
    │
    ├── experience_bank/               # Categorized successes/failures
    │   ├── routing/
    │   │   ├── successes.jsonl
    │   │   └── failures.jsonl
    │   ├── data_agent/
    │   │   ├── successes.jsonl
    │   │   └── failures.jsonl
    │   └── ...
    │
    └── proposals/                     # Evolution proposals
        ├── pending/
        │   └── prop_20260521_001.yaml
        ├── approved/
        ├── rejected/
        └── applied/
```

**External dependencies (not in this repo):**
- `bruce_questions/` — Benchmark suite. Path configurable via `evolution_config.yaml` or `EVOLUTION_BENCHMARK_PATH` env var.
- `bvbrc-mcp-server` — MCP server repo (cloned into `mcp_server/` by `setup.sh`). Eligible for evolution, but changes are committed to the `bvbrc-mcp-server` repo separately. Targets use `repo: "mcp_server"` in the config.

---

## Quick Wins (No Infrastructure Required)

Before building any of this, you can get value from three things you can do today:

### 1. Judge Existing Benchmark Results

You already have benchmark JSON files in the `bruce_questions/results/` directory (external to this repo). Write a one-off script that feeds each result through an LLM judge and outputs a scoring report. No trace collector needed -- the data is already there.

### 2. Add Few-Shot Examples to Agent Prompts

Take 3-5 of your best benchmark results (where the agent gave a correct answer efficiently) and manually add them as examples in `_STRATEGY` or `_PROBE_STRATEGY` (in `agents/data_agent/prompts/system.py`). This is the "Offline Experience Compilation" pattern from the self-evolving agents survey and it requires zero infrastructure.

### 3. Expand Fallback Keywords

The keyword lists in `orchestrator/orchestrator/router/router.py:_fallback_routing()` (lines ~222-255) are your safety net when the routing LLM fails. Review recent interactions and add any missing keywords. This is a 10-minute code change with outsized reliability impact.

---

## Dependencies to Install

For the `self_evolving_agents/venv`:

```bash
cd self_evolving_agents
source venv/bin/activate
pip install pyyaml openai pydantic aiohttp aiofiles

# Phase 4-5 only (not needed for Phases 1-3):
pip install gepa litellm
```

- `pyyaml`: Parse `evolution_config.yaml`
- `openai`: LLM client (Argo API is OpenAI-compatible)
- `pydantic`: Data models for traces, judgments, proposals
- `aiohttp`: Async HTTP if needed
- `aiofiles`: Async file I/O for trace persistence
- `gepa`: Reflective evolutionary optimization for prompts/code (Phases 4-5)
- `litellm`: Required by GEPA for multi-provider LLM routing

You do NOT need the orchestrator's dependencies in this venv. The evolution system runs separately and reads trace files / source files from disk. It only shares the LLM config loader.
