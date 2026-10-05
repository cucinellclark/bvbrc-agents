"""A/B the routing call: current behaviour vs thinking-off + structured output."""
import asyncio, os, sys, time
sys.path.insert(0, "."); sys.path.insert(0, "orchestrator")

from orchestrator.config import OrchestratorConfig
from orchestrator.llm.config import LLMConfig
from orchestrator.llm.client import LLMClient
from orchestrator.router.models import build_routing_extra_body
from orchestrator.router.prompts import build_routing_prompt

CATALOG = """- **data**: Query BV-BRC Solr collections for genomes, features, AMR data
- **service**: Discover, populate and submit GoWe workflows (assembly, annotation, BLAST)
- **workspace**: Browse and inspect user workspace files
- **helpdesk**: BV-BRC usage guidance, FAQ, documentation, literature search
- **analysis**: Analyze completed job results
- **planning**: Multi-step plans and batch operations across agents"""

CASES = [
    ("assemble SRR12345678 and submit the job", "service"),
    ("find all Mycobacterium tuberculosis genomes with isoniazid resistance", "data"),
    ("how do I use the genome assembly service?", "helpdesk"),
    ("what files are in my home directory?", "workspace"),
    ("hello", "direct"),
    ("assemble all the read files in my reads folder", "planning"),
    ("analyze the results of my BLAST job", "analysis"),
]

class _StubRegistry:
    """Minimal stand-in: _parse_routing_response validates agent_key against it."""
    agents = {k: object() for k in
              ("data", "service", "workspace", "helpdesk", "analysis", "planning")}

_REGISTRY = _StubRegistry()


def mk(cfg, extra):
    return LLMClient(LLMConfig(
        base_url=cfg.routing_base_url, api_key=cfg.routing_api_key,
        model=cfg.routing_model, temperature=0.0,
        max_tokens=cfg.routing_max_tokens if extra else 16384,
        timeout_seconds=120, extra_body=extra or None))

async def run(label, client):
    print(f"\n===== {label} =====")
    ok = empty = 0
    t0 = time.monotonic()
    for query, expected in CASES:
        sysp, userp = build_routing_prompt(query=query, agent_catalog=CATALOG)
        try:
            raw = await client.complete(prompt=userp, system_prompt=sysp, temperature=0.0)
        except Exception as e:
            print(f"  {query[:42]:<44} ERROR {type(e).__name__}: {e}")
            continue
        if not raw or not raw.strip():
            empty += 1
            print(f"  {query[:42]:<44} EMPTY RESPONSE")
            continue
        from orchestrator.router.router import _parse_routing_response
        try:
            d = _parse_routing_response(raw, query, _REGISTRY)
            got = (d.plan.steps[0].agent_key if (d.decision == "agent" and d.plan and d.plan.steps) else d.decision)
            hit = "ok " if got == expected else "MISS"
            if got == expected: ok += 1
            print(f"  {query[:42]:<44} {hit} got={got!r} want={expected!r} ({len(raw)} chars)")
        except Exception as e:
            print(f"  {query[:42]:<44} PARSE FAIL {type(e).__name__}: {e} raw={raw[:80]!r}")
    print(f"  --> {ok}/{len(CASES)} correct, {empty} empty, {time.monotonic()-t0:.1f}s total")

async def main():
    cfg = OrchestratorConfig.from_yaml("orchestrator/config/agents.yaml")
    await run("BEFORE: thinking on, no schema, max_tokens=16384", mk(cfg, None))
    await run("AFTER:  thinking off + response_format, max_tokens=2048",
              mk(cfg, build_routing_extra_body(True, True)))

asyncio.run(main())
