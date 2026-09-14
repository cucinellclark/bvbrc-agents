#!/usr/bin/env python3
"""
Admission stress-test harness for BV-BRC Copilot.

Sends concurrent POST /copilot-agent requests through the gateway (nginx),
reads SSE responses, and records per-request metrics to JSONL.

Only allowed load URL:
  POST https://dev-8.bv-brc.org/copilot-api/chatbrc/copilot-agent

No direct proxy/orchestrator/mango/GoWe calls.  No job-submit prompts.

Usage:
  python stress_copilot.py \
    --token "$(cat /path/to/token)" \
    --concurrency 8 \
    --requests 24 \
    --model "Qwen/Qwen3.6-35B-A3B"

Auth: pass a BV-BRC bearer token (NOT the server-side workspace token
from config.json).  Do not commit tokens to the repo.
"""

import argparse
import asyncio
import json
import random
import sys
import time
import uuid
from pathlib import Path

import aiohttp

# ---------------------------------------------------------------------------
# Query mix (no job-submit prompts)
# ---------------------------------------------------------------------------

HELPDESK_QUERIES = [
    "How do I upload a FASTA file to BV-BRC?",
    "What file formats does BV-BRC accept for genome annotation?",
    "How do I use the phylogenetic tree viewer?",
    "What is the difference between PATRIC and BV-BRC?",
    "How do I create a genome group?",
    "What workflows are available in BV-BRC?",
    "How do I compare two genomes?",
    "What is comprehensive genome analysis?",
    "How do I search for antibiotic resistance data?",
    "How do I download genome sequences from BV-BRC?",
]

DATA_QUERIES = [
    "How many public E. coli genomes are in BV-BRC?",
    "Search for Staphylococcus aureus genomes with AMR data",
    "How many Salmonella genomes were added in 2024?",
    "Find genomes with genome_status 'Complete'",
    "How many Mycobacterium tuberculosis genomes are public?",
    "Search for Klebsiella pneumoniae genomes",
    "How many genomes have antibiotic resistance phenotype data?",
    "Find public Pseudomonas aeruginosa genomes",
    "Search for viral genomes in the SARS-CoV-2 collection",
    "How many Clostridioides difficile genomes are available?",
]

WORKSPACE_QUERIES = [
    "List files in my home folder",
    "What files are in my workspace?",
    "Browse my home directory",
    "Show me the contents of my workspace",
    "List my workspace folders",
]

def pick_query() -> tuple[str, str]:
    """Return (query, category) based on the 40/40/20 mix."""
    r = random.random()
    if r < 0.4:
        return random.choice(HELPDESK_QUERIES), "helpdesk"
    elif r < 0.8:
        return random.choice(DATA_QUERIES), "data"
    else:
        return random.choice(WORKSPACE_QUERIES), "workspace"

# ---------------------------------------------------------------------------
# SSE reader
# ---------------------------------------------------------------------------

async def read_sse(response: aiohttp.ClientResponse) -> dict:
    """Read SSE events until 'done' or 'error'. Return metrics dict."""
    events = []
    first_event_time = None
    t0 = time.monotonic()
    final_error = None
    event_type = "unknown"

    # Read raw chunks and split into lines ourselves to avoid aiohttp's
    # default line-size limit (which raises "Chunk too big" on large
    # SSE data lines like streaming token payloads).
    buf = ""
    async for chunk in response.content.iter_any():
        buf += chunk.decode("utf-8", errors="replace")
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.rstrip("\r")

            if not line:
                # Empty line = SSE event boundary (we track per-field)
                continue
            elif line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: "):
                data_str = line[6:]
                elapsed = time.monotonic() - t0

                if first_event_time is None:
                    first_event_time = elapsed

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    data = {"raw": data_str[:200]}

                event_record = {
                    "event": event_type,
                    "elapsed_s": round(elapsed, 3),
                }
                if isinstance(data, dict) and data.get("error"):
                    event_record["error"] = data["error"]
                    final_error = data["error"]

                events.append(event_record)

                # Exit on terminal events
                if event_type in ("done", "error"):
                    break
            elif line.startswith(":"):
                # SSE comment (heartbeat / connected)
                pass

        # Check if we hit a terminal event
        if events and events[-1]["event"] in ("done", "error"):
            break

    total_time = time.monotonic() - t0
    event_types = [e["event"] for e in events]

    return {
        "ttfe_s": round(first_event_time, 3) if first_event_time else None,
        "total_s": round(total_time, 3),
        "event_count": len(events),
        "event_types": event_types,
        "had_queued": "queued" in event_types,
        "had_error": "error" in event_types,
        "error": final_error,
    }

# ---------------------------------------------------------------------------
# Single virtual user
# ---------------------------------------------------------------------------

async def run_virtual_user(
    session: aiohttp.ClientSession,
    url: str,
    token: str,
    model: str,
    user_id: int,
    turns: int,
    jsonl_path: Path,
    jsonl_lock: asyncio.Lock,
    think_time: tuple[float, float] = (0, 0),
) -> list[dict]:
    """Run `turns` chat turns for one virtual user."""
    session_id = str(uuid.uuid4())
    results = []

    for turn in range(turns):
        query, category = pick_query()
        t0 = time.monotonic()

        record = {
            "user_id": user_id,
            "session_id": session_id,
            "turn": turn,
            "category": category,
            "query": query,
            "timestamp": time.time(),
        }

        try:
            body = {
                "query": query,
                "session_id": session_id,
                "model": model,
                "stream": True,
                "save_chat": False,  # don't pollute Mongo
            }
            headers = {
                "Authorization": token,
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
            }

            async with session.post(url, json=body, headers=headers, timeout=aiohttp.ClientTimeout(total=660)) as resp:
                record["http_status"] = resp.status

                if resp.status == 429:
                    body_text = await resp.text()
                    record["error"] = body_text[:500]
                    record["is_429"] = True
                    record["total_s"] = round(time.monotonic() - t0, 3)
                elif resp.status >= 500:
                    body_text = await resp.text()
                    record["error"] = body_text[:500]
                    record["is_5xx"] = True
                    record["total_s"] = round(time.monotonic() - t0, 3)
                elif resp.content_type and "text/event-stream" in resp.content_type:
                    sse_metrics = await read_sse(resp)
                    record.update(sse_metrics)
                else:
                    body_text = await resp.text()
                    record["error"] = f"Unexpected content-type: {resp.content_type}"
                    record["body_preview"] = body_text[:500]
                    record["total_s"] = round(time.monotonic() - t0, 3)

        except asyncio.TimeoutError:
            record["error"] = "client_timeout"
            record["total_s"] = round(time.monotonic() - t0, 3)
        except Exception as e:
            record["error"] = str(e)
            record["total_s"] = round(time.monotonic() - t0, 3)

        results.append(record)

        # Write to JSONL immediately
        async with jsonl_lock:
            with open(jsonl_path, "a") as f:
                f.write(json.dumps(record) + "\n")

        # Print summary
        status = record.get("http_status", "???")
        total = record.get("total_s", "?")
        ttfe = record.get("ttfe_s", "?")
        queued = " [QUEUED]" if record.get("had_queued") else ""
        err = f" ERROR: {record.get('error', '')[:60]}" if record.get("error") else ""
        print(f"  user={user_id:3d}  turn={turn}  status={status}  ttfe={ttfe}s  total={total}s{queued}{err}")

        # Think time between turns
        if turn < turns - 1 and think_time[1] > 0:
            await asyncio.sleep(random.uniform(*think_time))

    return results

# ---------------------------------------------------------------------------
# Burst launcher
# ---------------------------------------------------------------------------

async def run_burst(
    url: str,
    token: str,
    model: str,
    concurrency: int,
    turns_per_user: int,
    jsonl_path: Path,
    think_time: tuple[float, float] = (0, 0),
) -> list[dict]:
    """Launch `concurrency` virtual users simultaneously."""
    jsonl_lock = asyncio.Lock()

    connector = aiohttp.TCPConnector(limit=concurrency + 5, limit_per_host=concurrency + 5)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            run_virtual_user(
                session=session,
                url=url,
                token=token,
                model=model,
                user_id=i,
                turns=turns_per_user,
                jsonl_path=jsonl_path,
                jsonl_lock=jsonl_lock,
                think_time=think_time,
            )
            for i in range(concurrency)
        ]
        all_results = await asyncio.gather(*tasks)

    flat = [r for user_results in all_results for r in user_results]
    return flat

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]):
    """Print a summary table of the burst results."""
    total = len(results)
    ok = sum(1 for r in results if r.get("http_status") == 200 and not r.get("error"))
    queued = sum(1 for r in results if r.get("had_queued"))
    errs_429 = sum(1 for r in results if r.get("is_429"))
    errs_5xx = sum(1 for r in results if r.get("is_5xx"))
    errs_other = sum(1 for r in results if r.get("error") and not r.get("is_429") and not r.get("is_5xx"))
    timeouts = sum(1 for r in results if r.get("error") == "client_timeout")

    ttfe_vals = [r["ttfe_s"] for r in results if r.get("ttfe_s") is not None]
    total_vals = [r["total_s"] for r in results if r.get("total_s") is not None]

    print("\n" + "=" * 60)
    print("BURST SUMMARY")
    print("=" * 60)
    print(f"  Total requests:    {total}")
    print(f"  Successful (200):  {ok}")
    print(f"  Queued:            {queued}")
    print(f"  429 (busy):        {errs_429}")
    print(f"  5xx:               {errs_5xx}")
    print(f"  Other errors:      {errs_other}")
    print(f"  Client timeouts:   {timeouts}")
    if ttfe_vals:
        print(f"  TTFE (min/med/max): {min(ttfe_vals):.1f} / {sorted(ttfe_vals)[len(ttfe_vals)//2]:.1f} / {max(ttfe_vals):.1f} s")
    if total_vals:
        print(f"  Total (min/med/max): {min(total_vals):.1f} / {sorted(total_vals)[len(total_vals)//2]:.1f} / {max(total_vals):.1f} s")
    print("=" * 60)

    # Pass/fail checks
    if errs_5xx > 0:
        print("  ** HARD FAIL: 5xx responses detected **")
    if errs_429 > 0:
        print(f"  ** NOTE: {errs_429} requests got 429 (should be rare with wait-in-SSE) **")
    if errs_5xx == 0 and errs_429 == 0:
        print("  PASS: No 5xx or 429 responses")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Copilot admission stress test")
    parser.add_argument("--url", default="https://dev-8.bv-brc.org/copilot-api/chatbrc/copilot-agent",
                        help="Copilot agent endpoint URL")
    parser.add_argument("--token", required=True,
                        help="BV-BRC auth token (full 'un=...|sig=...' string)")
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B",
                        help="Model name to use")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Number of concurrent virtual users")
    parser.add_argument("--turns", type=int, default=3,
                        help="Number of turns per virtual user")
    parser.add_argument("--think-min", type=float, default=0,
                        help="Min think time between turns (seconds)")
    parser.add_argument("--think-max", type=float, default=0,
                        help="Max think time between turns (seconds)")
    parser.add_argument("--jsonl", default=None,
                        help="Output JSONL file path (default: auto-generated)")
    parser.add_argument("--ramp", action="store_true",
                        help="Run the full ramp sequence: 2, 4, 8, 12, 16, 24")
    parser.add_argument("--ramp-pause", type=float, default=30,
                        help="Seconds to pause between ramp steps")
    args = parser.parse_args()

    if args.jsonl:
        jsonl_path = Path(args.jsonl)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        jsonl_path = Path(f"/tmp/stress_{ts}.jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    think_time = (args.think_min, args.think_max)

    print(f"Copilot Admission Stress Test")
    print(f"  URL:         {args.url}")
    print(f"  Model:       {args.model}")
    print(f"  JSONL:       {jsonl_path}")
    print()

    if args.ramp:
        # Full ramp sequence
        ramp_steps = [2, 4, 8, 12, 16, 24]
        all_results = []
        for step_concurrency in ramp_steps:
            print(f"\n{'='*60}")
            print(f"RAMP STEP: {step_concurrency} concurrent users, {args.turns} turns each")
            print(f"{'='*60}")
            results = asyncio.run(run_burst(
                url=args.url,
                token=args.token,
                model=args.model,
                concurrency=step_concurrency,
                turns_per_user=args.turns,
                jsonl_path=jsonl_path,
                think_time=think_time,
            ))
            all_results.extend(results)
            print_summary(results)

            # Check for hard fail
            if any(r.get("is_5xx") for r in results):
                print("\n** HARD FAIL — stopping ramp **")
                break

            if step_concurrency < ramp_steps[-1]:
                print(f"\nPausing {args.ramp_pause}s before next step...")
                time.sleep(args.ramp_pause)

        print(f"\n{'='*60}")
        print("OVERALL RAMP SUMMARY")
        print_summary(all_results)
    else:
        # Single burst
        print(f"  Concurrency: {args.concurrency}")
        print(f"  Turns/user:  {args.turns}")
        print()

        results = asyncio.run(run_burst(
            url=args.url,
            token=args.token,
            model=args.model,
            concurrency=args.concurrency,
            turns_per_user=args.turns,
            jsonl_path=jsonl_path,
            think_time=think_time,
        ))
        print_summary(results)

    print(f"\nFull results written to: {jsonl_path}")

if __name__ == "__main__":
    main()
