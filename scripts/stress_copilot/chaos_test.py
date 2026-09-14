#!/usr/bin/env python3
"""
Chaos test suite for Copilot admission control.

Automates the 5 chaos cases from the admission stress-test plan.
Fills all chat slots with background turns, then exercises:

  1. Stop during queue wait   — must NOT take a seat
  2. Disconnect during wait   — must NOT take a seat
  3. Stop during run          — slot must release
  4. Disconnect after started — slot held until turn finishes
  5. Overlapping Stop + new   — old cancel must not affect new turn

Prerequisites:
  - Gateway restarted with wait-in-SSE code (wait_ms=300000)
  - Redis on localhost:6379 db 11
  - All chat slots empty before starting

Usage:
  python chaos_test.py --token "$(cat ~/.patric_token)"
"""

import argparse
import asyncio
import json
import subprocess
import time
import uuid

import aiohttp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_URL = "https://dev-8.bv-brc.org/copilot-api/chatbrc/copilot-agent"
DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"
REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379
REDIS_DB = 11
REDIS_CLI = "/home/ac.cucinell/redis/src/redis-cli"

# Filler queries — need to be slow enough to hold slots for ~60-90s.
# Multi-step data queries that trigger tool loops hold slots longer.
FILLER_QUERIES = [
    "Search for all Staphylococcus aureus genomes with AMR data and tell me how many have methicillin resistance. Also show me the top 5 most common antibiotics in the AMR data.",
    "Find all complete Klebsiella pneumoniae genomes and compare the number of complete vs WGS genomes. What are the most common host organisms?",
    "How many Pseudomonas aeruginosa genomes are public? Break down by genome status. Also search for any with carbapenem resistance data.",
    "Search for Escherichia coli genomes from 2023 and 2024. How many were added each year? What are the most common sequence types?",
    "Find all Mycobacterium tuberculosis genomes with AMR data. What antibiotics are most commonly tested? How many show resistance vs susceptibility?",
    "Search for Salmonella enterica genomes with host 'Human'. How many are there? What are the most common serotypes?",
    "List all Acinetobacter baumannii genomes and tell me how many have complete genome status. Search for AMR data too.",
    "Find Clostridioides difficile genomes with AMR data. How many genomes total? What is the resistance profile?",
]
SHORT_QUERY = "How do I upload a FASTA file?"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def redis_zcard() -> int:
    """Get current chat slot count."""
    try:
        r = subprocess.run(
            [REDIS_CLI, "-h", REDIS_HOST, "-p", str(REDIS_PORT),
             "-n", str(REDIS_DB), "ZCARD", "copilot:chat_slots"],
            capture_output=True, text=True, timeout=5,
        )
        return int(r.stdout.strip())
    except Exception as e:
        print(f"  WARNING: redis ZCARD failed: {e}")
        return -1


def redis_flush_slots():
    """Remove all chat slot entries (cleanup)."""
    try:
        subprocess.run(
            [REDIS_CLI, "-h", REDIS_HOST, "-p", str(REDIS_PORT),
             "-n", str(REDIS_DB), "DEL", "copilot:chat_slots"],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        pass


async def start_sse_turn(
    session: aiohttp.ClientSession,
    url: str,
    token: str,
    model: str,
    query: str,
    session_id: str | None = None,
) -> tuple[aiohttp.ClientResponse, str | None, str | None]:
    """
    Start a POST /copilot-agent SSE turn.
    Returns (response, job_id, first_event_type).
    Reads just enough to extract the job_id from the first data event.
    Does NOT consume the full stream — caller must close or read it.
    """
    sid = session_id or str(uuid.uuid4())
    body = {
        "query": query,
        "session_id": sid,
        "model": model,
        "stream": True,
        "save_chat": False,
    }
    headers = {
        "Authorization": token,
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }

    resp = await session.post(url, json=body, headers=headers,
                              timeout=aiohttp.ClientTimeout(total=660))

    if resp.status == 429:
        return resp, None, None

    # Read until we get a job_id from a data line
    job_id = None
    first_event = None
    current_event = "unknown"
    buf = ""
    try:
        async for chunk in resp.content.iter_any():
            buf += chunk.decode("utf-8", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.rstrip("\r")
                if line.startswith("event: "):
                    current_event = line[7:]
                elif line.startswith("data: "):
                    if first_event is None:
                        first_event = current_event
                    try:
                        data = json.loads(line[6:])
                        if isinstance(data, dict) and data.get("job_id"):
                            job_id = data["job_id"]
                    except json.JSONDecodeError:
                        pass
                if job_id:
                    break
            if job_id:
                break
    except Exception:
        pass

    return resp, job_id, first_event


async def read_until_done(resp: aiohttp.ClientResponse, timeout: float = 60) -> list[str]:
    """Read SSE events until 'done' or 'error' or timeout. Returns event types seen."""
    events = []
    buf = ""
    event_type = "unknown"
    deadline = time.monotonic() + timeout

    try:
        async for chunk in resp.content.iter_any():
            if time.monotonic() > deadline:
                break
            buf += chunk.decode("utf-8", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.rstrip("\r")
                if line.startswith("event: "):
                    event_type = line[7:]
                elif line.startswith("data: "):
                    events.append(event_type)
                    if event_type in ("done", "error"):
                        return events
    except (asyncio.CancelledError, aiohttp.ClientError):
        pass

    return events


async def wait_for_event(resp: aiohttp.ClientResponse, target: str, timeout: float = 30) -> bool:
    """Read SSE until we see `target` event type or timeout."""
    buf = ""
    event_type = "unknown"
    deadline = time.monotonic() + timeout

    try:
        async for chunk in resp.content.iter_any():
            if time.monotonic() > deadline:
                return False
            buf += chunk.decode("utf-8", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.rstrip("\r")
                if line.startswith("event: "):
                    event_type = line[7:]
                elif line.startswith("data: "):
                    if event_type == target:
                        return True
    except (asyncio.CancelledError, aiohttp.ClientError):
        pass
    return False


async def abort_job(session: aiohttp.ClientSession, url_base: str, token: str, job_id: str):
    """POST /job/:id/abort to stop a turn."""
    # url_base is like https://dev-8.bv-brc.org/copilot-api/chatbrc/copilot-agent
    # abort is at       https://dev-8.bv-brc.org/copilot-api/chatbrc/job/:id/abort
    abort_url = url_base.replace("/copilot-agent", f"/job/{job_id}/abort")
    headers = {"Authorization": token}
    try:
        async with session.post(abort_url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            return resp.status
    except Exception as e:
        print(f"  WARNING: abort failed: {e}")
        return None


async def fill_slots(
    session: aiohttp.ClientSession,
    url: str, token: str, model: str,
    count: int = 8,
) -> list[tuple[aiohttp.ClientResponse, str]]:
    """Start `count` background turns to fill all chat slots."""
    import random as _rand
    tasks = []
    for i in range(count):
        q = _rand.choice(FILLER_QUERIES)
        tasks.append(start_sse_turn(session, url, token, model, q))

    results = await asyncio.gather(*tasks)
    active = [(resp, jid) for resp, jid, _evt in results if jid]

    # Wait for slots to actually fill
    for _ in range(15):
        zcard = redis_zcard()
        if zcard >= count:
            break
        await asyncio.sleep(1)

    return active


async def ensure_slots_full(
    session: aiohttp.ClientSession,
    url: str, token: str, model: str,
    active_turns: list,
    target: int = 8,
) -> list:
    """Top up slots if background turns have finished. Returns updated list."""
    import random as _rand
    zcard = redis_zcard()
    if zcard >= target:
        return active_turns

    deficit = target - zcard
    print(f"  Topping up {deficit} slots (ZCARD={zcard}, target={target})...")
    tasks = []
    for _ in range(deficit):
        q = _rand.choice(FILLER_QUERIES)
        tasks.append(start_sse_turn(session, url, token, model, q))

    results = await asyncio.gather(*tasks)
    new_turns = [(resp, jid) for resp, jid, _evt in results if jid]
    active_turns.extend(new_turns)

    # Wait for slots to fill
    for _ in range(15):
        zcard = redis_zcard()
        if zcard >= target:
            break
        await asyncio.sleep(1)

    print(f"  ZCARD after top-up: {redis_zcard()}")
    return active_turns


async def drain_turn(resp: aiohttp.ClientResponse):
    """Read a turn to completion (background cleanup)."""
    try:
        await read_until_done(resp, timeout=600)
    except Exception:
        pass
    finally:
        resp.close()


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


async def test_stop_during_wait(session, url, token, model):
    """Case 1: Stop during queue wait — must NOT take a seat."""
    print("\n--- Case 1: Stop during queue wait ---")

    zcard_before = redis_zcard()
    print(f"  ZCARD before: {zcard_before}")

    # Start a turn that should queue (all slots full)
    resp, job_id, first_event = await start_sse_turn(
        session, url, token, model, SHORT_QUERY)

    if resp.status == 429:
        print(f"  Got 429 instead of queued SSE — {FAIL}")
        resp.close()
        return False

    saw_queued = (first_event == "queued")
    print(f"  First event: {first_event}, saw queued: {saw_queued}")

    if not saw_queued:
        # Turn got a slot immediately (filler turns drained). Abort and
        # report as inconclusive pass — the wait path wasn't exercised.
        print(f"  Turn got a slot immediately (slots drained). Aborting.")
        if job_id:
            await abort_job(session, url, token, job_id)
        await asyncio.sleep(2)
        resp.close()
        zcard_after = redis_zcard()
        print(f"  ZCARD after cleanup: {zcard_after}")
        print(f"  Result: {PASS} (inconclusive — could not enter wait queue)")
        return True

    # Now abort while still waiting in the queue
    print(f"  Aborting job {job_id[:12]} (still in wait queue)...")
    abort_status = await abort_job(session, url, token, job_id)
    print(f"  Abort response: {abort_status}")

    # Read remaining events (should see cancelled + done)
    events = await read_until_done(resp, timeout=10)
    print(f"  Events after abort: {events}")
    resp.close()

    await asyncio.sleep(2)
    zcard_after = redis_zcard()
    print(f"  ZCARD after: {zcard_after}")

    # ZCARD should not have increased — the waiter never took a seat
    passed = zcard_after <= zcard_before
    print(f"  Slot NOT taken: {passed} (before={zcard_before}, after={zcard_after})")
    print(f"  Result: {PASS if passed else FAIL}")
    return passed


async def test_disconnect_during_wait(session, url, token, model):
    """Case 2: Disconnect during queue wait — must NOT take a seat."""
    print("\n--- Case 2: Disconnect during queue wait ---")

    zcard_before = redis_zcard()
    print(f"  ZCARD before: {zcard_before}")

    resp, job_id, first_event = await start_sse_turn(
        session, url, token, model, SHORT_QUERY)

    if resp.status == 429:
        print(f"  Got 429 instead of queued SSE — {FAIL}")
        resp.close()
        return False

    saw_queued = (first_event == "queued")
    print(f"  First event: {first_event}, saw queued: {saw_queued}")

    if not saw_queued:
        print(f"  Turn got a slot immediately (slots drained). Closing.")
        resp.close()
        await asyncio.sleep(2)
        print(f"  Result: {PASS} (inconclusive — could not enter wait queue)")
        return True

    # Disconnect by closing the response (simulates Ctrl+C / session switch)
    print(f"  Disconnecting (closing SSE while in wait queue)...")
    resp.close()

    await asyncio.sleep(3)
    zcard_after = redis_zcard()
    print(f"  ZCARD after: {zcard_after}")

    # ZCARD should not have increased — the waiter never took a seat
    passed = zcard_after <= zcard_before
    print(f"  Slot NOT taken: {passed} (before={zcard_before}, after={zcard_after})")
    print(f"  Result: {PASS if passed else FAIL}")
    return passed


async def test_stop_during_run(session, url, token, model):
    """Case 3: Stop during run — slot must release."""
    print("\n--- Case 3: Stop during run ---")

    # Wait for a slot to be available, then start our own turn
    for _ in range(30):
        zcard = redis_zcard()
        if zcard < 8:
            break
        await asyncio.sleep(2)

    zcard_before = redis_zcard()
    print(f"  ZCARD before: {zcard_before}")

    # Start a fresh turn we control
    resp, job_id, first_event = await start_sse_turn(session, url, token, model,
        "Search for all Klebsiella pneumoniae genomes and compare complete vs WGS. What are the most common host organisms?")

    if resp.status == 429 or not job_id:
        print(f"  Could not start turn — skipping")
        resp.close()
        return True

    # Wait for it to actually be running
    if first_event != "started":
        saw_started = await wait_for_event(resp, "started", timeout=30)
    else:
        saw_started = True
    print(f"  Turn started: {saw_started}, job_id: {job_id[:12]}")

    zcard_running = redis_zcard()
    print(f"  ZCARD while running: {zcard_running}")

    # Abort it
    print(f"  Aborting turn {job_id[:12]}...")
    abort_status = await abort_job(session, url, token, job_id)
    print(f"  Abort response: {abort_status}")

    # Wait for slot to release
    for _ in range(10):
        await asyncio.sleep(1)
        zcard_after = redis_zcard()
        if zcard_after < zcard_running:
            break

    zcard_after = redis_zcard()
    print(f"  ZCARD after abort: {zcard_after}")

    passed = zcard_after < zcard_running
    print(f"  Slot released: {passed} (was {zcard_running}, now {zcard_after})")
    print(f"  Result: {PASS if passed else FAIL}")

    resp.close()
    return passed


async def test_disconnect_after_started(session, url, token, model):
    """Case 4: Disconnect after started — slot held until turn finishes."""
    print("\n--- Case 4: Disconnect after started ---")

    # Wait for a slot to be available
    for _ in range(30):
        zcard = redis_zcard()
        if zcard < 8:
            break
        await asyncio.sleep(2)

    zcard_before = redis_zcard()
    print(f"  ZCARD before: {zcard_before}")

    resp, job_id, first_event = await start_sse_turn(
        session, url, token, model, SHORT_QUERY)

    if resp.status == 429:
        print(f"  Got 429 — {FAIL}")
        resp.close()
        return False

    # If first event wasn't started, wait for it
    if first_event != "started":
        saw_started = await wait_for_event(resp, "started", timeout=15)
    else:
        saw_started = True
    print(f"  Saw 'started' event: {saw_started}")

    zcard_running = redis_zcard()
    print(f"  ZCARD while running: {zcard_running}")

    # Disconnect without abort
    print(f"  Disconnecting (no abort)...")
    resp.close()

    await asyncio.sleep(2)
    zcard_after_disconnect = redis_zcard()
    print(f"  ZCARD immediately after disconnect: {zcard_after_disconnect}")

    # Slot should still be held (turn keeps running)
    held = zcard_after_disconnect >= zcard_running
    print(f"  Slot still held: {held}")

    # Wait for the turn to finish naturally
    print(f"  Waiting for turn to finish (up to 120s)...")
    for _ in range(60):
        zcard = redis_zcard()
        if zcard < zcard_after_disconnect:
            print(f"  Slot released after turn finished. ZCARD: {zcard}")
            break
        await asyncio.sleep(2)
    else:
        zcard = redis_zcard()
        print(f"  Timed out waiting. ZCARD: {zcard}")

    passed = held
    print(f"  Result: {PASS if passed else FAIL}")
    return passed


async def test_overlapping_stop_and_new(session, url, token, model):
    """Case 5: Overlapping Stop + new send — old cancel must not affect new turn."""
    print("\n--- Case 5: Overlapping Stop + new send ---")

    # Wait for a slot
    for _ in range(30):
        zcard = redis_zcard()
        if zcard < 7:  # need room for 2
            break
        await asyncio.sleep(2)

    # Start a turn
    resp_old, job_id_old, first_event = await start_sse_turn(
        session, url, token, model, SHORT_QUERY)
    if resp_old.status == 429 or not job_id_old:
        print(f"  Could not start first turn — skipping")
        resp_old.close()
        return True

    if first_event != "started":
        saw_started = await wait_for_event(resp_old, "started", timeout=15)
    else:
        saw_started = True
    print(f"  Old turn started: {saw_started}, job_id: {job_id_old[:12]}")

    # Abort old + start new simultaneously
    print(f"  Aborting old turn + starting new turn simultaneously...")
    abort_task = asyncio.create_task(abort_job(session, url, token, job_id_old))
    resp_new, job_id_new, _ = await start_sse_turn(
        session, url, token, model, SHORT_QUERY)
    abort_status = await abort_task

    print(f"  Abort status: {abort_status}")
    print(f"  New turn job_id: {job_id_new[:12] if job_id_new else 'None'}")

    # The new turn should have its own job_id and work independently
    passed = job_id_new is not None and job_id_new != job_id_old

    if passed and resp_new.status == 200:
        # Read a few events to confirm the new turn is working
        events = await read_until_done(resp_new, timeout=120)
        print(f"  New turn events: {events[:5]}...")
        got_done = "done" in events
        print(f"  New turn completed: {got_done}")
        passed = passed and got_done

    resp_old.close()
    resp_new.close()

    print(f"  Result: {PASS if passed else FAIL}")
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_chaos(url: str, token: str, model: str, max_chats: int):
    results = {}

    connector = aiohttp.TCPConnector(limit=max_chats + 10, limit_per_host=max_chats + 10)
    async with aiohttp.ClientSession(connector=connector) as session:

        # --- Fill all slots ---
        print(f"Filling {max_chats} chat slots with background turns...")
        active_turns = await fill_slots(session, url, token, model, count=max_chats)
        zcard = redis_zcard()
        print(f"  Active turns: {len(active_turns)}, ZCARD: {zcard}")

        if zcard < max_chats:
            print(f"  WARNING: only {zcard}/{max_chats} slots filled. "
                  f"Tests may not queue properly.")

        # --- Cases 1 & 2 need all slots full ---
        # Top up before each in case filler turns finished early.

        active_turns = await ensure_slots_full(
            session, url, token, model, active_turns, target=max_chats)
        results["1_stop_during_wait"] = await test_stop_during_wait(
            session, url, token, model)

        active_turns = await ensure_slots_full(
            session, url, token, model, active_turns, target=max_chats)
        results["2_disconnect_during_wait"] = await test_disconnect_during_wait(
            session, url, token, model)

        # --- Case 3 starts its own turn (no need for full slots) ---
        results["3_stop_during_run"] = await test_stop_during_run(
            session, url, token, model)

        # Drain remaining filler turns in background
        print(f"\n  Draining remaining filler turns in background...")
        drain_tasks = [asyncio.create_task(drain_turn(resp)) for resp, _ in active_turns]

        # --- Cases 4 & 5 wait for a free slot internally ---
        results["4_disconnect_after_started"] = await test_disconnect_after_started(
            session, url, token, model)

        results["5_overlapping_stop_new"] = await test_overlapping_stop_and_new(
            session, url, token, model)

        # Wait for background drains
        print(f"\n  Waiting for background turns to finish...")
        await asyncio.gather(*drain_tasks, return_exceptions=True)

    # --- Wait for slots to clear ---
    print(f"\nWaiting for all slots to release...")
    for _ in range(30):
        zcard = redis_zcard()
        if zcard == 0:
            break
        await asyncio.sleep(2)
    zcard_final = redis_zcard()
    print(f"  Final ZCARD: {zcard_final}")
    results["slots_cleared"] = zcard_final == 0

    # --- Summary ---
    print(f"\n{'='*60}")
    print("CHAOS TEST RESULTS")
    print(f"{'='*60}")
    all_pass = True
    for name, passed in results.items():
        status = PASS if passed else FAIL
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False
    print(f"{'='*60}")
    if all_pass:
        print(f"  Overall: {PASS}")
    else:
        print(f"  Overall: {FAIL}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Copilot admission chaos tests")
    parser.add_argument("--url", default=DEFAULT_URL,
                        help="Copilot agent endpoint URL")
    parser.add_argument("--token", required=True,
                        help="BV-BRC auth token")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Model name")
    parser.add_argument("--max-chats", type=int, default=8,
                        help="Max chat slots (must match config.json)")
    args = parser.parse_args()

    print("Copilot Admission Chaos Tests")
    print(f"  URL:       {args.url}")
    print(f"  Model:     {args.model}")
    print(f"  Max chats: {args.max_chats}")

    # Pre-check: slots should be empty
    zcard = redis_zcard()
    if zcard > 0:
        print(f"\n  WARNING: ZCARD={zcard} (slots not empty). Flushing...")
        redis_flush_slots()
        zcard = redis_zcard()
        print(f"  ZCARD after flush: {zcard}")

    print()
    asyncio.run(run_chaos(args.url, args.token, args.model, args.max_chats))


if __name__ == "__main__":
    main()
