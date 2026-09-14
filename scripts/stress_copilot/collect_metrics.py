#!/usr/bin/env python3
"""
Metrics collector for admission stress test.

Polls holly and ash observation endpoints (read-only) and writes
timestamped metrics to JSONL.  Runs alongside stress_copilot.py.

This is observation, not load generation.  All requests are local or
to the holly proxy health endpoint — never through nginx.

Usage:
  python collect_metrics.py --jsonl /tmp/metrics_20260914.jsonl

  # Or with custom endpoints:
  python collect_metrics.py \
    --proxy-health http://140.221.78.15:8005/health \
    --redis-host 127.0.0.1 --redis-port 6379 --redis-db 11 \
    --interval 2
"""

import argparse
import asyncio
import json
import os
import resource
import subprocess
import time
from pathlib import Path

import aiohttp

REDIS_CLI = "/home/ac.cucinell/redis/src/redis-cli"

# ---------------------------------------------------------------------------
# Metric collectors
# ---------------------------------------------------------------------------

async def fetch_proxy_health(session: aiohttp.ClientSession, url: str) -> dict:
    """Poll holly admission proxy /health."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status == 200:
                data = await resp.json()
                return {
                    "proxy_in_flight": data.get("in_flight"),
                    "proxy_waiting": data.get("waiting"),
                    "proxy_max_in_flight": data.get("max_in_flight"),
                    "proxy_max_waiting": data.get("max_waiting"),
                }
            return {"proxy_error": f"HTTP {resp.status}"}
    except Exception as e:
        return {"proxy_error": str(e)}


def fetch_redis_zcard(redis_host: str, redis_port: int, redis_db: int) -> dict:
    """Poll Redis ZCARD for chat slots."""
    try:
        result = subprocess.run(
            [REDIS_CLI, "-h", redis_host, "-p", str(redis_port), "-n", str(redis_db),
             "ZCARD", "copilot:chat_slots"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            count = int(result.stdout.strip())
            return {"redis_chat_slots": count}
        return {"redis_error": result.stderr.strip()}
    except Exception as e:
        return {"redis_error": str(e)}


def fetch_redis_zrange(redis_host: str, redis_port: int, redis_db: int) -> dict:
    """Get the full slot list (for debugging leaks)."""
    try:
        result = subprocess.run(
            [REDIS_CLI, "-h", redis_host, "-p", str(redis_port), "-n", str(redis_db),
             "ZRANGE", "copilot:chat_slots", "0", "-1", "WITHSCORES"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
            return {"redis_slot_members": len(lines) // 2, "redis_slot_raw": lines[:20]}
        return {}
    except Exception:
        return {}


def fetch_process_stats(pid: int) -> dict:
    """Get RSS and CPU for a process (Linux /proc)."""
    try:
        with open(f"/proc/{pid}/statm") as f:
            pages = int(f.read().split()[1])  # RSS in pages
            rss_mb = pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)

        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().split()
            utime = int(fields[13])
            stime = int(fields[14])

        return {
            "rss_mb": round(rss_mb, 1),
            "cpu_ticks": utime + stime,
        }
    except Exception as e:
        return {"proc_error": str(e)}


def find_pid_by_port(port: int) -> int | None:
    """Find the PID listening on a port via ss."""
    try:
        result = subprocess.run(
            ["ss", "-tlnp"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if f":{port}" in line:
                # Extract PID from pid=NNNN
                for part in line.split(","):
                    if "pid=" in part:
                        pid_str = part.split("pid=")[1].split(",")[0].split(")")[0]
                        return int(pid_str)
    except Exception:
        pass
    return None


def fetch_open_fds(pid: int) -> dict:
    """Count open file descriptors for a process."""
    try:
        fd_path = f"/proc/{pid}/fd"
        count = len(os.listdir(fd_path))
        return {"open_fds": count}
    except Exception as e:
        return {"fd_error": str(e)}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def collect_loop(
    jsonl_path: Path,
    proxy_health_url: str,
    redis_host: str,
    redis_port: int,
    redis_db: int,
    interval: float,
    orchestrator_pid: int | None,
    gateway_pid: int | None,
    duration: float | None,
):
    """Run the collection loop."""
    connector = aiohttp.TCPConnector(limit=5)
    async with aiohttp.ClientSession(connector=connector) as session:
        start = time.monotonic()
        tick = 0

        while True:
            if duration and (time.monotonic() - start) > duration:
                print(f"\nDuration limit reached ({duration}s), stopping.")
                break

            record = {
                "tick": tick,
                "timestamp": time.time(),
                "elapsed_s": round(time.monotonic() - start, 1),
            }

            # Proxy health (async)
            proxy_data = await fetch_proxy_health(session, proxy_health_url)
            record.update(proxy_data)

            # Redis ZCARD (sync subprocess)
            redis_data = fetch_redis_zcard(redis_host, redis_port, redis_db)
            record.update(redis_data)

            # Orchestrator process stats
            if orchestrator_pid:
                orch_stats = fetch_process_stats(orchestrator_pid)
                record["orchestrator_rss_mb"] = orch_stats.get("rss_mb")
                orch_fds = fetch_open_fds(orchestrator_pid)
                record["orchestrator_fds"] = orch_fds.get("open_fds")

            # Gateway process stats
            if gateway_pid:
                gw_stats = fetch_process_stats(gateway_pid)
                record["gateway_rss_mb"] = gw_stats.get("rss_mb")
                gw_fds = fetch_open_fds(gateway_pid)
                record["gateway_fds"] = gw_fds.get("open_fds")

            # Write to JSONL
            with open(jsonl_path, "a") as f:
                f.write(json.dumps(record) + "\n")

            # Print one-liner
            slots = record.get("redis_chat_slots", "?")
            inflight = record.get("proxy_in_flight", "?")
            waiting = record.get("proxy_waiting", "?")
            orch_rss = record.get("orchestrator_rss_mb", "?")
            gw_rss = record.get("gateway_rss_mb", "?")
            print(
                f"  t={record['elapsed_s']:6.1f}s  "
                f"slots={slots}  proxy_inflight={inflight}  proxy_wait={waiting}  "
                f"orch_rss={orch_rss}MB  gw_rss={gw_rss}MB"
            )

            tick += 1
            await asyncio.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Admission stress test metrics collector")
    parser.add_argument("--proxy-health", default="http://140.221.78.15:8005/health",
                        help="Holly admission proxy health URL")
    parser.add_argument("--redis-host", default="127.0.0.1",
                        help="Redis host (ash)")
    parser.add_argument("--redis-port", type=int, default=6379,
                        help="Redis port")
    parser.add_argument("--redis-db", type=int, default=11,
                        help="Redis DB number")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Polling interval in seconds")
    parser.add_argument("--duration", type=float, default=None,
                        help="Max collection duration in seconds (default: run forever)")
    parser.add_argument("--orchestrator-pid", type=int, default=None,
                        help="Orchestrator PID (auto-detect from port 9100 if omitted)")
    parser.add_argument("--gateway-pid", type=int, default=None,
                        help="Gateway PID (auto-detect from port 12008 if omitted)")
    parser.add_argument("--jsonl", default=None,
                        help="Output JSONL path (default: auto-generated)")
    args = parser.parse_args()

    # Auto-detect PIDs
    orch_pid = args.orchestrator_pid or find_pid_by_port(9100)
    gw_pid = args.gateway_pid or find_pid_by_port(12008)

    if orch_pid:
        print(f"Orchestrator PID: {orch_pid}")
    else:
        print("WARNING: Could not detect orchestrator PID (port 9100)")

    if gw_pid:
        print(f"Gateway PID: {gw_pid}")
    else:
        print("WARNING: Could not detect gateway PID (port 12008)")

    if args.jsonl:
        jsonl_path = Path(args.jsonl)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        jsonl_path = Path(f"/tmp/metrics_{ts}.jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Metrics output: {jsonl_path}")
    print(f"Polling every {args.interval}s")
    print(f"Press Ctrl+C to stop.\n")

    try:
        asyncio.run(collect_loop(
            jsonl_path=jsonl_path,
            proxy_health_url=args.proxy_health,
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            redis_db=args.redis_db,
            interval=args.interval,
            orchestrator_pid=orch_pid,
            gateway_pid=gw_pid,
            duration=args.duration,
        ))
    except KeyboardInterrupt:
        print(f"\nStopped. Metrics written to: {jsonl_path}")


if __name__ == "__main__":
    main()
