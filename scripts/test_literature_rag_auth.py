#!/usr/bin/env python3
"""Diagnostic script for literature RAG auth forwarding.

This script is intended to debug 401 Unauthorized errors seen from the
`search_literature` shared tool.

It performs requests against the configured gateway URL using:
- No Authorization header
- Authorization header (exact token)
- authorization header (lowercase key)
- Bearer-wrapped token

It also calls the shared tool directly with different (headers, config)
combinations to mimic agent execution.

Usage:
  python scripts/test_literature_rag_auth.py \
    --token "un=...|tokenid=..." \
    --url "http://ash.cels.anl.gov:12006" \
    --query "SARS-CoV-2 Spike ACE2 interaction" \
    --top-k 3

If --token is omitted, the script will try BV_BRC_AUTH_TOKEN from env.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests

# Ensure repo root (bvbrc-agents/) is on sys.path so `shared` imports work
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


@dataclass
class DummyConfig:
    literature_rag_url: str
    literature_rag_timeout_seconds: int = 45
    bvbrc_auth_token: Optional[str] = None


def _post(
    url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None
) -> None:
    try:
        r = requests.post(url, json=payload, headers=headers or {}, timeout=30)
        print(f"HTTP {r.status_code}")
        ct = r.headers.get("content-type", "")
        if "application/json" in ct:
            try:
                print(json.dumps(r.json(), indent=2)[:1500])
            except Exception:
                print(r.text[:1500])
        else:
            print(r.text[:1500])
    except Exception as e:
        print(f"REQUEST FAILED: {type(e).__name__}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--url", default="http://ash.cels.anl.gov:12006", help="Gateway base URL"
    )
    ap.add_argument("--token", default=None, help="BV-BRC auth token (raw value)")
    ap.add_argument("--query", default="SARS-CoV-2 Spike ACE2 interaction")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--use-graph", action="store_true")
    args = ap.parse_args()

    token = args.token or os.environ.get("BV_BRC_AUTH_TOKEN")

    retrieve_url = args.url.rstrip("/") + "/copilot-api/rag/retrieve"
    payload = {
        "query": args.query,
        "top_k": args.top_k,
        "use_graph": bool(args.use_graph),
    }

    print("=== Direct HTTP tests ===")
    print(f"URL: {retrieve_url}")
    print("\n[1] No Authorization header")
    _post(retrieve_url, payload, headers={"Content-Type": "application/json"})

    if token:
        print("\n[2] Authorization header (raw token)")
        _post(
            retrieve_url,
            payload,
            headers={"Content-Type": "application/json", "Authorization": token},
        )

        print("\n[3] authorization header (lowercase key)")
        _post(
            retrieve_url,
            payload,
            headers={"Content-Type": "application/json", "authorization": token},
        )

        print("\n[4] Authorization: Bearer <token>")
        _post(
            retrieve_url,
            payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
        )
    else:
        print("\n(no token provided; skipping authenticated cases)")

    print("\n=== Shared tool tests (mirrors agent execution) ===")
    try:
        from shared.tools.literature import search_literature

        # 1) No headers, no config token
        print("\n[tool 1] headers=None, config has no bvbrc_auth_token")
        cfg = DummyConfig(literature_rag_url=args.url, bvbrc_auth_token=None)
        out = search_literature  # async function
        # Run async function in simplest possible way
        import asyncio

        res = asyncio.run(
            out(
                query=args.query,
                top_k=args.top_k,
                use_graph=args.use_graph,
                config=cfg,
                headers=None,
            )
        )
        print(json.dumps(res, indent=2)[:1500])

        if token:
            print("\n[tool 2] headers has Authorization")
            cfg2 = DummyConfig(literature_rag_url=args.url, bvbrc_auth_token=None)
            res = asyncio.run(
                out(
                    query=args.query,
                    top_k=args.top_k,
                    use_graph=args.use_graph,
                    config=cfg2,
                    headers={"Authorization": token},
                )
            )
            print(json.dumps(res, indent=2)[:1500])

            print("\n[tool 3] headers has authorization (lowercase)")
            cfg3 = DummyConfig(literature_rag_url=args.url, bvbrc_auth_token=None)
            res = asyncio.run(
                out(
                    query=args.query,
                    top_k=args.top_k,
                    use_graph=args.use_graph,
                    config=cfg3,
                    headers={"authorization": token},
                )
            )
            print(json.dumps(res, indent=2)[:1500])

            print("\n[tool 4] token only in config.bvbrc_auth_token")
            cfg4 = DummyConfig(literature_rag_url=args.url, bvbrc_auth_token=token)
            res = asyncio.run(
                out(
                    query=args.query,
                    top_k=args.top_k,
                    use_graph=args.use_graph,
                    config=cfg4,
                    headers=None,
                )
            )
            print(json.dumps(res, indent=2)[:1500])

    except Exception as e:
        print(f"Shared tool test failed: {type(e).__name__}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
