"""
Shared Similar Genome Finder tool.

Calls the BV-BRC MinHash service (``minhash_service``) to find public
genomes similar to a query genome ID or a workspace FASTA file.

The MinHash service exposes two JSON-RPC methods:
  - ``Minhash.compute_genome_distance_for_genome2``  (genome ID input)
  - ``Minhash.compute_genome_distance_for_fasta2``   (FASTA file input)

Each returns a list of 4-tuples: ``[genome_id, distance, pvalue, kmer_counts]``.
"""

from __future__ import annotations

import uuid
import logging
from typing import Any, Dict, Optional

from shared.tools._mcp_imports import get_json_rpc

logger = logging.getLogger(__name__)

# Module-level cache so we only build the JsonRpcCaller once per process.
_minhash_caller = None
_minhash_caller_url: str | None = None


def _get_minhash_caller(config: Any = None):
    """Build (or reuse) a ``JsonRpcCaller`` pointed at the MinHash service."""
    global _minhash_caller, _minhash_caller_url

    url = (
        getattr(config, "similar_genome_finder_url", None)
        or "https://p3.theseed.org/services/minhash_service"
    )

    # Reuse if URL hasn't changed.
    if _minhash_caller is not None and _minhash_caller_url == url:
        return _minhash_caller

    json_rpc_mod = get_json_rpc(getattr(config, "mcp_server_path", None))
    _minhash_caller = json_rpc_mod.JsonRpcCaller(url, timeout=120.0)
    _minhash_caller_url = url
    return _minhash_caller


def _get_auth(config: Any = None) -> str:
    return getattr(config, "bvbrc_auth_token", None) or ""


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------


async def find_similar_genomes(
    genome_id: Optional[str] = None,
    fasta_file: Optional[str] = None,
    max_pvalue: float = 0.01,
    max_distance: float = 0.01,
    max_hits: int = 50,
    scope: str = "reference",
    include_bacterial: bool = True,
    include_viral: bool = True,
    config: Any = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Find public BV-BRC genomes similar to a query using Mash/MinHash.

    Provide exactly ONE of ``genome_id`` or ``fasta_file``.

    Args:
        genome_id: A BV-BRC genome ID (e.g. ``"83332.12"``).
        fasta_file: Workspace path to a FASTA/contigs file.
        max_pvalue: P-value threshold (default 0.01).
        max_distance: Mash distance threshold (default 0.01).
        max_hits: Max results to return (default 50).
        scope: ``"reference"`` (ref+rep only) or ``"all"`` (all public).
        include_bacterial: Include bacterial/archaeal genomes.
        include_viral: Include viral genomes.
        config: Agent config object (provides auth token and MinHash URL).
        headers: HTTP headers (unused; present for dispatcher compat).

    Returns:
        Dict with ``results`` (list of dicts with genome_id, distance,
        pvalue, kmer_counts), ``count``, ``query`` echo, and ``source``.
    """
    # --- auth ---
    auth_token = _get_auth(config)
    if not auth_token:
        return {
            "error": "No authentication token available",
            "errorType": "AUTHENTICATION_FAILED",
            "source": "bvbrc-similar-genome",
        }

    # --- input validation ---
    has_genome = bool(genome_id and str(genome_id).strip())
    has_fasta = bool(fasta_file and str(fasta_file).strip())

    if not has_genome and not has_fasta:
        return {
            "error": "Either genome_id or fasta_file is required",
            "errorType": "INVALID_PARAMETERS",
            "hint": (
                "Provide a BV-BRC genome ID (e.g. '83332.12') "
                "or a workspace path to a FASTA file"
            ),
            "source": "bvbrc-similar-genome",
        }
    if has_genome and has_fasta:
        return {
            "error": "Provide genome_id OR fasta_file, not both",
            "errorType": "INVALID_PARAMETERS",
            "source": "bvbrc-similar-genome",
        }
    if not include_bacterial and not include_viral:
        return {
            "error": "At least one of include_bacterial or include_viral must be true",
            "errorType": "INVALID_PARAMETERS",
            "source": "bvbrc-similar-genome",
        }

    # --- map scope to reference/representative flags ---
    if scope == "reference":
        inc_ref, inc_rep = 1, 1
    else:
        inc_ref, inc_rep = 0, 0

    inc_bac = 1 if include_bacterial else 0
    inc_vir = 1 if include_viral else 0

    # --- build JSON-RPC call ---
    if has_genome:
        method = "Minhash.compute_genome_distance_for_genome2"
        query_value = str(genome_id).strip()
    else:
        method = "Minhash.compute_genome_distance_for_fasta2"
        query_value = str(fasta_file).strip()

    params = [
        query_value, max_pvalue, max_distance, max_hits,
        inc_ref, inc_rep, inc_bac, inc_vir,
    ]
    request_id = int(str(uuid.uuid4().int)[:12])

    try:
        caller = _get_minhash_caller(config)
        raw = await caller.acall(method, params, request_id, auth_token)
    except ValueError as e:
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "hint": (
                "The MinHash service returned an error. "
                "Check that the genome ID or FASTA path is valid."
            ),
            "source": "bvbrc-similar-genome",
        }
    except Exception as e:
        logger.error("MinHash call failed: %s", e)
        return {
            "error": f"MinHash service error: {type(e).__name__}: {e}",
            "errorType": "API_ERROR",
            "source": "bvbrc-similar-genome",
        }

    # --- parse results ---
    # The service returns result[0] as a list of 4-tuples:
    #   [genome_id, distance, pvalue, kmer_counts]
    hits = []
    try:
        records = raw
        if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list):
            records = raw[0]

        if isinstance(records, list):
            for entry in records:
                if isinstance(entry, (list, tuple)) and len(entry) >= 4:
                    hits.append({
                        "genome_id": entry[0],
                        "distance": entry[1],
                        "pvalue": entry[2],
                        "kmer_counts": entry[3],
                    })
                elif isinstance(entry, dict):
                    hits.append(entry)
    except Exception:
        return {
            "raw_result": raw,
            "error": "Could not parse MinHash service response",
            "errorType": "PARSE_ERROR",
            "source": "bvbrc-similar-genome",
        }

    return {
        "results": hits,
        "count": len(hits),
        "query": {
            "genome_id": genome_id if has_genome else None,
            "fasta_file": fasta_file if has_fasta else None,
            "max_pvalue": max_pvalue,
            "max_distance": max_distance,
            "max_hits": max_hits,
            "scope": scope,
            "include_bacterial": include_bacterial,
            "include_viral": include_viral,
        },
        "source": "bvbrc-similar-genome",
    }
