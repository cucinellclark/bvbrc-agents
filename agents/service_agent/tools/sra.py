"""
Tool wrapper for SRA metadata retrieval.

Thin async wrapper that calls the MCP server's sra_functions module
to retrieve metadata for SRA run accessions (SRR IDs) using the
p3-sra tool inside a Singularity container.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional
from types import ModuleType

from service_agent.models import AgentConfig


# ---------------------------------------------------------------------------
# MCP server import helper
# ---------------------------------------------------------------------------

_sra_functions: Optional[ModuleType] = None
_path_added: bool = False


def _ensure_path(config: AgentConfig | None = None) -> None:
    global _path_added
    if _path_added:
        return
    cfg = config or AgentConfig()
    mcp_path = cfg.mcp_server_path
    if mcp_path and mcp_path not in sys.path:
        sys.path.insert(0, mcp_path)
        _path_added = True


def _get_sra_functions(config: AgentConfig | None = None) -> ModuleType:
    global _sra_functions
    if _sra_functions is None:
        _ensure_path(config)
        from functions import sra_functions
        _sra_functions = sra_functions
    return _sra_functions


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------

async def get_sra_metadata(
    sra_ids: List[str],
    config: Optional[AgentConfig] = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Retrieve metadata for one or more SRA run accessions.

    Uses the p3-sra tool inside a Singularity container to fetch metadata
    including organism name, sequencing platform, library strategy, and
    sample details.

    Args:
        sra_ids: List of SRA run accession IDs (e.g., ["SRR37956035"]).
        config: Agent configuration (provides Singularity container path).
        headers: HTTP headers (unused, present for dispatcher compatibility).

    Returns:
        Dict with results (one per SRA ID), each containing:
          - sra_id: The SRA accession
          - success: bool
          - metadata: dict with fields like sample_organism, sample_taxon, etc.
          - error: str or None
    """
    cfg = config or AgentConfig()
    sra_fn = _get_sra_functions(cfg)

    container_path = cfg.singularity_container_path
    if not container_path:
        return {
            "error": "Singularity container path not configured.",
            "source": "service-agent",
        }

    try:
        result = sra_fn.get_sra_metadata_func(sra_ids, container_path)
        return result

    except Exception as e:
        return {
            "error": f"SRA metadata retrieval failed: {type(e).__name__}: {str(e)}",
            "sra_ids": sra_ids,
        }
