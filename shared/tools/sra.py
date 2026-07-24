"""
Shared SRA metadata retrieval tool.

Wraps the MCP server's ``sra_functions`` module to fetch metadata for
SRA run accessions (SRR IDs) using the ``p3-sra`` tool inside a
Singularity container.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from shared.tools._mcp_imports import get_sra_functions


async def get_sra_metadata(
    sra_ids: List[str],
    config: Any = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Retrieve metadata for one or more SRA run accessions.

    Args:
        sra_ids: List of SRA run accession IDs (e.g. ``["SRR37956035"]``).
        config: Agent configuration (provides ``singularity_container_path``,
                ``mcp_server_path``).
        headers: HTTP headers (unused, present for dispatcher compatibility).

    Returns:
        Dict with results (one per SRA ID), each containing ``sra_id``,
        ``success``, ``metadata``, and ``error``.
    """
    sra_fn = get_sra_functions(getattr(config, "mcp_server_path", None))

    container_path = getattr(config, "singularity_container_path", None)
    if not container_path:
        return {
            "error": "Singularity container path not configured.",
            "source": "shared-tools",
        }

    try:
        result = sra_fn.get_sra_metadata_func(sra_ids, container_path)
        return result

    except Exception as e:
        return {
            "error": f"SRA metadata retrieval failed: {type(e).__name__}: {str(e)}",
            "sra_ids": sra_ids,
        }
