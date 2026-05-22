#!/usr/bin/env python3
"""
SRA Tools

This module contains MCP tools for retrieving SRA (Sequence Read Archive) metadata.
"""

import json
import sys
from typing import List, Dict, Any, Optional
from fastmcp import FastMCP

from functions.sra_functions import get_sra_metadata_func


def register_sra_tools(mcp: FastMCP, config: dict = None):
    """
    Register all SRA-related MCP tools with the FastMCP server.
    
    Args:
        mcp: FastMCP server instance
        config: Configuration dictionary for SRA tools settings
    """
    if config is None:
        config = {}
    
    singularity_container = config.get("singularity_container")
    
    if not singularity_container:
        print("Warning: singularity_container not configured for SRA tools", file=sys.stderr)
    
    @mcp.tool()
    def get_sra_metadata(sra_ids: List[str], session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve metadata for one or more SRA (Sequence Read Archive) IDs.
        
        This tool uses the p3-sra command within a Singularity container to fetch
        metadata for SRA runs. The metadata includes information about the sequencing
        run, library preparation, sample details, and experimental design.

        USE THIS TOOL FOR:
        - Metadata lookup by known SRA run IDs (e.g., SRR...)

        DO NOT USE THIS TOOL FOR:
        - Downloading reads or sequence content
        - General BV-BRC workspace/file operations
        - Natural-language help/FAQ guidance
        
        Args:
            sra_ids: List of SRA run IDs (e.g., ["SRR37108646", "SRR37108647"])
            session_id: does nothing, just here for compatibility
        
        Returns:
            Dictionary with:
            - results: List of result objects, one per SRA ID, each containing:
              - sra_id: The SRA ID that was processed
              - success: Boolean indicating if the request succeeded
              - metadata: JSON object with SRA metadata (if successful)
              - error: Error message (if failed)
            - total_requested: Total number of SRA IDs requested
            - total_successful: Number of successfully processed IDs
            - total_failed: Number of failed IDs
        """
        if not singularity_container:
            return {
                "error": True,
                "errorType": "CONFIGURATION_ERROR",
                "message": "Singularity container path not configured",
                "source": "bvbrc-sra-tools"
            }
        
        print(f"Retrieving metadata for SRA IDs: {sra_ids}...", file=sys.stderr)
        
        try:
            result = get_sra_metadata_func(sra_ids, singularity_container)
            return result
        except Exception as e:
            return {
                "error": True,
                "errorType": "EXECUTION_ERROR",
                "message": f"Error retrieving SRA metadata: {str(e)}",
                "source": "bvbrc-sra-tools"
            }

