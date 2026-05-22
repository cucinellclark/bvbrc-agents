#!/usr/bin/env python3
"""
RAG Database Tools

This module contains MCP tools for querying and managing RAG (Retrieval-Augmented Generation) databases.
"""

import json
import sys
from typing import Optional, Dict, Any
from fastmcp import FastMCP

from functions.rag_database_functions import (
    query_rag_helpdesk_func,
    list_publication_datasets_func,
)

def register_rag_database_tools(mcp: FastMCP, config: dict = None):
    """
    Register all RAG database-related MCP tools with the FastMCP server.
    
    Args:
        mcp: FastMCP server instance
        config: Configuration dictionary for RAG database settings
    """
    if config is None:
        config = {}
    
    top_k_default = config.get("top_k_default", 5)

    @mcp.tool(name="helpdesk_service_usage")
    def helpdesk_service_usage(
        query: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Search BV-BRC help guides/FAQs and return a short grounded usage answer.

        Use this for "how-to" guidance, feature usage, and support-style questions
        about the BV-BRC website and applications.

        ⚠️ USE THIS TOOL FOR:
        - How to use BV-BRC services, applications, and workflows
        - FAQ-style questions about platform features and capabilities
        - Troubleshooting guidance, parameter explanations, and example runs
        - Questions about BV-BRC website pages, tools, and documentation
        - General "how does this work?" questions about BV-BRC usage

        🚫 DO NOT USE THIS TOOL FOR:
        - Authoritative app/service listings (use list_service_apps)
        - Exact submission parameter schemas for a service (use get_service_submission_schema)
        - Querying biological records from data collections (use bvbrc_search_data with advanced=true)
        - Broad cross-collection record search (use bvbrc_search_data with advanced=false)
        - Retrieving specific biological records or dataset content directly
        - Workspace file browsing/downloading tasks (use workspace tools)
        
        Args:
            query: A help-guide or FAQ question about using BV-BRC features.
            top_k: Number of top results to return (uses config default if not provided).

        Returns:
            JSON string with query results:
            - results: list of retrieved documents with scores and metadata
            - summary: LLM-generated summary that answers the query using the retrieved documents
            - used_documents: documents that were provided to the summarizer
            - count: number of results returned
            - query: the original query
        """
        exec_top_k = top_k if top_k is not None else top_k_default

        print(
            f"Querying BV-BRC helpdesk: {query} (top_k={exec_top_k})...",
            file=sys.stderr,
        )
        try:
            result = query_rag_helpdesk_func(
                query=query,
                top_k=exec_top_k,
                config=config,
            )
            return result
        except Exception as e:
            return {
                "error": f"Error querying RAG helpdesk: {str(e)}",
                "errorType": "API_ERROR",
                "results": [],
                "count": 0,
                "source": "bvbrc-rag"
            }

    # @mcp.tool()
    def list_publication_datasets(
        query: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Find publication datasets relevant to a natural-language topic query.

        This tool is intended for publication/dataset discovery semantics
        (dataset titles/metadata), not direct record retrieval from BV-BRC
        Solr collections.

        ⚠️ USE THIS TOOL FOR:
        - "What publication datasets are relevant to X?"
        - Finding candidate datasets before downstream analysis

        🚫 DO NOT USE THIS TOOL FOR:
        - Querying primary data collections (use bvbrc_search_data)
        - Help/FAQ usage guidance (use helpdesk_service_usage)
        - Workspace file operations (use workspace tools)

        Args:
            query: The search query string, typically describing an analysis or data need.
            top_k: Optional maximum number of datasets to return (uses config default if not provided).

        Returns:
            JSON string with dataset results:
            - results: list of datasets with identifiers, titles, and metadata
            - count: number of results returned
            - query: the original query
        """
        exec_top_k = top_k if top_k is not None else top_k_default

        print(
            f"Listing publication datasets for query: {query} (top_k={exec_top_k})...",
            file=sys.stderr,
        )
        try:
            result = list_publication_datasets_func(
                query=query,
                top_k=exec_top_k,
                config=config,
            )
            return result
        except Exception as e:
            return {
                "error": f"Error listing publication datasets: {str(e)}",
                "errorType": "API_ERROR",
                "results": [],
                "count": 0,
                "source": "bvbrc-rag"
            }

