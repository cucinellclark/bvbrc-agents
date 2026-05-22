#!/usr/bin/env python3
"""
Test script for the RAG API.

This script queries the RAG API with a test query and prints the returned documents.
"""

import json
import sys
import argparse
import requests
from typing import Optional


def query_rag_api(
    query: str,
    database_name: str = "bvbrc_helpdesk",
    api_base_url: str = "http://127.0.0.1:8000",
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
    timeout: int = 45,
) -> dict:
    """
    Query the RAG API and return the response.
    
    Args:
        query: The search query text
        database_name: Name of the RAG database to query
        api_base_url: Base URL of the RAG API
        top_k: Number of documents to retrieve (uses API default if None)
        score_threshold: Minimum similarity score (uses API default if None)
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary containing the API response
    """
    query_url = f"{api_base_url.rstrip('/')}/query/{database_name}"
    
    payload = {
        "query": query,
    }
    
    if top_k is not None:
        payload["top_k"] = top_k
    if score_threshold is not None:
        payload["score_threshold"] = score_threshold
    
    print(f"Querying RAG API: {query_url}", file=sys.stderr)
    print(f"Query: {query}", file=sys.stderr)
    print(f"Payload: {json.dumps(payload, indent=2)}", file=sys.stderr)
    print("-" * 80, file=sys.stderr)
    
    try:
        response = requests.post(
            query_url,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to query RAG API: {e}", file=sys.stderr)
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}", file=sys.stderr)
            print(f"Response body: {e.response.text}", file=sys.stderr)
        sys.exit(1)


def print_documents(response: dict):
    """
    Print the documents from the RAG API response in a readable format.
    
    Args:
        response: The JSON response from the RAG API
    """
    print("\n" + "=" * 80)
    print("RAG API Response")
    print("=" * 80)
    
    print(f"\nQuery: {response.get('query', 'N/A')}")
    print(f"Database: {response.get('database', 'N/A')}")
    print(f"Total Results: {response.get('total_results', 0)}")
    
    documents = response.get('documents', [])
    
    if not documents:
        print("\nNo documents returned.")
        return
    
    print(f"\nRetrieved {len(documents)} document(s):\n")
    
    for i, doc in enumerate(documents, 1):
        print("-" * 80)
        print(f"Document {i}")
        print("-" * 80)
        print(f"Score: {doc.get('score', 'N/A')}")
        
        metadata = doc.get('metadata', {})
        if metadata:
            print(f"Metadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
        
        content = doc.get('content', '')
        if content:
            print(f"\nContent:")
            # Print content with word wrapping
            words = content.split()
            line = ""
            for word in words:
                if len(line) + len(word) + 1 > 80:
                    print(line)
                    line = word
                else:
                    line = line + (" " if line else "") + word
            if line:
                print(line)
        else:
            print("\nContent: (empty)")
        
        print()
    
    # Optionally print embedding info
    embedding = response.get('embedding')
    if embedding:
        print("-" * 80)
        print(f"Query Embedding: {len(embedding)} dimensions")
        print("-" * 80)


def main():
    """Main entry point for the test script."""
    parser = argparse.ArgumentParser(
        description="Test script for querying the RAG API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default query
  python test_rag_api.py
  
  # Custom query
  python test_rag_api.py --query "how do I submit a job?"
  
  # Custom database and API URL
  python test_rag_api.py --database my_database --api-url http://localhost:8000
  
  # Limit results
  python test_rag_api.py --top-k 5
        """
    )
    
    parser.add_argument(
        "--query",
        type=str,
        default="how do i use the comprehensive genome analysis service?",
        help="The search query (default: 'how do i use the comprehensive genome analysis service?')"
    )
    
    parser.add_argument(
        "--database",
        type=str,
        default="bvbrc_helpdesk",
        help="Name of the RAG database to query (default: 'bvbrc_helpdesk')"
    )
    
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Base URL of the RAG API (default: 'http://127.0.0.1:8000')"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of documents to retrieve (uses API default if not specified)"
    )
    
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Minimum similarity score threshold (uses API default if not specified)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=45,
        help="Request timeout in seconds (default: 45)"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON response instead of formatted text"
    )
    
    args = parser.parse_args()
    
    # Query the API
    response = query_rag_api(
        query=args.query,
        database_name=args.database,
        api_base_url=args.api_url,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        timeout=args.timeout,
    )
    
    # Print results
    if args.json:
        print(json.dumps(response, indent=2))
    else:
        print_documents(response)


if __name__ == "__main__":
    main()

