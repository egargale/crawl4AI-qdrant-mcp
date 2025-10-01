#!/usr/bin/env python3
"""
Test script for the Qdrant retrieval functionality.
This demonstrates how to use the retrieve_docs_qdrant.py script programmatically.
"""

import os
import argparse
from dotenv import load_dotenv
from retrieve_docs_qdrant import (
    get_qdrant_client,
    get_embedding_client,
    retrieve_documents,
    format_results
)

def test_retrieval(query_text="sc.AddAlertLine", collection_name="docs", n_results=5):
    """Test the retrieval functionality with a sample query."""
    # Load environment variables
    load_dotenv()
    
    # Initialize clients
    qdrant_client = get_qdrant_client()
    embedding_client = get_embedding_client()
    
    print(f"Testing retrieval with query: '{query_text}'")
    print(f"Using collection: '{collection_name}'")
    print(f"Number of results: {n_results}")
    print("-" * 50)
    
    try:
        # Retrieve documents
        results = retrieve_documents(
            qdrant_client=qdrant_client,
            embedding_client=embedding_client,
            collection_name=collection_name,
            model_name="text-embedding-v3",
            embedding_dim=1024,
            query_text=query_text,
            n_results=n_results
        )
        
        # Format and display results
        formatted_output = format_results(results, show_full_content=False)
        print(formatted_output)
        
        return True
        
    except Exception as e:
        print(f"Error during retrieval test: {e}")
        return False

def main():
    """Main function to parse command line arguments and run the test."""
    parser = argparse.ArgumentParser(description="Test Qdrant retrieval functionality")
    parser.add_argument("query", nargs="?", default="sc.AddAlertLine", help="Search query text (default: 'sc.AddAlertLine')")
    parser.add_argument("--collection", default="docs", help="Qdrant collection name (default: 'docs')")
    parser.add_argument("--n-results", type=int, default=5, help="Number of results to return (default: 5)")
    
    args = parser.parse_args()
    
    success = test_retrieval(args.query, args.collection, args.n_results)
    if success:
        print("\nRetrieval test completed successfully!")
    else:
        print("\nRetrieval test failed!")
        exit(1)

if __name__ == "__main__":
    main()