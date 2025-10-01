#!/usr/bin/env python3
"""
Simple Qdrant document retrieval script.

This script provides a simple interface to query documents from a Qdrant collection
using semantic search with embeddings.
"""

import os
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from openai import OpenAI


def get_qdrant_client() -> QdrantClient:
    """Initialize and return a Qdrant client."""
    QDRANT_URL = os.getenv('QDRANT_URL')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
    
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in the .env file")
    
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def get_embedding_client() -> OpenAI:
    """Initialize and return an OpenAI-compatible embedding client."""
    return OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    )


def generate_embedding(client: OpenAI, text: str, model_name: str, dimensions: int) -> List[float]:
    """Generate embedding for a single text."""
    resp = client.embeddings.create(
        model=model_name,
        input=[text],
        dimensions=dimensions
    )
    return resp.data[0].embedding


def retrieve_documents(
    qdrant_client: QdrantClient,
    embedding_client: OpenAI,
    collection_name: str,
    model_name: str,
    embedding_dim: int,
    query_text: str,
    n_results: int = 5,
    score_threshold: float = None
) -> List[Dict[str, Any]]:
    """Retrieve documents from Qdrant collection based on semantic similarity.
    
    Args:
        qdrant_client: Qdrant client instance
        embedding_client: OpenAI-compatible embedding client
        collection_name: Name of the Qdrant collection
        model_name: Name of the embedding model
        embedding_dim: Dimension of embeddings
        query_text: Text to search for
        n_results: Number of results to return
        score_threshold: Minimum similarity score (0-1) to filter results
        
    Returns:
        List of dictionaries containing retrieved documents with metadata
    """
    # Generate embedding for the query
    query_embedding = generate_embedding(embedding_client, query_text, model_name, embedding_dim)
    
    # Query the collection
    search_result = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=n_results,
        with_payload=True,
        score_threshold=score_threshold
    )
    
    # Format results
    results = []
    for point in search_result.points:
        result = {
            "id": point.id,
            "score": point.score,
            "document": point.payload.get("document", ""),
            "metadata": {k: v for k, v in point.payload.items() if k != "document"}
        }
        results.append(result)
    
    return results


def format_results(results: List[Dict[str, Any]], show_full_content: bool = False) -> str:
    """Format retrieval results for display.
    
    Args:
        results: List of retrieved documents
        show_full_content: Whether to show full document content
        
    Returns:
        Formatted string for display
    """
    if not results:
        return "No documents found matching the query."
    
    output = []
    output.append(f"Found {len(results)} relevant documents:\n")
    
    for i, result in enumerate(results, 1):
        output.append(f"Result {i}:")
        output.append(f"  ID: {result['id']}")
        output.append(f"  Score: {result['score']:.4f}")
        
        # Add metadata
        if result['metadata']:
            output.append("  Metadata:")
            for key, value in result['metadata'].items():
                output.append(f"    {key}: {value}")
        
        # Add document content
        if show_full_content:
            output.append(f"  Content:\n{result['document']}")
        else:
            # Show first 200 characters
            preview = result['document'][:200] + "..." if len(result['document']) > 200 else result['document']
            output.append(f"  Content Preview:\n{preview}")
        
        output.append("")  # Empty line between results
    
    return "\n".join(output)


def main():
    """Main function to handle command-line arguments and run retrieval."""
    parser = argparse.ArgumentParser(
        description="Simple Qdrant document retrieval using semantic search"
    )
    parser.add_argument("query", help="Search query text")
    parser.add_argument("--collection", default="docs", help="Qdrant collection name (default: docs)")
    parser.add_argument("--embedding-model", default="text-embedding-v3", help="Embedding model name (default: text-embedding-v3)")
    parser.add_argument("--embedding-dim", type=int, default=1024, help="Embedding dimension (default: 1024)")
    parser.add_argument("--n-results", type=int, default=5, help="Number of results to return (default: 5)")
    parser.add_argument("--score-threshold", type=float, help="Minimum similarity score (0-1) to filter results")
    parser.add_argument("--full-content", action="store_true", help="Show full document content instead of preview")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    try:
        # Initialize clients
        qdrant_client = get_qdrant_client()
        embedding_client = get_embedding_client()
        
        print(f"Connected to Qdrant at: {os.getenv('QDRANT_URL')}")
        print(f"Querying collection '{args.collection}' for: '{args.query}'")
        print("-" * 50)
        
        # Retrieve documents
        results = retrieve_documents(
            qdrant_client=qdrant_client,
            embedding_client=embedding_client,
            collection_name=args.collection,
            model_name=args.embedding_model,
            embedding_dim=args.embedding_dim,
            query_text=args.query,
            n_results=args.n_results,
            score_threshold=args.score_threshold
        )
        
        # Format and display results
        formatted_output = format_results(results, show_full_content=args.full_content)
        print(formatted_output)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
       