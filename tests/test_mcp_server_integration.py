#!/usr/bin/env python3
"""
Integration test for querying the 'manuali' collection using direct Qdrant client approach.
This approach works with the existing collection configuration.
"""

import os
import sys
import asyncio
import argparse
from typing import List
from dotenv import load_dotenv

# Add the mcp-server-qdrant src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mcp-server-qdrant', 'src'))

from qdrant_client import QdrantClient
from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider


async def test_manuali_collection_query():
    """Test querying the 'manuali' collection using direct Qdrant client approach."""
    
    # Load environment variables
    load_dotenv()
    
    collection_name = "manuali"
    query_text = "fastmcp and OAuth providers"
    
    print(f"Querying collection: {collection_name}")
    print(f"Search query: '{query_text}'")
    print(f"Using embedding method: fastembed")
    
    try:
        # Create Qdrant client
        client = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))
        
        # Create embedding provider
        embedding_provider = FastEmbedProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Generate embedding for the query
        print("Generating embedding for query...")
        query_vector = await embedding_provider.embed_query(query_text)
        print(f"Generated query vector with {len(query_vector)} dimensions")
        
        # Query the collection directly
        print(f"Searching for '{query_text}' in collection '{collection_name}'...")
        search_result = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=10,
            with_payload=True
        )
        
        print(f"Found {len(search_result.points)} results")
        
        # Display results
        if search_result.points:
            print("\nResults:")
            for i, point in enumerate(search_result.points, 1):
                print(f"\n--- Result {i} ---")
                print(f"ID: {point.id}")
                print(f"Score: {point.score}")
                if hasattr(point, 'payload') and point.payload:
                    # Print document content if available
                    if 'document' in point.payload:
                        content = str(point.payload['document'])
                        print(f"Content: {content[:500]}...")
                    # Print other payload data
                    other_payload = {k: v for k, v in point.payload.items() if k != 'document'}
                    if other_payload:
                        print(f"Metadata: {other_payload}")
        else:
            print("No results found for the query.")
            
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run the integration test."""
    print("Running integration test for querying 'manuali' collection...")
    print("Note: The 'manuali' collection contains AgentScope documentation.")
    
    try:
        # Run the async test
        success = asyncio.run(test_manuali_collection_query())
        
        if success:
            print("\nQuery test completed successfully!")
            print("This demonstrates that the 'manuali' collection can be queried successfully")
            print("using fastembed embeddings, which is the core requirement.")
            return 0
        else:
            print("\nQuery test failed!")
            return 1
            
    except Exception as e:
        print(f"\nQuery test error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())