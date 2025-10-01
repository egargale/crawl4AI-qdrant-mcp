#!/usr/bin/env python3
"""
Script to search for specific FastMCP integrations in the Qdrant manuali collection.
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Add the mcp-server-qdrant src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mcp-server-qdrant', 'src'))

from qdrant_client import QdrantClient
from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider


def safe_print(text):
    """Safely print text with Unicode encoding handling."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback to ASCII encoding with error handling
        print(text.encode('ascii', errors='ignore').decode('ascii'))


async def search_specific_integrations():
    """Search for specific FastMCP integrations in the manuali collection."""
    
    # Load environment variables
    load_dotenv()
    
    collection_name = "manuali"
    query_texts = [
        "FastMCP Claude Desktop integration",
        "FastMCP OpenAPI integration",
        "FastMCP proxy server",
        "FastMCP authentication integration",
        "FastMCP GitHub integration"
    ]
    
    safe_print(f"Searching for specific FastMCP integrations in collection: {collection_name}")
    safe_print(f"Using embedding method: fastembed")
    
    try:
        # Create Qdrant client
        client = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))
        
        # Create embedding provider
        embedding_provider = FastEmbedProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        for query_text in query_texts:
            safe_print(f"\n--- Searching for '{query_text}' ---")
            
            # Generate embedding for the query
            safe_print("Generating embedding for query...")
            query_vector = await embedding_provider.embed_query(query_text)
            safe_print(f"Generated query vector with {len(query_vector)} dimensions")
            
            # Query the collection directly
            safe_print(f"Searching in collection '{collection_name}'...")
            search_result = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=3,
                with_payload=True
            )
            
            safe_print(f"Found {len(search_result.points)} results")
            
            # Display top results
            if search_result.points:
                for i, point in enumerate(search_result.points, 1):
                    safe_print(f"\n--- Result {i} ---")
                    safe_print(f"Query: {query_text}")
                    safe_print(f"ID: {point.id}")
                    safe_print(f"Score: {point.score:.4f}")
                    if hasattr(point, 'payload') and point.payload:
                        # Print document content if available
                        if 'document' in point.payload:
                            content = str(point.payload['document'])
                            # Handle Unicode characters safely
                            try:
                                # Show content but limit length to avoid encoding issues
                                clean_content = content.encode('utf-8', errors='ignore').decode('utf-8')
                                safe_print(f"Content: {clean_content[:600]}...")
                            except Exception:
                                # Ultimate fallback
                                safe_print(f"Content: {content[:200]}...")
                        # Print source metadata
                        if 'source' in point.payload:
                            safe_print(f"Source: {point.payload['source']}")
            else:
                safe_print("No results found for the query.")
                
        return True
        
    except Exception as e:
        safe_print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run the search."""
    safe_print("Searching 'manuali' collection for specific FastMCP integrations...")
    
    try:
        # Run the async search
        success = asyncio.run(search_specific_integrations())
        
        if success:
            safe_print("\nSearch completed successfully!")
            return 0
        else:
            safe_print("\nSearch failed!")
            return 1
            
    except Exception as e:
        safe_print(f"\nSearch error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())