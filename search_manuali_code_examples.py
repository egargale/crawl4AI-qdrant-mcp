#!/usr/bin/env python3
"""
Script to search the 'manuali' collection for code examples and functions.
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


async def search_manuali_collection():
    """Search the 'manuali' collection for code examples and functions."""
    
    # Load environment variables
    load_dotenv()
    
    collection_name = "manuali"
    # More specific query terms for code examples and functions
    query_texts = [
        "FastMCP tool example",
        "MCP server implementation",
        "OAuth provider configuration",
        "function decorator usage",
        "API endpoint definition",
        "code snippet example",
        "FastMCP CLI command",
        "authentication setup",
        "server initialization",
        "tool registration"
    ]
    
    print(f"Searching collection: {collection_name}")
    print(f"Using embedding method: fastembed")
    
    try:
        # Create Qdrant client
        client = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))
        
        # Create embedding provider
        embedding_provider = FastEmbedProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        all_results = []
        
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
                limit=3,  # Limit to top 3 results per query
                with_payload=True
            )
            
            safe_print(f"Found {len(search_result.points)} results")
            all_results.extend([(query_text, point) for point in search_result.points])
            
            # Display top results
            if search_result.points:
                for i, point in enumerate(search_result.points[:2], 1):  # Show only top 2 results
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
                                safe_print(f"Content: {clean_content[:500]}...")
                            except Exception:
                                # Ultimate fallback
                                safe_print(f"Content: {content[:200]}...")
                        # Print source metadata
                        if 'source' in point.payload:
                            safe_print(f"Source: {point.payload['source']}")
            else:
                safe_print("No results found for the query.")
        
        # Summary
        safe_print(f"\n--- SEARCH SUMMARY ---")
        safe_print(f"Total queries performed: {len(query_texts)}")
        safe_print(f"Total results found: {len(all_results)}")
        
        return True
        
    except Exception as e:
        safe_print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run the search."""
    safe_print("Searching 'manuali' collection for code examples and functions...")
    safe_print("This script searches for specific code-related terms in the MCP documentation.")
    
    try:
        # Run the async search
        success = asyncio.run(search_manuali_collection())
        
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