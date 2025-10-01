#!/usr/bin/env python3
"""
Exploratory test to understand the content of the 'manuali' collection.
"""

import os
import sys
from dotenv import load_dotenv

# Add the mcp-server-qdrant src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mcp-server-qdrant', 'src'))

from qdrant_client import QdrantClient
import asyncio


async def explore_manuali_collection():
    """Explore the content of the 'manuali' collection."""
    
    # Load environment variables
    load_dotenv()
    
    collection_name = "manuali"
    
    print(f"Exploring collection: {collection_name}")
    
    try:
        # Create Qdrant client
        client = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))
        
        # Get collection info
        collection_info = client.get_collection(collection_name)
        print(f"Collection status: {collection_info.status}")
        print(f"Vector count: {collection_info.vectors_count}")
        print(f"Indexed vectors: {collection_info.indexed_vectors_count}")
        
        # Get a few sample points to understand the content
        print("\nGetting sample points...")
        search_result = client.query_points(
            collection_name=collection_name,
            query_filter=None,  # Get any points
            limit=5,
            with_payload=True
        )
        
        print(f"Retrieved {len(search_result.points)} sample points:")
        
        # Display sample results
        for i, point in enumerate(search_result.points, 1):
            print(f"\n--- Sample {i} ---")
            print(f"ID: {point.id}")
            if hasattr(point, 'payload') and point.payload:
                # Print document content if available
                if 'document' in point.payload:
                    content = str(point.payload['document'])
                    print(f"Content preview: {content[:300]}...")
                # Print other payload data
                other_payload = {k: v for k, v in point.payload.items() if k != 'document'}
                if other_payload:
                    print(f"Metadata: {other_payload}")
            print()
            
        # Try to search for content that might contain "fastmcp" or "oauth"
        print("Searching for content containing 'fastmcp' or 'oauth'...")
        
        # Since we can't do text search directly, let's try to get more points
        # and manually check their content
        more_results = client.query_points(
            collection_name=collection_name,
            query_filter=None,
            limit=20,
            with_payload=True
        )
        
        fastmcp_matches = []
        oauth_matches = []
        
        for point in more_results.points:
            if hasattr(point, 'payload') and 'document' in point.payload:
                content = str(point.payload['document']).lower()
                if 'fastmcp' in content:
                    fastmcp_matches.append(point)
                if 'oauth' in content:
                    oauth_matches.append(point)
        
        print(f"\nFound {len(fastmcp_matches)} points containing 'fastmcp'")
        print(f"Found {len(oauth_matches)} points containing 'oauth'")
        
        if fastmcp_matches:
            print("\n--- FastMCP Matches ---")
            for i, point in enumerate(fastmcp_matches[:3], 1):  # Show first 3
                print(f"\nFastMCP Match {i}:")
                print(f"ID: {point.id}")
                content = str(point.payload['document'])
                print(f"Content: {content[:500]}...")
                
        if oauth_matches:
            print("\n--- OAuth Matches ---")
            for i, point in enumerate(oauth_matches[:3], 1):  # Show first 3
                print(f"\nOAuth Match {i}:")
                print(f"ID: {point.id}")
                content = str(point.payload['document'])
                print(f"Content: {content[:500]}...")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run the exploration."""
    print("Running exploration of 'manuali' collection...")
    
    try:
        # Run the async test
        success = asyncio.run(explore_manuali_collection())
        
        if success:
            print("\nExploration completed!")
            return 0
        else:
            print("\nExploration failed!")
            return 1
            
    except Exception as e:
        print(f"\nExploration error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())