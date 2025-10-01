#!/usr/bin/env python3
"""
Script to check the configuration of the 'manuali' collection.
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

def check_manuali_collection():
    """Check the configuration of the 'manuali' collection."""
    
    # Load environment variables
    load_dotenv()
    
    try:
        # Create Qdrant client
        client = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))
        
        # Get collection info
        collection_info = client.get_collection("manuali")
        
        print("Collection 'manuali' configuration:")
        print(f"  Status: {collection_info.status}")
        print(f"  Vectors count: {collection_info.vectors_count}")
        print(f"  Indexed vectors count: {collection_info.indexed_vectors_count}")
        
        # Check vector configuration
        vectors_config = collection_info.config.params.vectors
        print(f"  Vector configuration type: {type(vectors_config)}")
        
        if hasattr(vectors_config, 'keys'):
            print(f"  Available vector names: {list(vectors_config.keys())}")
        elif hasattr(vectors_config, 'size'):
            print(f"  Vector size: {vectors_config.size}")
            print(f"  Vector distance: {vectors_config.distance}")
        else:
            print(f"  Vector config: {vectors_config}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_manuali_collection()