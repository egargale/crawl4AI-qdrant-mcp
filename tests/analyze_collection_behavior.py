#!/usr/bin/env python3
"""
Script to analyze the behavior of insert_docs_qdrant.py when adding documents to an existing collection.
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

def check_collection_before_and_after():
    """Check the collection state before and after running insert_docs_qdrant.py"""
    
    # Load environment variables
    load_dotenv()
    
    collection_name = "manuali"
    
    try:
        # Create Qdrant client
        client = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))
        
        # Check if collection exists and get info
        try:
            collection_info = client.get_collection(collection_name)
            print(f"Collection '{collection_name}' exists:")
            print(f"  Status: {collection_info.status}")
            print(f"  Vectors count: {collection_info.vectors_count}")
            print(f"  Indexed vectors count: {collection_info.indexed_vectors_count}")
            
            # Check vector configuration
            vectors_config = collection_info.config.params.vectors
            if hasattr(vectors_config, 'size'):
                print(f"  Vector size: {vectors_config.size}")
                print(f"  Vector distance: {vectors_config.distance}")
            else:
                print(f"  Vector config: {vectors_config}")
                
        except Exception as e:
            print(f"Collection '{collection_name}' does not exist or error: {e}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Checking collection state...")
    check_collection_before_and_after()