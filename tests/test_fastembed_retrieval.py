#!/usr/bin/env python3
"""
Test retrieval functionality with fastembed embeddings
"""

import os
import argparse
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Try to import fastembed
try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
    print("FastEmbed is available")
except ImportError:
    FASTEMBED_AVAILABLE = False
    print("FastEmbed is not available")

def test_fastembed_retrieval(collection_name="test_fastembed_mcp"):
    """Test retrieval with fastembed embeddings."""
    if not FASTEMBED_AVAILABLE:
        print("Cannot test fastembed - not available")
        return False
    
    # Load environment variables
    load_dotenv()
    
    # Initialize Qdrant client
    QDRANT_URL = os.getenv('QDRANT_URL')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
    
    if not QDRANT_URL or not QDRANT_API_KEY:
        print("QDRANT_URL and QDRANT_API_KEY must be set in the .env file")
        return False
    
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    print(f"Connected to Qdrant at: {QDRANT_URL}")
    
    # Initialize the embedding model
    model_name = "BAAI/bge-small-en"
    embedding_model = TextEmbedding(model_name=model_name)
    
    # Test queries
    test_queries = [
        "Which integration is best for agents?",
        "How to use fastembed with Qdrant?",
        "MCP server compatibility"
    ]
    
    print(f"\nTesting retrieval from collection: {collection_name}")
    print("-" * 50)
    
    for query_text in test_queries:
        print(f"\nQuery: '{query_text}'")
        
        try:
            # Generate query embedding
            query_embedding = list(embedding_model.embed([query_text]))[0]
            
            # Perform search
            search_result = client.query_points(
                collection_name=collection_name,
                query=query_embedding.tolist(),
                limit=3,
                with_payload=True
            ).points
            
            print(f"Found {len(search_result)} results:")
            for i, result in enumerate(search_result):
                print(f"  {i+1}. Score: {result.score:.4f}")
                if result.payload:
                    doc_content = result.payload.get('document', 'No document content')
                    source = result.payload.get('source', 'Unknown source')
                    print(f"     Document: {doc_content[:100]}{'...' if len(doc_content) > 100 else ''}")
                    print(f"     Source: {source}")
                print()
                
        except Exception as e:
            print(f"Error during search: {e}")
            return False
    
    print("FastEmbed retrieval test completed successfully!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Test retrieval with fastembed embeddings")
    parser.add_argument("--collection", default="test_fastembed_mcp", help="Qdrant collection name")
    args = parser.parse_args()
    
    try:
        success = test_fastembed_retrieval(args.collection)
        if success:
            print("\nRetrieval test completed successfully!")
        else:
            print("\nRetrieval test failed!")
            return 1
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())