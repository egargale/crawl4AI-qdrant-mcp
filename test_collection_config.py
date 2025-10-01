#!/usr/bin/env python3
"""
Test script to verify fastembed integration with Qdrant collections
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

# Try to import fastembed
try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
    print("FastEmbed is available")
except ImportError:
    FASTEMBED_AVAILABLE = False
    print("FastEmbed is not available")

def test_collection_configuration():
    """Test Qdrant collection configuration for fastembed compatibility."""
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
    
    # Test with a simple collection
    collection_name = "test_fastembed_config"
    
    # Delete collection if it exists
    try:
        client.delete_collection(collection_name=collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except:
        pass
    
    # Initialize the embedding model to get the correct dimension
    model_name = "BAAI/bge-small-en"
    embedding_model = TextEmbedding(model_name=model_name)
    
    # Generate a test embedding to get the dimension
    test_embedding = list(embedding_model.embed(["test"]))[0]
    dimension = len(test_embedding)
    print(f"Model {model_name} produces embeddings with {dimension} dimensions")
    
    # Create collection with correct configuration
    print(f"Creating collection {collection_name} with {dimension}-dimensional vectors")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=dimension, 
            distance=models.Distance.COSINE
        ),
    )
    
    # Insert a test document
    test_docs = ["This is a test document for fastembed integration"]
    embeddings = list(embedding_model.embed(test_docs))
    
    client.upload_points(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=1,
                vector=embeddings[0].tolist(),
                payload={"document": test_docs[0], "source": "test"}
            )
        ]
    )
    
    print("Test document inserted successfully!")
    
    # Test search
    query_embedding = list(embedding_model.embed(["test document"]))[0]
    search_result = client.query_points(
        collection_name=collection_name,
        query=query_embedding.tolist(),
        limit=1
    ).points
    
    if search_result:
        print("Search test successful!")
        print(f"Found result with score: {search_result[0].score}")
    else:
        print("Search returned no results")
    
    # Clean up
    client.delete_collection(collection_name=collection_name)
    print(f"Cleaned up collection {collection_name}")
    
    print("Collection configuration test completed successfully!")
    return True

if __name__ == "__main__":
    try:
        success = test_collection_configuration()
        if success:
            print("\nTest completed successfully!")
        else:
            print("\nTest failed!")
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()