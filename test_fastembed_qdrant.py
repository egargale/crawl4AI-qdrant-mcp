#!/usr/bin/env python3
"""
Direct test of fastembed with Qdrant to verify compatibility with MCP servers
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

def test_fastembed_with_qdrant():
    """Test fastembed integration with Qdrant."""
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
    
    # Test documents
    docs = [
        "Qdrant has a LangChain integration for chatbots.",
        "Qdrant has a LlamaIndex integration for agents.",
        "FastEmbed is a lightweight library for generating embeddings.",
        "MCP servers can use fastembed for efficient vector search."
    ]
    
    # Initialize the embedding model
    model_name = "BAAI/bge-small-en"
    embedding_model = TextEmbedding(model_name=model_name)
    
    # Generate embeddings
    print(f"Generating embeddings using model: {model_name}")
    embeddings = list(embedding_model.embed(docs))
    print(f"Generated {len(embeddings)} embeddings with {len(embeddings[0])} dimensions")
    
    # Create collection
    collection_name = "test_fastembed_mcp"
    print(f"Creating collection: {collection_name}")
    
    # Delete collection if it exists
    try:
        client.delete_collection(collection_name=collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except:
        pass
    
    # Create new collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=len(embeddings[0]), 
            distance=models.Distance.COSINE
        ),
    )
    
    # Prepare metadata
    metadata = [
        {"source": "langchain-docs", "document": docs[0]},
        {"source": "llamaindex-docs", "document": docs[1]},
        {"source": "fastembed-docs", "document": docs[2]},
        {"source": "mcp-docs", "document": docs[3]},
    ]
    
    # Insert documents
    print("Inserting documents into Qdrant...")
    client.upload_points(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload=metadata[i]
            )
            for i, embedding in enumerate(embeddings)
        ]
    )
    
    print("Documents inserted successfully!")
    
    # Test search
    print("Testing search functionality...")
    query_text = "Which integration is best for agents?"
    query_embedding = list(embedding_model.embed([query_text]))[0]
    
    search_result = client.query_points(
        collection_name=collection_name,
        query=query_embedding.tolist(),
        limit=3
    ).points
    
    print(f"Found {len(search_result)} results:")
    for i, result in enumerate(search_result):
        print(f"  {i+1}. Score: {result.score:.4f}")
        print(f"     Document: {result.payload.get('document', '')}")
        print(f"     Source: {result.payload.get('source', '')}")
    
    print("FastEmbed with Qdrant test completed successfully!")
    return True

if __name__ == "__main__":
    try:
        success = test_fastembed_with_qdrant()
        if success:
            print("\nTest completed successfully!")
        else:
            print("\nTest failed!")
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()