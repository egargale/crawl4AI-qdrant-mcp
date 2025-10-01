#!/usr/bin/env python3
"""
Script to insert test documents into Qdrant manually.
This avoids the encoding issues with Crawl4AI on Windows.
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from openai import OpenAI

def get_qdrant_client():
    """Initialize and return a Qdrant client."""
    QDRANT_URL = os.getenv('QDRANT_URL')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
    
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in the .env file")
    
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def get_embedding_client():
    """Initialize and return an OpenAI-compatible embedding client."""
    return OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    )

def generate_embeddings(client, texts, model_name, dimensions):
    """Generate embeddings for a list of texts."""
    all_embeddings = []
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(
            model=model_name,
            input=batch,
            dimensions=dimensions
        )
        all_embeddings.extend([data.embedding for data in resp.data])
    return all_embeddings

def insert_test_documents():
    """Insert test documents into Qdrant."""
    # Load environment variables
    load_dotenv()
    
    # Initialize clients
    qdrant_client = get_qdrant_client()
    embedding_client = get_embedding_client()
    
    # Test documents
    documents = [
        "Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data.",
        "Web development with Python can be done using frameworks like Django and Flask.",
        "Data science involves extracting insights from structured and unstructured data using statistical methods.",
        "Cloud computing provides on-demand access to computing resources over the internet."
    ]
    
    # Metadata for each document
    metadatas = [
        {"source": "python_guide.txt", "topic": "programming"},
        {"source": "ml_guide.txt", "topic": "ai"},
        {"source": "web_dev_guide.txt", "topic": "programming"},
        {"source": "data_science_guide.txt", "topic": "data"},
        {"source": "cloud_guide.txt", "topic": "infrastructure"}
    ]
    
    collection_name = "test_docs"
    model_name = "text-embedding-v3"
    embedding_dim = 1024
    
    print(f"Inserting {len(documents)} test documents into Qdrant collection '{collection_name}'...")
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings(embedding_client, documents, model_name, embedding_dim)
    print(f"Generated {len(embeddings)} embeddings with {embedding_dim} dimensions")
    
    # Create collection if it doesn't exist
    try:
        collection_info = qdrant_client.get_collection(collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except:
        print(f"Creating collection '{collection_name}'...")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=embedding_dim, distance=models.Distance.COSINE)
        )
    
    # Upload points with pre-computed embeddings
    points = [
        models.PointStruct(
            id=i,
            vector=embeddings[i],
            payload={
                "document": documents[i],
                "source": metadatas[i]["source"],
                "topic": metadatas[i]["topic"]
            }
        )
        for i in range(len(documents))
    ]
    
    qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )
    
    print(f"Successfully added {len(documents)} documents to Qdrant collection '{collection_name}'.")

if __name__ == "__main__":
    try:
        insert_test_documents()
        print("\nTest documents inserted successfully!")
    except Exception as e:
        print(f"\nError inserting test documents: {e}")
        exit(1)