#!/usr/bin/env python3
"""
Simple test to check what the retrieve function returns.
"""

import os
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

def get_qdrant_client() -> QdrantClient:
    """Initialize and return a Qdrant client."""
    QDRANT_URL = os.getenv('QDRANT_URL')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
    
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in the .env file")
    
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def get_embedding_client() -> OpenAI:
    """Initialize and return an OpenAI-compatible embedding client."""
    return OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    )

def generate_embeddings(client: OpenAI, texts: list, model_name: str, dimensions: int) -> list:
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

def query_collection(
    client: QdrantClient,
    embedding_client: OpenAI,
    collection_name: str,
    model_name: str,
    embedding_dim: int,
    query_text: str,
    n_results: int = 5,
):
    """Query a Qdrant collection for similar documents."""
    # Generate embedding for the query
    query_embedding = generate_embeddings(embedding_client, [query_text], model_name, embedding_dim)[0]
    
    # Query the collection
    search_result = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=n_results,
        with_payload=True
    )
    
    # Print debug information
    print(f"Found {len(search_result.points)} points from Qdrant")
    for i, point in enumerate(search_result.points):
        print(f"Point {i}: ID={point.id}, Score={point.score}")
        print(f"Point {i} Payload keys: {list(point.payload.keys())}")
        print(f"Point {i} Document content: {point.payload.get('document', 'NOT FOUND')}")
        print("---")
    
    # Format results similar to ChromaDB format for compatibility
    return {
        "ids": [[result.id for result in search_result.points]],
        "documents": [[result.payload.get("document", "") for result in search_result.points]],
        "metadatas": [[result.payload for result in search_result.points]],
        "distances": [[1 - result.score for result in search_result.points]]  # Convert score to distance
    }

def format_results_as_context(query_results) -> str:
    """Format query results as a context string for the agent."""
    context = "CONTEXT INFORMATION:\n\n"
    
    print(f"Formatting {len(query_results['documents'][0])} documents as context")
    
    for i, (doc, metadata, distance) in enumerate(zip(
        query_results["documents"][0],
        query_results["metadatas"][0],
        query_results["distances"][0]
    )):
        print(f"Document {i}: Content: {doc[:100]}...")
        # Add document information
        context += f"Document {i+1} (Relevance: {1 - distance:.2f}):\n"
        
        # Add metadata if available
        if metadata:
            for key, value in metadata.items():
                # Skip the document content as it's already included separately
                if key != "document":
                    context += f"{key}: {value}\n"
        
        # Add document content
        context += f"Content: {doc}\n\n"
    
    return context

def main():
    """Main function to test the retrieval."""
    question = "explain function AddAndManageSingleTextDrawingForStudy"
    collection_name = "docs_test"
    embedding_model = "text-embedding-v3"
    embedding_dim = 1024
    n_results = 5
    
    # Initialize clients
    qdrant_client = get_qdrant_client()
    embedding_client = get_embedding_client()
    
    print(f"Testing retrieval for query: '{question}'")
    print(f"Using collection: '{collection_name}'")
    
    # Query the collection
    query_results = query_collection(
        qdrant_client,
        embedding_client,
        collection_name,
        embedding_model,
        embedding_dim,
        question,
        n_results=n_results
    )
    
    # Format the results as context
    context_str = format_results_as_context(query_results)
    print(f"\nFormatted context:\n{context_str}")

if __name__ == "__main__":
    main()