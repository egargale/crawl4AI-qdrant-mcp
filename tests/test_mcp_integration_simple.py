#!/usr/bin/env python3
"""
Simple integration test for verifying that documents inserted by insert_docs_qdrant.py
can be correctly found by mcp-server-qdrant find functionality.
This version uses FastEmbed for both insertion and retrieval to ensure compatibility.
"""

import os
import sys
import uuid
import asyncio
from dotenv import load_dotenv

# Add the mcp-server-qdrant src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mcp-server-qdrant', 'src'))

from qdrant_client import QdrantClient
from qdrant_client import models

# Import our insert script functions
from insert_docs_qdrant import (
    get_qdrant_client,
    smart_chunk_markdown
)

# Import mcp-server-qdrant components
from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from mcp_server_qdrant.qdrant import QdrantConnector, Entry

# Try to import fastembed, but make it optional
try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False


def get_fastembed_vector_name(model_name: str) -> str:
    """Get the vector name that FastEmbedProvider uses for Qdrant collections."""
    model_name_part = model_name.split("/")[-1].lower()
    return f"fast-{model_name_part}"


def generate_embeddings_fastembed(texts: list[str], model_name: str) -> list[list[float]]:
    """Generate embeddings for a list of texts using fastembed."""
    if not FASTEMBED_AVAILABLE:
        raise ImportError("fastembed is not installed. Please install it with: pip install fastembed")
    
    # Initialize the embedding model
    embedding_model = TextEmbedding(model_name=model_name)
    
    # Generate embeddings
    embeddings = list(embedding_model.embed(texts))
    
    # Convert numpy arrays to lists
    return [embedding.tolist() for embedding in embeddings]


def insert_documents_to_qdrant_fastembed(
    qdrant_client: QdrantClient, 
    collection_name: str, 
    model_name: str,
    ids: list[int], 
    documents: list[str], 
    metadatas: list[dict], 
    batch_size: int = 100
) -> None:
    """Add documents to a Qdrant collection in batches using FastEmbed."""
    # Generate embeddings for all documents
    print(f"Generating embeddings using FastEmbed method...")
    
    embeddings = generate_embeddings_fastembed(documents, model_name)
    print(f"Generated {len(embeddings)} embeddings")
    
    # Get the actual dimension of the fastembed model
    actual_dim = len(embeddings[0]) if embeddings else 384
    print(f"Embeddings have {actual_dim} dimensions")
    
    # Get the vector name that FastEmbedProvider will use
    vector_name = get_fastembed_vector_name(model_name)
    print(f"Using vector name: {vector_name}")
    
    # Check if collection already exists and handle dimension mismatch
    try:
        # Try to get the collection info
        collection_info = qdrant_client.get_collection(collection_name)
        print(f"Collection '{collection_name}' exists")
        
        # Check if the vector configuration matches
        if vector_name not in collection_info.config.params.vectors:
            print(f"Vector configuration mismatch in collection")
            print("Deleting existing collection and recreating with correct configuration...")
            qdrant_client.delete_collection(collection_name=collection_name)
            
            # Recreate collection with correct dimensions
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    vector_name: models.VectorParams(
                        size=actual_dim, distance=models.Distance.COSINE)
                }
            )
        else:
            print(f"Collection '{collection_name}' already exists with correct vector configuration.")
            
    except Exception as e:
        # Collection doesn't exist, so create it
        print(f"Collection '{collection_name}' does not exist. Creating it now.")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                vector_name: models.VectorParams(
                    size=actual_dim, distance=models.Distance.COSINE)
            }
        )
    
    # Upload points with pre-computed embeddings in batches
    print("Inserting documents into Qdrant...")
    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        batch_points = [
            models.PointStruct(
                id=ids[j],
                vector={vector_name: embeddings[j]},
                payload=metadatas[j]
            )
            for j in range(i, end_idx)
        ]
        
        qdrant_client.upsert(
            collection_name=collection_name,
            points=batch_points
        )
        
        print(f"Inserted batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")


def create_test_document() -> str:
    """Create a test document with known content for testing."""
    return """
# Integration Test Document

This document is used to test the integration between insert_docs_qdrant.py 
and mcp-server-qdrant find functionality.

## Unique Test Identifier

TEST_MARKER_INTEGRATION_12345

## Test Content

This is specific content that we will search for to verify the integration works:
- Qdrant retrieval system
- Embedding consistency check
- Cross-system document discovery
"""


async def test_simple_insert_and_find():
    """Simple test that documents inserted by our script can be found by mcp-server-qdrant."""
    
    # Load environment variables
    load_dotenv()
    
    # Use a unique collection name for this test
    collection_name = f"test_integration_simple_{uuid.uuid4().hex[:8]}"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Testing with collection: {collection_name}")
    print(f"Using model: {model_name}")
    
    try:
        # Step 1: Insert test document using our script's functionality with FastEmbed
        print("1. Inserting test document with FastEmbed...")
        qdrant_client = get_qdrant_client()
        
        # Create and chunk test document
        test_doc_content = create_test_document()
        chunks = smart_chunk_markdown(test_doc_content, max_len=1000)
        
        # Prepare documents for insertion
        ids = list(range(len(chunks)))
        documents = chunks
        metadatas = []
        for i, chunk in enumerate(chunks):
            metadatas.append({
                "document": chunk,
                "chunk_index": i,
                "source": "integration_test",
                "test_marker": "TEST_MARKER_INTEGRATION_12345"
            })
        
        # Insert documents using FastEmbed
        insert_documents_to_qdrant_fastembed(
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            model_name=model_name,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            batch_size=100
        )
        print("   [OK] Documents inserted successfully")
        
        # Step 2: Use mcp-server-qdrant to find the documents
        print("2. Searching with mcp-server-qdrant...")
        
        # Create embedding provider with the same model
        embedding_provider = FastEmbedProvider(model_name=model_name)
        
        # Create Qdrant connector
        mcp_connector = QdrantConnector(
            qdrant_url=os.getenv('QDRANT_URL'),
            qdrant_api_key=os.getenv('QDRANT_API_KEY'),
            collection_name=collection_name,
            embedding_provider=embedding_provider
        )
        
        # Search for our test marker
        results = await mcp_connector.search(
            "TEST_MARKER_INTEGRATION_12345", 
            collection_name=collection_name, 
            limit=5
        )
        
        print(f"   Found {len(results)} results")
        
        # Step 3: Verify results
        if len(results) > 0:
            # Check if we found our test content
            found_marker = any("TEST_MARKER_INTEGRATION_12345" in entry.content for entry in results)
            
            if found_marker:
                print("   [SUCCESS] Found test marker in mcp-server-qdrant search results!")
                return True
            else:
                print("   [ERROR] Found results but test marker not in content!")
                for i, entry in enumerate(results):
                    print(f"     Result {i+1}: {entry.content[:100]}...")
                return False
        else:
            print("   [ERROR] No results found!")
            return False
            
    except Exception as e:
        print(f"   [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up: delete the test collection
        try:
            qdrant_client = get_qdrant_client()
            qdrant_client.delete_collection(collection_name=collection_name)
            print(f"   [CLEANUP] Cleaned up test collection: {collection_name}")
        except Exception as e:
            print(f"   [WARNING] Could not delete test collection: {e}")


def main():
    """Main function to run the integration test."""
    print("Running simple integration test between insert_docs_qdrant.py and mcp-server-qdrant...")
    print("Using FastEmbed for both insertion and retrieval to ensure compatibility.")
    
    try:
        # Run the async test
        success = asyncio.run(test_simple_insert_and_find())
        
        if success:
            print("\n[SUCCESS] Integration test PASSED!")
            return 0
        else:
            print("\n[FAILED] Integration test FAILED!")
            return 1
            
    except Exception as e:
        print(f"\n[ERROR] Integration test ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())