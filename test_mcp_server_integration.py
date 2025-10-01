#!/usr/bin/env python3
"""
Integration test for verifying that documents inserted by insert_docs_qdrant.py
can be correctly found by mcp-server-qdrant find functionality.
"""

import os
import sys
import uuid
import asyncio
import argparse
from typing import List
from dotenv import load_dotenv

# Add the mcp-server-qdrant src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mcp-server-qdrant', 'src'))

from qdrant_client import QdrantClient
from openai import OpenAI

# Import our insert script functions
from insert_docs_qdrant import (
    get_qdrant_client,
    get_embedding_client,
    smart_chunk_markdown,
    insert_documents_to_qdrant,
    generate_embeddings
)

# Import mcp-server-qdrant components
from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from mcp_server_qdrant.qdrant import QdrantConnector, Entry


def create_test_document() -> str:
    """Create a test document with known content for testing."""
    return """
# Test Document for Integration Testing

This is a test document to verify that the integration between insert_docs_qdrant.py 
and mcp-server-qdrant works correctly.

## Section 1: Introduction

The purpose of this document is to test the end-to-end workflow of:
1. Inserting documents into Qdrant using insert_docs_qdrant.py functionality
2. Retrieving those documents using mcp-server-qdrant find functionality

## Section 2: Technical Details

This test focuses on ensuring that:
- Documents are properly chunked and stored with metadata
- Embeddings are generated consistently between both systems
- The mcp-server-qdrant can find documents that were inserted by our script

## Section 3: Test Content

Here's some specific content we'll search for:
- Unique test identifier: INTEGRATION_TEST_MARKER_12345
- Technical term: QdrantVectorDatabase
- Concept: semantic search retrieval
"""


async def test_insert_and_find_integration(collection_name: str = "test_integration_collection"):
    """Test that documents inserted by our script can be found by mcp-server-qdrant."""
    
    # Load environment variables
    load_dotenv()
    
    print(f"Running integration test with collection: {collection_name}")
    
    # Step 1: Insert test documents using our script's functionality
    print("1. Setting up Qdrant client and embedding client...")
    qdrant_client = get_qdrant_client()
    embedding_client = get_embedding_client()
    
    # Create test document
    test_doc_content = create_test_document()
    
    # Chunk the document
    print("2. Chunking test document...")
    chunks = smart_chunk_markdown(test_doc_content, max_len=1000)
    print(f"   Created {len(chunks)} chunks")
    
    # Prepare documents for insertion
    ids = list(range(len(chunks)))
    documents = chunks
    metadatas = []
    for i, chunk in enumerate(chunks):
        metadatas.append({
            "document": chunk,
            "chunk_index": i,
            "source": "integration_test",
            "test_id": "integration_test_12345"
        })
    
    # Insert documents
    print("3. Inserting documents into Qdrant...")
    insert_documents_to_qdrant(
        qdrant_client=qdrant_client,
        embedding_client=embedding_client,
        collection_name=collection_name,
        model_name="text-embedding-v3",
        embedding_dim=1024,
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        batch_size=100,
        embedding_method="openai"
    )
    print("   Documents inserted successfully")
    
    # Step 2: Use mcp-server-qdrant to find the documents
    print("4. Setting up mcp-server-qdrant connector...")
    
    # Create embedding provider (using the same model as in settings)
    embedding_provider = FastEmbedProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create Qdrant connector
    mcp_connector = QdrantConnector(
        qdrant_url=os.getenv('QDRANT_URL'),
        qdrant_api_key=os.getenv('QDRANT_API_KEY'),
        collection_name=collection_name,
        embedding_provider=embedding_provider
    )
    
    # Step 3: Search for documents using mcp-server-qdrant
    print("5. Searching for documents using mcp-server-qdrant...")
    
    # Search for content that should be in our test document
    search_queries = [
        "INTEGRATION_TEST_MARKER_12345",
        "QdrantVectorDatabase",
        "semantic search retrieval",
        "integration between insert_docs_qdrant.py and mcp-server-qdrant"
    ]
    
    all_results = []
    for query in search_queries:
        print(f"   Searching for: '{query}'")
        results = await mcp_connector.search(query, collection_name=collection_name, limit=5)
        all_results.extend(results)
        print(f"   Found {len(results)} results")
    
    # Step 4: Verify results
    print("6. Verifying results...")
    
    if not all_results:
        print("   ERROR: No results found!")
        return False
    
    # Check if we found content from our test document
    found_test_content = False
    found_marker = False
    
    for entry in all_results:
        if "INTEGRATION_TEST_MARKER_12345" in entry.content:
            found_marker = True
        if "integration between insert_docs_qdrant.py and mcp-server-qdrant" in entry.content:
            found_test_content = True
    
    if found_marker and found_test_content:
        print("   SUCCESS: Found test content in mcp-server-qdrant search results!")
        print(f"   Total unique results: {len(set(r.content for r in all_results))}")
        return True
    else:
        print("   ERROR: Could not find expected test content!")
        print(f"   Found marker: {found_marker}")
        print(f"   Found test content: {found_test_content}")
        return False


def main():
    """Main function to run the integration test."""
    parser = argparse.ArgumentParser(description="Test integration between insert_docs_qdrant.py and mcp-server-qdrant")
    parser.add_argument("--collection", default=f"test_integration_{uuid.uuid4().hex[:8]}", 
                       help="Qdrant collection name to use for testing")
    
    args = parser.parse_args()
    
    try:
        # Run the async test
        success = asyncio.run(test_insert_and_find_integration(args.collection))
        
        if success:
            print("\nIntegration test PASSED!")
            return 0
        else:
            print("\nIntegration test FAILED!")
            return 1
            
    except Exception as e:
        print(f"\nIntegration test ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())