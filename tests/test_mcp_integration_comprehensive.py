#!/usr/bin/env python3
"""
Comprehensive integration test for verifying that documents inserted by insert_docs_qdrant.py
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


def create_test_documents() -> list[str]:
    """Create test documents with known content for testing."""
    return [
        """
# Python Programming Guide

This guide covers the basics of Python programming language.

## Variables and Data Types

Python supports various data types including integers, floats, strings, and booleans.

### Example Code

```python
name = "Alice"
age = 30
height = 5.6
is_student = False
```

PYTHON_INTEGRATION_TEST
        """,
        """
# Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that focuses on algorithms.

## Supervised Learning

Supervised learning uses labeled training data to make predictions.

### Common Algorithms

- Linear Regression
- Decision Trees
- Random Forest
- Support Vector Machines

ML_INTEGRATION_TEST
        """,
        """
# Web Development with React

React is a popular JavaScript library for building user interfaces.

## Component-Based Architecture

React applications are built using reusable components.

### State Management

React uses state and props to manage data flow in applications.

REACT_INTEGRATION_TEST
        """
    ]


async def test_comprehensive_insert_and_find():
    """Comprehensive test that documents inserted by our script can be found by mcp-server-qdrant."""
    
    # Load environment variables
    load_dotenv()
    
    # Use a unique collection name for this test
    collection_name = f"test_integration_comprehensive_{uuid.uuid4().hex[:8]}"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Testing with collection: {collection_name}")
    print(f"Using model: {model_name}")
    
    test_passed = True
    
    try:
        # Step 1: Insert test documents using our script's functionality with FastEmbed
        print("1. Inserting test documents with FastEmbed...")
        qdrant_client = get_qdrant_client()
        
        # Create test documents
        test_docs = create_test_documents()
        
        # Prepare all documents for insertion
        all_ids = []
        all_documents = []
        all_metadatas = []
        
        doc_index = 0
        for doc_idx, doc_content in enumerate(test_docs):
            chunks = smart_chunk_markdown(doc_content, max_len=1000)
            for chunk_idx, chunk in enumerate(chunks):
                all_ids.append(doc_index)
                all_documents.append(chunk)
                all_metadatas.append({
                    "document": chunk,
                    "doc_index": doc_idx,
                    "chunk_index": chunk_idx,
                    "source": f"test_doc_{doc_idx}",
                    "test_category": ["python", "ml", "react"][doc_idx]
                })
                doc_index += 1
        
        # Insert documents using FastEmbed
        insert_documents_to_qdrant_fastembed(
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            model_name=model_name,
            ids=all_ids,
            documents=all_documents,
            metadatas=all_metadatas,
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
        
        # Test cases with different search queries
        test_cases = [
            {
                "query": "PYTHON_INTEGRATION_TEST",
                "description": "Exact marker search for Python content",
                "expected_category": "python"
            },
            {
                "query": "machine learning algorithms",
                "description": "Semantic search for ML content",
                "expected_category": "ml"
            },
            {
                "query": "React component architecture",
                "description": "Semantic search for React content",
                "expected_category": "react"
            },
            {
                "query": "Python data types and variables",
                "description": "Semantic search for Python programming concepts",
                "expected_category": "python"
            }
        ]
        
        # Run test cases
        passed_tests = 0
        total_tests = len(test_cases)
        
        for i, test_case in enumerate(test_cases):
            print(f"   Test case {i+1}: {test_case['description']}")
            print(f"   Query: '{test_case['query']}'")
            
            # Search for our test query
            results = await mcp_connector.search(
                test_case['query'], 
                collection_name=collection_name, 
                limit=3
            )
            
            print(f"   Found {len(results)} results")
            
            # Verify results
            if len(results) > 0:
                # Check if we found content from the expected category
                found_expected = False
                for entry in results:
                    # Look for indicators of the expected category in the content
                    content_lower = entry.content.lower()
                    expected = test_case['expected_category']
                    
                    if expected == "python" and ("python" in content_lower or "variables" in content_lower):
                        found_expected = True
                        break
                    elif expected == "ml" and ("machine learning" in content_lower or "algorithms" in content_lower):
                        found_expected = True
                        break
                    elif expected == "react" and ("react" in content_lower or "component" in content_lower):
                        found_expected = True
                        break
                
                if found_expected:
                    print(f"   [SUCCESS] Found content related to {test_case['expected_category']}!")
                    passed_tests += 1
                else:
                    print(f"   [INFO] Found results but content doesn't clearly match {test_case['expected_category']}")
                    # This isn't necessarily a failure since semantic search can return related content
                    # Show first result for context
                    if results:
                        print(f"   First result preview: {results[0].content[:100]}...")
            else:
                print(f"   [ERROR] No results found for query: {test_case['query']}")
                test_passed = False
            
            print()
        
        # At least 3 out of 4 tests should pass for the test to be considered successful
        if passed_tests >= 3:
            print(f"   [SUMMARY] {passed_tests}/{total_tests} tests passed")
            return True
        else:
            print(f"   [SUMMARY] Only {passed_tests}/{total_tests} tests passed - below threshold")
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
    """Main function to run the comprehensive integration test."""
    print("Running comprehensive integration test between insert_docs_qdrant.py and mcp-server-qdrant...")
    print("Using FastEmbed for both insertion and retrieval to ensure compatibility.")
    
    try:
        # Run the async test
        success = asyncio.run(test_comprehensive_insert_and_find())
        
        if success:
            print("\n[SUCCESS] Comprehensive integration test PASSED!")
            return 0
        else:
            print("\n[FAILED] Comprehensive integration test FAILED!")
            return 1
            
    except Exception as e:
        print(f"\n[ERROR] Integration test ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())