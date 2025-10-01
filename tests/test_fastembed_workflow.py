#!/usr/bin/env python3
"""
Comprehensive test of the fastembed workflow including insertion and retrieval
"""

import os
import sys
import tempfile
from dotenv import load_dotenv

def create_test_documents():
    """Create test documents for fastembed workflow testing."""
    documents = [
        """# FastEmbed Integration Guide

FastEmbed is a lightweight, efficient library for generating embeddings. It's designed to work seamlessly with Qdrant for vector search applications.

## Key Features

- Fast inference using ONNX Runtime
- Quantized model weights for reduced memory usage
- Support for multiple embedding models
- Easy integration with Qdrant

## Installation

```bash
pip install "qdrant-client[fastembed]>=1.14.1"
```

## Usage with Qdrant

FastEmbed integrates directly with Qdrant client, making it easy to generate and store embeddings.""",
        
        """# MCP Server Compatibility

MCP (Model Controller Protocol) servers can leverage fastembed for efficient vector search operations. This compatibility ensures that MCP-based applications can take advantage of fastembed's performance benefits.

## Benefits

- Reduced latency for embedding generation
- Lower resource consumption
- Better integration with local development environments
- Compatible with various MCP tools and extensions

## Supported Models

- BAAI/bge-small-en (384 dimensions)
- BAAI/bge-base-en (768 dimensions)
- sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)""",
        
        """# OAuth Provider Integration

FastEmbed can work alongside OAuth providers in MCP server environments. This integration allows for secure, authenticated access to embedding services while maintaining the performance benefits of local embedding generation.

## Security Considerations

- No external API calls required for embedding generation
- Reduced attack surface compared to cloud-based solutions
- Easy to audit and monitor embedding operations"""
    ]
    
    return documents

def test_fastembed_workflow():
    """Test the complete fastembed workflow: insertion and retrieval."""
    # Load environment variables
    load_dotenv()
    
    print("Testing fastembed workflow...")
    
    # Create test documents
    test_docs = create_test_documents()
    
    # Create temporary files
    temp_files = []
    for i, doc in enumerate(test_docs):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(doc)
            temp_files.append(f.name)
            print(f"Created test document {i+1}: {f.name}")
    
    try:
        # Test insertion using fastembed
        from insert_docs_qdrant import (
            get_qdrant_client,
            get_embedding_client,
            insert_documents_to_qdrant,
            smart_chunk_markdown,
            extract_section_info
        )
        
        # Initialize clients
        qdrant_client = get_qdrant_client()
        embedding_client = None  # Not used for fastembed
        
        # Prepare documents for insertion
        ids, documents, metadatas = [], [], []
        chunk_idx = 0
        
        for i, temp_file in enumerate(temp_files):
            with open(temp_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Chunk the document
            chunks = smart_chunk_markdown(content, max_len=500)
            for chunk in chunks:
                ids.append(chunk_idx)
                documents.append(chunk)
                meta = extract_section_info(chunk)
                meta["document"] = chunk
                meta["chunk_index"] = chunk_idx
                meta["source"] = f"test_doc_{i+1}.md"
                metadatas.append(meta)
                chunk_idx += 1
        
        print(f"\nPrepared {len(documents)} chunks for insertion")
        
        # Insert documents using fastembed
        collection_name = "test_fastembed_workflow"
        model_name = "BAAI/bge-small-en"
        
        print(f"Inserting documents into collection '{collection_name}' using fastembed...")
        insert_documents_to_qdrant(
            qdrant_client=qdrant_client,
            embedding_client=embedding_client,
            collection_name=collection_name,
            model_name=model_name,
            embedding_dim=384,  # Will be updated by the function
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embedding_method="fastembed",
            fastembed_model=model_name
        )
        
        print("Documents inserted successfully!")
        
        # Test retrieval
        from retrieve_docs_qdrant import (
            retrieve_documents,
            format_results
        )
        
        test_queries = [
            "How to use fastembed with Qdrant?",
            "MCP server compatibility with fastembed",
            "OAuth provider integration"
        ]
        
        print(f"\nTesting retrieval from collection '{collection_name}':")
        print("-" * 50)
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            
            # Retrieve documents using fastembed
            results = retrieve_documents(
                qdrant_client=qdrant_client,
                embedding_client=None,
                collection_name=collection_name,
                model_name=model_name,
                embedding_dim=384,
                query_text=query,
                n_results=3,
                embedding_method="fastembed"
            )
            
            # Format and display results
            formatted_output = format_results(results, show_full_content=False)
            print(formatted_output)
        
        print("FastEmbed workflow test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during fastembed workflow test: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        # Clean up test collection
        try:
            from insert_docs_qdrant import get_qdrant_client
            qdrant_client = get_qdrant_client()
            qdrant_client.delete_collection(collection_name="test_fastembed_workflow")
            print(f"Cleaned up test collection 'test_fastembed_workflow'")
        except:
            pass

if __name__ == "__main__":
    success = test_fastembed_workflow()
    if success:
        print("\nFastEmbed workflow test completed successfully!")
        sys.exit(0)
    else:
        print("\nFastEmbed workflow test failed!")
        sys.exit(1)