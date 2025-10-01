#!/usr/bin/env python3
"""
Test the existing retrieval functionality with fastembed embeddings
"""

import os
import argparse
from dotenv import load_dotenv

# Try to import fastembed
try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
    print("FastEmbed is available")
except ImportError:
    FASTEMBED_AVAILABLE = False
    print("FastEmbed is not available")

def test_existing_retrieval_with_fastembed(collection_name="test_fastembed_mcp"):
    """Test the existing retrieval functionality with fastembed embeddings."""
    if not FASTEMBED_AVAILABLE:
        print("Cannot test fastembed - not available")
        return False
    
    # Load environment variables
    load_dotenv()
    
    # Import the retrieval functions
    try:
        from retrieve_docs_qdrant import (
            get_qdrant_client,
            retrieve_documents,
            format_results
        )
    except ImportError as e:
        print(f"Error importing retrieval functions: {e}")
        return False
    
    # Initialize clients
    qdrant_client = get_qdrant_client()
    print(f"Connected to Qdrant")
    
    # Initialize the embedding model to get model info
    model_name = "BAAI/bge-small-en"
    embedding_model = TextEmbedding(model_name=model_name)
    
    # Generate a test embedding to get the dimension
    test_embedding = list(embedding_model.embed(["test"]))[0]
    embedding_dim = len(test_embedding)
    print(f"Using model {model_name} with {embedding_dim} dimensions")
    
    # Test queries
    test_queries = [
        "Which integration is best for agents?",
        "How to use fastembed with Qdrant?",
        "MCP server compatibility"
    ]
    
    print(f"\nTesting existing retrieval functionality from collection: {collection_name}")
    print("-" * 70)
    
    for query_text in test_queries:
        print(f"\nQuery: '{query_text}'")
        
        try:
            # Use the existing retrieve_documents function
            results = retrieve_documents(
                qdrant_client=qdrant_client,
                embedding_client=None,  # Not used for fastembed
                collection_name=collection_name,
                model_name=model_name,
                embedding_dim=embedding_dim,
                query_text=query_text,
                n_results=3,
                embedding_method="fastembed"  # Specify fastembed method
            )
            
            # Format and display results
            formatted_output = format_results(results, show_full_content=False)
            print(formatted_output)
                
        except Exception as e:
            print(f"Error during retrieval: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("Existing retrieval functionality test with fastembed completed successfully!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Test existing retrieval functionality with fastembed embeddings")
    parser.add_argument("--collection", default="test_fastembed_mcp", help="Qdrant collection name")
    args = parser.parse_args()
    
    try:
        success = test_existing_retrieval_with_fastembed(args.collection)
        if success:
            print("\nExisting retrieval functionality test completed successfully!")
        else:
            print("\nExisting retrieval functionality test failed!")
            return 1
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())