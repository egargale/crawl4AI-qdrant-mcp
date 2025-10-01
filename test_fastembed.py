#!/usr/bin/env python3
"""
Test script for the fastembed implementation in insert_docs_qdrant.py
"""

import os
import sys
import tempfile
from dotenv import load_dotenv

def create_test_markdown_file():
    """Create a simple test markdown file."""
    content = """# Test Document

This is a simple test document to verify that the fastembed implementation works correctly.

## Section 1

This is the first section of the test document. It contains some sample text that we can use to test the embedding generation.

## Section 2

This is the second section. We'll use this to verify that chunking works properly with the fastembed method.

### Subsection

A small subsection to test hierarchical chunking.
"""
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(content)
        return f.name

def test_fastembed():
    """Test the fastembed implementation."""
    # Load environment variables
    load_dotenv()
    
    # Create test markdown file
    test_file = create_test_markdown_file()
    print(f"Created test file: {test_file}")
    
    try:
        # Test with fastembed method
        from insert_docs_qdrant import (
            get_qdrant_client,
            generate_embeddings_fastembed
        )
        
        # Test fastembed embedding generation
        test_texts = [
            "This is a test document for fastembed.",
            "Another test sentence to verify embeddings.",
            "Qdrant integration with fastembed should work."
        ]
        
        print("Testing fastembed embedding generation...")
        embeddings = generate_embeddings_fastembed(test_texts, "BAAI/bge-small-en")
        print(f"Generated {len(embeddings)} embeddings")
        print(f"First embedding dimension: {len(embeddings[0])}")
        print("Fastembed test completed successfully!")
        
        # Clean up
        os.unlink(test_file)
        return True
        
    except Exception as e:
        print(f"Error during fastembed test: {e}")
        # Clean up
        if os.path.exists(test_file):
            os.unlink(test_file)
        return False

if __name__ == "__main__":
    success = test_fastembed()
    if success:
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nTests failed!")
        sys.exit(1)