#!/usr/bin/env python3
"""
Test script to verify the setup of the crawl4AI-agent-v2 project.
This script checks if all required dependencies are installed and APIs are accessible.
"""

import os
import sys
from dotenv import load_dotenv

def test_environment_variables():
    """Test if required environment variables are set."""
    print("Testing environment variables...")
    
    load_dotenv()
    
    required_vars = ["DASHSCOPE_API_KEY", "QDRANT_URL"]
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
            print(f"  ❌ {var}: Not set")
        else:
            # Show first and last 4 characters for API keys, full value for URLs
            if "KEY" in var:
                masked_value = f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "****"
                print(f"  ✓ {var}: {masked_value}")
            else:
                print(f"  ✓ {var}: {value}")
    
    if missing_vars:
        print(f"  ❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    else:
        print("  ✓ All required environment variables are set")
        return True

def test_imports():
    """Test if all required packages can be imported."""
    print("\nTesting imports...")
    
    required_packages = [
        "crawl4ai",
        "qdrant_client",
        "langchain",
        "langchain_qdrant",
        "openai",
        "pydantic_ai",
        "playwright"
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError as e:
            failed_imports.append(package)
            print(f"  ❌ {package}: {e}")
    
    if failed_imports:
        print(f"  ❌ Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("  ✓ All required packages imported successfully")
        return True

def test_qdrant_connection():
    """Test connection to Qdrant."""
    print("\nTesting Qdrant connection...")
    
    try:
        from qdrant_client import QdrantClient
        
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not qdrant_url:
            print("  ❌ QDRANT_URL not set")
            return False
            
        # Initialize Qdrant client
        if qdrant_api_key:
            client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            client = QdrantClient(url=qdrant_url)
            
        # Test connection
        collections = client.get_collections()
        print(f"  ✓ Connected to Qdrant at {qdrant_url}")
        print(f"  ✓ Collections: {len(collections.collections)} found")
        return True
    except Exception as e:
        print(f"  ❌ Failed to connect to Qdrant: {e}")
        return False

def test_dashscope_api():
    """Test DashScope API access."""
    print("\nTesting DashScope API...")
    
    try:
        from openai import OpenAI
        
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            print("  ❌ DASHSCOPE_API_KEY not set")
            return False
            
        # Initialize client with DashScope endpoint
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        
        # Test embedding
        response = client.embeddings.create(
            model="text-embedding-v4",
            input=["Hello, world!"]
        )
        embedding_dim = len(response.data[0].embedding)
        print(f"  ✓ Embedding API working, dimension: {embedding_dim}")
        
        # Test language model
        completion = client.chat.completions.create(
            model="qwen-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"}
            ],
            max_tokens=50
        )
        print(f"  ✓ Language model API working")
        print(f"  ✓ Sample response: {completion.choices[0].message.content[:50]}...")
        return True
    except Exception as e:
        print(f"  ❌ Failed to access DashScope API: {e}")
        return False

def main():
    """Main function to run all tests."""
    print("crawl4AI-agent-v2 Setup Test")
    print("=" * 40)
    
    tests = [
        test_environment_variables,
        test_imports,
        test_qdrant_connection,
        test_dashscope_api
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 40)
    if all(results):
        print("🎉 All tests passed! Your setup is ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above and fix them.")
        return 1

if __name__ == "__main__":
    sys.exit(main())