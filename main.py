#!/usr/bin/env python3
"""
Main entry point for the crawl4AI agent project.
This file demonstrates the basic usage of the project components.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Main function to demonstrate project setup."""
    print("crawl4AI Agent Project")
    print("======================")
    print("Project successfully initialized with uv!")
    print()
    print("Environment variables loaded:")
    print(f"  QDRANT_URL: {os.getenv('QDRANT_URL', 'Not set')}")
    print(f"  QDRANT_API_KEY: {'Set' if os.getenv('QDRANT_API_KEY') else 'Not set'}")
    print(f"  DASHSCOPE_API_KEY: {'Set' if os.getenv('DASHSCOPE_API_KEY') else 'Not set'}")
    print()
    print("Available scripts:")
    print("  - insert_docs_qdrant.py: Crawl and insert documents into Qdrant")
    print("  - retrieve_docs_qdrant.py: Retrieve documents from Qdrant")
    print("  - rag_agent_qdrant.py: Run the RAG agent")
    print("  - test_retrieval.py: Test the retrieval functionality")

if __name__ == "__main__":
    main()