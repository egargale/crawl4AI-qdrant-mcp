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
    print("Project successfully initialized!")
    print()
    print("Environment variables loaded:")
    print(f"  QDRANT_URL: {os.getenv('QDRANT_URL', 'Not set')}")
    print(f"  QDRANT_API_KEY: {'Set' if os.getenv('QDRANT_API_KEY') else 'Not set'}")
    print(f"  DASHSCOPE_API_KEY: {'Set' if os.getenv('DASHSCOPE_API_KEY') else 'Not set'}")
    print()
    print("Core scripts:")
    print("  - website_downloader.py: Crawl websites and extract content")
    print("  - rag_setup.py: Process and store content in Qdrant")
    print("  - rag_query.py: Query content with RAG")
    print("  - rag_agent_qdrant.py: Run the RAG agent")
    print("  - retrieve_docs_qdrant.py: Retrieve documents from Qdrant")
    print()
    print("Documentation:")
    print("  - README.md: Complete project documentation")
    print("  - GETTING_STARTED.md: Step-by-step setup guide")
    print("  - QDRANT_SETUP.md: Qdrant configuration guide")
    print("  - DASHSCOPE_SETUP.md: DashScope configuration guide")
    print("  - example_usage.py: Usage examples")
    print("  - test_setup.py: Setup verification script")

if __name__ == "__main__":
    main()