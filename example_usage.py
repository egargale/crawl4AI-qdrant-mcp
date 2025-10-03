#!/usr/bin/env python3
"""
Example script demonstrating the basic usage of the crawl4AI-agent-v2 project.
This script shows how to crawl a website, process the content, and query it.
"""

import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def example_website_crawling():
    """Example of crawling a website."""
    print("Example 1: Website Crawling")
    print("-" * 30)
    print("To crawl a website, run:")
    print("python website_downloader.py https://example.com -o example_content --max-depth 2")
    print()
    print("This will:")
    print("1. Crawl https://example.com and up to 2 levels of links")
    print("2. Extract content using LLM-based processing")
    print("3. Save content in both markdown and JSONL formats")
    print("4. Store files in the 'example_content' directory")
    print()

def example_rag_setup():
    """Example of setting up RAG."""
    print("Example 2: RAG Setup")
    print("-" * 30)
    print("To process and store crawled content, run:")
    print("python rag_setup.py example_content --collection example_docs")
    print()
    print("This will:")
    print("1. Load all markdown/JSONL files from 'example_content'")
    print("2. Split documents into chunks for better retrieval")
    print("3. Generate embeddings using DashScope")
    print("4. Store documents in Qdrant collection 'example_docs'")
    print()

def example_rag_query():
    """Example of querying with RAG."""
    print("Example 3: RAG Query")
    print("-" * 30)
    print("To ask questions about your content, run:")
    print("python rag_query.py \"What is this website about?\" --collection example_docs")
    print()
    print("This will:")
    print("1. Generate embedding for your question")
    print("2. Search for relevant documents in Qdrant")
    print("3. Use Qwen LLM to answer based on retrieved content")
    print("4. Return the answer with source information")
    print()

def example_rag_agent():
    """Example of using the RAG agent."""
    print("Example 4: RAG Agent")
    print("-" * 30)
    print("To use the Pydantic AI agent, run:")
    print("python rag_agent_qdrant.py --question \"What are the main features?\" --collection example_docs")
    print()
    print("This will:")
    print("1. Use a custom AI agent with retrieval tools")
    print("2. Automatically retrieve relevant documents")
    print("3. Provide detailed answers with source references")
    print()

def example_document_retrieval():
    """Example of simple document retrieval."""
    print("Example 5: Document Retrieval")
    print("-" * 30)
    print("To perform semantic search on documents, run:")
    print("python retrieve_docs_qdrant.py \"search query\" --collection example_docs")
    print()
    print("This will:")
    print("1. Perform semantic search using DashScope embeddings")
    print("2. Return the most relevant documents")
    print("3. Show content previews with relevance scores")
    print()

def main():
    """Main function to show all examples."""
    print("crawl4AI-agent-v2 Usage Examples")
    print("=" * 40)
    print()
    
    # Show all examples
    example_website_crawling()
    example_rag_setup()
    example_rag_query()
    example_rag_agent()
    example_document_retrieval()
    
    print("Next Steps:")
    print("-" * 30)
    print("1. Create a .env file with your API keys")
    print("2. Run the test_setup.py script to verify your setup")
    print("3. Try the examples above with your own website")
    print("4. Check the README.md for detailed documentation")

if __name__ == "__main__":
    main()