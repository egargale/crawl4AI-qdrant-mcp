#!/usr/bin/env python3
"""
Optimized RAG Query Tool for Qdrant and DashScope.

This script provides functionality to query an existing Qdrant collection
that was created by rag_setup_optimized.py, fully leveraging DashScope's
semantic search optimization capabilities.
"""

import os
import sys
import argparse
from typing import List, Tuple
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.embeddings import Embeddings
import requests
import numpy as np


class OptimizedDashScopeEmbeddings(Embeddings):
    """
    Custom embeddings class that fully leverages DashScope's semantic search optimization.
    
    This implementation uses the text_type parameter to optimize embeddings for either
    queries or documents, which significantly improves search performance.
    """
    
    def __init__(self, api_key: str, model: str = "text-embedding-v4"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        self.batch_size = 10  # DashScope has a limit of 10 texts per request
    
    def _get_embeddings(self, texts: List[str], text_type: str = "document") -> List[List[float]]:
        """
        Internal method to get embeddings with specified text_type.
        
        Args:
            texts: List of texts to embed
            text_type: 'query' for search queries, 'document' for document text
            
        Returns:
            List of embeddings
        """
        url = f"{self.base_url}/embeddings"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Handle both single text and list of texts
        input_texts = texts if isinstance(texts, list) else [texts]
        
        # Process in batches to respect API limits
        all_embeddings = []
        for i in range(0, len(input_texts), self.batch_size):
            batch = input_texts[i:i+self.batch_size]
            
            data = {
                "model": self.model,
                "input": batch,
                "text_type": text_type  # This is the key optimization parameter
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            if response.status_code != 200:
                raise ValueError(f"API request failed: {response.status_code} - {response.text}")
            
            result = response.json()
            
            # Extract embeddings from response
            embeddings = [item['embedding'] for item in result['data']]
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents with document-optimized embeddings.
        
        Args:
            texts: List of document texts to embed
            
        Returns:
            List of document embeddings
        """
        return self._get_embeddings(texts, text_type="document")
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query with query-optimized embeddings.
        
        Args:
            text: Query text to embed
            
        Returns:
            Query embedding
        """
        embeddings = self._get_embeddings([text], text_type="query")
        return embeddings[0]


def setup_optimized_query_chain(collection_name: str = "website_docs", 
                               qdrant_url: str = "http://localhost:6333", 
                               qdrant_api_key: str = None) -> RetrievalQA:
    """
    Set up the optimized query chain for an existing Qdrant collection.
    
    Args:
        collection_name: Name of the Qdrant collection to query
        qdrant_url: URL of the Qdrant instance
        qdrant_api_key: API key for Qdrant (optional for local instances)
    
    Returns:
        RetrievalQA chain for querying
    """
    # Load environment variables
    load_dotenv()
    
    # Get DashScope API key
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    if not dashscope_api_key:
        raise ValueError("DASHSCOPE_API_KEY not found in environment variables")
    
    print(f"[+] Using Qdrant collection: {collection_name}")
    print(f"[+] Qdrant URL: {qdrant_url}")
    
    # Initialize Qdrant client
    if qdrant_api_key and qdrant_api_key.strip():
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        print("[+] Using Qdrant with API key authentication")
    else:
        client = QdrantClient(url=qdrant_url)
        print("[+] Using Qdrant without authentication (local instance)")
    
    # Verify connection
    try:
        collections = client.get_collections()
        print("[+] Successfully connected to Qdrant")
    except Exception as e:
        print(f"[!] Warning: Connection verification failed: {e}")
    
    # Create optimized embeddings (use our custom embeddings that fully leverage DashScope capabilities)
    print("[+] Setting up optimized DashScope embeddings...")
    embeddings = OptimizedDashScopeEmbeddings(
        api_key=dashscope_api_key,
        model="text-embedding-v4"
    )
    
    # Test embeddings
    try:
        test_embedding = embeddings.embed_query("test")
        print(f"[+] Embeddings working, dimension: {len(test_embedding)}")
        print("[+] Using text_type parameter for optimal query/document embedding")
    except Exception as e:
        print(f"[!] Embedding test failed: {e}")
        raise e
    
    # Create vectorstore from existing collection
    print(f"[+] Connecting to existing collection: {collection_name}")
    try:
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings
        )
        print("[+] Successfully connected to vectorstore")
    except Exception as e:
        print(f"[!] Failed to connect to vectorstore: {e}")
        raise e
    
    # Set up QA chain with Qwen LLM using OpenAI-compatible endpoint
    print("[+] Setting up Qwen LLM...")
    
    # Get model and API details from environment
    model_name = os.getenv("MODEL_CHOICE", "qwen-turbo")
    # Use the compatible-mode URL for OpenAI compatibility
    dashscope_url = os.getenv("DASHSCOPE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    
    # Check if we're using the correct URL format
    if "compatible-mode" not in dashscope_url:
        # Fix the URL to use compatible-mode endpoint
        dashscope_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        print(f"[+] Corrected DashScope URL to: {dashscope_url}")
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=model_name,
        api_key=dashscope_api_key,
        base_url=dashscope_url,
    )
    
    print(f"[+] Using model: {model_name}")
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    
    print("[+] Query chain setup complete!")
    return qa_chain


def semantic_search_only(query: str, 
                        collection_name: str = "website_docs",
                        qdrant_url: str = "http://localhost:6333",
                        qdrant_api_key: str = None,
                        top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Perform semantic search only (without LLM) using DashScope embeddings with optimized text_type.
    
    Args:
        query: Query string to search for
        collection_name: Name of the Qdrant collection to query
        qdrant_url: URL of the Qdrant instance
        qdrant_api_key: API key for Qdrant (optional for local instances)
        top_k: Number of top results to return
    
    Returns:
        List of tuples (document_content, similarity_score)
    """
    # Load environment variables
    load_dotenv()
    
    # Get DashScope API key
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    if not dashscope_api_key:
        raise ValueError("DASHSCOPE_API_KEY not found in environment variables")
    
    # Initialize Qdrant client
    if qdrant_api_key and qdrant_api_key.strip():
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        client = QdrantClient(url=qdrant_url)
    
    # Create optimized embeddings
    embeddings = OptimizedDashScopeEmbeddings(
        api_key=dashscope_api_key,
        model="text-embedding-v4"
    )
    
    # Generate query embedding with query optimization
    query_embedding = embeddings.embed_query(query)
    
    # Search in Qdrant
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k,
        with_payload=True
    )
    
    # Format results
    results = []
    for result in search_result:
        content = result.payload.get('page_content', '')
        score = result.score
        results.append((content, score))
    
    return results


def main():
    """Main function for the optimized RAG query tool."""
    parser = argparse.ArgumentParser(description="Optimized RAG Query Tool for Qdrant and DashScope")
    parser.add_argument("query", nargs="?", help="Query string to search for")
    parser.add_argument("--collection", default="website_docs", help="Qdrant collection name (default: website_docs)")
    parser.add_argument("--qdrant-url", default="http://localhost:6333", help="Qdrant URL (default: http://localhost:6333)")
    parser.add_argument("--search-only", action="store_true", help="Perform semantic search only without LLM")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return (default: 5)")
    
    args = parser.parse_args()
    
    # Check if query is provided
    if not args.query:
        print("[!] Error: Please provide a query string")
        print("Usage: python rag_query_optimized.py [--search-only] [--collection COLLECTION] [--qdrant-url URL] [--top-k K] 'your query here'")
        sys.exit(1)
    
    # Get Qdrant API key from environment
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if args.search_only:
        print("Semantic Search Only Mode")
        print("=" * 30)
        print(f"Query: {args.query}")
        
        try:
            print("[+] Searching...")
            results = semantic_search_only(
                query=args.query,
                collection_name=args.collection,
                qdrant_url=args.qdrant_url,
                qdrant_api_key=qdrant_api_key,
                top_k=args.top_k
            )
            
            print(f"\n[+] Top {len(results)} results:")
            for i, (content, score) in enumerate(results, 1):
                print(f"\n{i}. Score: {score:.4f}")
                # Truncate content for display
                truncated_content = content[:200] + "..." if len(content) > 200 else content
                print(f"   Content: {truncated_content}")
                
        except Exception as e:
            print(f"[!] Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("RAG Question Answering Mode")
        print("=" * 30)
        print(f"Query: {args.query}")
        
        try:
            # Set up the query chain
            qa_chain = setup_optimized_query_chain(
                collection_name=args.collection,
                qdrant_url=args.qdrant_url,
                qdrant_api_key=qdrant_api_key
            )
            
            print("[+] Thinking...")
            response = qa_chain.invoke({"query": args.query})
            
            print(f"\nAnswer: {response['result']}\n")
            
            if response.get('source_documents'):
                print("Sources:")
                for i, doc in enumerate(response['source_documents'], 1):
                    source = doc.metadata.get('source', 'Unknown')
                    content = doc.page_content
                    # Truncate content for display
                    truncated_content = content[:100] + "..." if len(content) > 100 else content
                    print(f"  {i}. {source}")
                    print(f"     Content: {truncated_content}")
            print()
            
        except Exception as e:
            print(f"[!] Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()