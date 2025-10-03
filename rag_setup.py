#!/usr/bin/env python3
"""
Optimized RAG Setup with Qdrant and DashScope for website content.

This script processes downloaded website content and sets up a RAG pipeline 
by storing documents in Qdrant with embeddings that fully leverage 
DashScope's semantic search optimization capabilities.
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
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


def load_markdown_files(directory: str) -> List[Document]:
    """Load all markdown files from a directory."""
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.md') or filename.endswith('.markdown'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # Create a Document object with metadata
                doc = Document(
                    page_content=content,
                    metadata={"source": file_path}
                )
                documents.append(doc)
    return documents


def load_jsonl_files(directory: str) -> List[Document]:
    """Load all JSONL files from a directory."""
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    try:
                        data = json.loads(line)
                        # Extract content from JSON object - use main_content field from website downloader
                        content = data.get('main_content', '')
                        if not content:
                            # Fallback to using the full data as string if main_content is not available
                            content = json.dumps(data)
                        # Create a Document object with metadata
                        doc = Document(
                            page_content=content,
                            metadata={
                                "source": file_path,
                                "url": data.get('url', ''),
                                "title": data.get('title', ''),
                                "line_number": line_num
                            }
                        )
                        documents.append(doc)
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON on line {line_num} in {filename}")
    return documents


def setup_optimized_rag_pipeline(input_directory: str, collection_name: str, file_format: str) -> tuple:
    """
    Set up the optimized RAG pipeline with website content using DashScope's full capabilities.
    
    Args:
        input_directory: Directory containing the downloaded website content
        collection_name: Name of the Qdrant collection
        file_format: Format of the files to process ('markdown', 'jsonl', or 'auto')
    
    Returns:
        Tuple of (vectorstore, file_format)
    """
    # Load environment variables
    load_dotenv()
    
    # Get API keys and URLs from environment variables
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    
    # Validate environment variables
    if not all([dashscope_api_key, qdrant_url]):
        raise ValueError("Missing required environment variables. Please check your .env file.")
    
    # Step 1: Load documents based on format
    print(f"[+] Loading {file_format} documents from {input_directory}...")
    
    if file_format == 'markdown':
        documents = load_markdown_files(input_directory)
    elif file_format == 'jsonl':
        documents = load_jsonl_files(input_directory)
    else:  # auto
        # Try to detect format based on available files
        markdown_files = [f for f in os.listdir(input_directory) if f.endswith(('.md', '.markdown'))]
        jsonl_files = [f for f in os.listdir(input_directory) if f.endswith('.jsonl')]
        
        if markdown_files and not jsonl_files:
            print("[+] Detected markdown files")
            documents = load_markdown_files(input_directory)
            file_format = 'markdown'
        elif jsonl_files and not markdown_files:
            print("[+] Detected JSONL files")
            documents = load_jsonl_files(input_directory)
            file_format = 'jsonl'
        elif markdown_files and jsonl_files:
            print("[!] Both markdown and JSONL files found. Please specify format explicitly.")
            print("    Use --format markdown or --format jsonl")
            sys.exit(1)
        else:
            print(f"[!] No supported files found in {input_directory}")
            print("[!] Supported formats: .md, .markdown, .jsonl")
            sys.exit(1)
    
    if not documents:
        print(f"[!] No documents found in {input_directory}")
        sys.exit(1)
    
    print(f"[+] Loaded {len(documents)} documents")
    
    # Step 2: Split documents into chunks
    print("[+] Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    print(f"[+] Split into {len(texts)} text chunks")
    
    # Step 3: Create optimized embeddings
    print("[+] Creating optimized DashScope embeddings...")
    print(f"[+] Using DashScope API key: {dashscope_api_key[:10]}...")
    
    # Use our optimized embeddings that fully leverage DashScope capabilities
    embeddings = OptimizedDashScopeEmbeddings(
        api_key=dashscope_api_key,
        model="text-embedding-v4"
    )
    
    # Test the embeddings
    try:
        test_embedding = embeddings.embed_query("test")
        print(f"[+] Embeddings working, dimension: {len(test_embedding)}")
        print("[+] Using text_type parameter for optimal query/document embedding")
    except Exception as e:
        print(f"[!] Embedding test failed: {e}")
        raise e
    
    # Step 4: Initialize Qdrant client
    print("[+] Initializing Qdrant client...")
    
    # Handle Qdrant connection based on whether we have an API key
    if qdrant_api_key and qdrant_api_key.strip():
        # Use API key authentication
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        print(f"[+] Using Qdrant with API key authentication")
    else:
        # For local instances without authentication
        client = QdrantClient(url=qdrant_url)
        print(f"[+] Using Qdrant without authentication (local instance)")
    
    # Step 5: Create or update vector store
    print(f"[+] Setting up Qdrant collection: {collection_name}")
    
    try:
        # Get embedding dimension
        test_embedding = embeddings.embed_query("test")
        embedding_dimension = len(test_embedding)
        print(f"[+] Embedding dimension: {embedding_dimension}")
        
        # Check if collection exists
        try:
            collection_info = client.get_collection(collection_name)
            print(f"[+] Collection '{collection_name}' already exists")
            # Check if dimensions match
            if collection_info.config.params.vectors.size != embedding_dimension:
                print(f"[!] Dimension mismatch. Deleting existing collection.")
                client.delete_collection(collection_name)
                raise Exception("Recreate collection with correct dimensions")
        except:
            # Collection doesn't exist, create it
            print(f"[+] Creating new collection: {collection_name}")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE)
            )
        
        # Create vectorstore
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings
        )
        
        # Add documents to vectorstore
        print("[+] Adding documents to vectorstore...")
        vectorstore.add_documents(texts)
        print("[+] Documents stored successfully")
        
    except Exception as e:
        print(f"[!] Error setting up vectorstore: {e}")
        raise e
    
    print("[+] Optimized RAG pipeline setup complete!")
    return vectorstore, file_format


def main():
    """Main function to process website content and set up optimized RAG pipeline."""
    parser = argparse.ArgumentParser(description="Optimized RAG Setup with Qdrant and DashScope")
    parser.add_argument("input_directory", help="Directory containing downloaded website content")
    parser.add_argument("--collection", default="website_docs", help="Qdrant collection name (default: website_docs)")
    parser.add_argument("--format", choices=['auto', 'markdown', 'jsonl'], default='auto', 
                        help="File format to process (default: auto)")
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_directory):
        print(f"[!] Input directory '{args.input_directory}' not found.")
        sys.exit(1)
    
    try:
        # Set up the optimized RAG pipeline
        vectorstore, detected_format = setup_optimized_rag_pipeline(
            args.input_directory, args.collection, args.format
        )
        
        # Print format information and recommendation
        print(f"\n[+] Processed files in {detected_format} format")
        if detected_format == 'jsonl':
            print("[+] JSONL format is recommended for structured data with metadata")
        elif detected_format == 'markdown':
            print("[+] Markdown format is recommended for clean, readable documentation")
            
    except Exception as e:
        print(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()