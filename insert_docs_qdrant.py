"""
insert_docs_qdrant.py
--------------
Command-line utility to crawl any URL using Crawl4AI, detect content type (sitemap, .txt, or regular page),
use the appropriate crawl method, chunk the resulting Markdown into <1000 character blocks by header hierarchy,
and insert all chunks into Qdrant with metadata.

Usage:
    python insert_docs_qdrant.py <URL> [--collection ...] [--embedding-model ...]
"""
import argparse
import sys
import re
import asyncio
from typing import List, Dict, Any
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher
import requests
from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient, models
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

def smart_chunk_markdown(markdown: str, max_len: int = 1000) -> List[str]:
    """Hierarchically splits markdown by #, ##, ### headers, then by characters, to ensure all chunks < max_len."""
    def split_by_header(md, header_pattern):
        indices = [m.start() for m in re.finditer(header_pattern, md, re.MULTILINE)]
        indices.append(len(md))
        return [md[indices[i]:indices[i+1]].strip() for i in range(len(indices)-1) if md[indices[i]:indices[i+1]].strip()]

    chunks = []

    for h1 in split_by_header(markdown, r'^# .+$'):
        if len(h1) > max_len:
            for h2 in split_by_header(h1, r'^## .+$'):
                if len(h2) > max_len:
                    for h3 in split_by_header(h2, r'^### .+$'):
                        if len(h3) > max_len:
                            for i in range(0, len(h3), max_len):
                                chunks.append(h3[i:i+max_len].strip())
                        else:
                            chunks.append(h3)
                else:
                    chunks.append(h2)
        else:
            chunks.append(h1)

    final_chunks = []

    for c in chunks:
        if len(c) > max_len:
            final_chunks.extend([c[i:i+max_len].strip() for i in range(0, len(c), max_len)])
        else:
            final_chunks.append(c)

    return [c for c in final_chunks if c]

def is_sitemap(url: str) -> bool:
    return url.endswith('sitemap.xml') or 'sitemap' in urlparse(url).path

def is_txt(url: str) -> bool:
    return url.endswith('.txt')

async def crawl_recursive_internal_links(start_urls, max_depth=3, max_concurrent=10) -> List[Dict[str,Any]]:
    """Recursive crawl using logic from 5-crawl_recursive_internal_links.py. Returns list of dicts with url and markdown."""
    browser_config = BrowserConfig(headless=True, verbose=False)
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    visited = set()

    def normalize_url(url):
        return urldefrag(url)[0]

    current_urls = set([normalize_url(u) for u in start_urls])
    results_all = []

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for depth in range(max_depth):
            urls_to_crawl = [normalize_url(url) for url in current_urls if normalize_url(url) not in visited]
            if not urls_to_crawl:
                break

            results = await crawler.arun_many(urls=urls_to_crawl, config=run_config, dispatcher=dispatcher)
            next_level_urls = set()

            for result in results:
                norm_url = normalize_url(result.url)
                visited.add(norm_url)

                if result.success and result.markdown:
                    results_all.append({'url': result.url, 'markdown': result.markdown})
                    for link in result.links.get("internal", []):
                        next_url = normalize_url(link["href"])
                        if next_url not in visited:
                            next_level_urls.add(next_url)

            current_urls = next_level_urls

    return results_all

async def crawl_markdown_file(url: str) -> List[Dict[str,Any]]:
    """Crawl a .txt or markdown file using logic from 4-crawl_and_chunk_markdown.py."""
    browser_config = BrowserConfig(headless=True)
    crawl_config = CrawlerRunConfig()

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=crawl_config)
        if result.success and result.markdown:
            return [{'url': url, 'markdown': result.markdown}]
        else:
            print(f"Failed to crawl {url}: {result.error_message}")
            return []

def parse_sitemap(sitemap_url: str) -> List[str]:
    resp = requests.get(sitemap_url)
    urls = []

    if resp.status_code == 200:
        try:
            tree = ElementTree.fromstring(resp.content)
            urls = [loc.text for loc in tree.findall('.//{*}loc')]
        except Exception as e:
            print(f"Error parsing sitemap XML: {e}")

    return urls

async def crawl_batch(urls: List[str], max_concurrent: int = 10) -> List[Dict[str,Any]]:
    """Batch crawl using logic from 3-crawl_sitemap_in_parallel.py."""
    browser_config = BrowserConfig(headless=True, verbose=False)
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        results = await crawler.arun_many(urls=urls, config=crawl_config, dispatcher=dispatcher)
        return [{'url': r.url, 'markdown': r.markdown} for r in results if r.success and r.markdown]

def extract_section_info(chunk: str) -> Dict[str, Any]:
    """Extracts headers and stats from a chunk."""
    headers = re.findall(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
    header_str = '; '.join([f'{h[0]} {h[1]}' for h in headers]) if headers else ''

    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split())
    }

def get_qdrant_client() -> QdrantClient:
    """Initialize and return a Qdrant client."""
    QDRANT_URL = os.getenv('QDRANT_URL')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
    
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in the .env file")
    
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def get_embedding_client() -> OpenAI:
    """Initialize and return an OpenAI-compatible embedding client."""
    return OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    )

def generate_embeddings(client: OpenAI, texts: List[str], model_name: str, dimensions: int) -> List[List[float]]:
    """Generate embeddings for a list of texts."""
    all_embeddings = []
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(
            model=model_name,
            input=batch,
            dimensions=dimensions
        )
        all_embeddings.extend([data.embedding for data in resp.data])
    return all_embeddings

def insert_documents_to_qdrant(
    qdrant_client: QdrantClient, 
    embedding_client: OpenAI,
    collection_name: str, 
    model_name: str,
    embedding_dim: int,
    ids: List[str], 
    documents: List[str], 
    metadatas: List[Dict[str, Any]], 
    batch_size: int = 100
) -> None:
    """Add documents to a Qdrant collection in batches."""
    # Generate embeddings for all documents
    print("Generating embeddings...")
    embeddings = generate_embeddings(embedding_client, documents, model_name, embedding_dim)
    print(f"Generated {len(embeddings)} embeddings with {embedding_dim} dimensions")
    
    # Check if collection already exists and handle dimension mismatch
    try:
        # Try to get the collection info
        collection_info = qdrant_client.get_collection(collection_name)
        print(f"Collection '{collection_name}' exists with {collection_info.config.params.vectors.size} dimensions")
        
        # Check if dimensions match
        if collection_info.config.params.vectors.size != embedding_dim:
            print(f"Dimension mismatch: expected {embedding_dim}, got {collection_info.config.params.vectors.size}")
            print("Deleting existing collection and recreating with correct dimensions...")
            qdrant_client.delete_collection(collection_name=collection_name)
            
            # Recreate collection with correct dimensions
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=embedding_dim, distance=models.Distance.COSINE)
            )
        else:
            print(f"Collection '{collection_name}' already exists with correct dimensions.")
            
    except Exception as e:
        # Collection doesn't exist, so create it
        print(f"Collection '{collection_name}' does not exist. Creating it now.")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=embedding_dim, distance=models.Distance.COSINE)
        )
    
    # Upload points with pre-computed embeddings in batches
    print("Inserting documents into Qdrant...")
    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        batch_points = [
            models.PointStruct(
                id=ids[j],
                vector=embeddings[j],
                payload=metadatas[j]
            )
            for j in range(i, end_idx)
        ]
        
        qdrant_client.upsert(
            collection_name=collection_name,
            points=batch_points
        )
        
        print(f"Inserted batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")

def main():
    parser = argparse.ArgumentParser(description="Insert crawled docs into Qdrant")
    parser.add_argument("url", help="URL to crawl (regular, .txt, or sitemap)")
    parser.add_argument("--collection", default="docs", help="Qdrant collection name")
    parser.add_argument("--embedding-model", default="text-embedding-v3", help="Embedding model name")
    parser.add_argument("--embedding-dim", type=int, default=1024, help="Embedding dimension")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Max chunk size (chars)")
    parser.add_argument("--max-depth", type=int, default=3, help="Recursion depth for regular URLs")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Max parallel browser sessions")
    parser.add_argument("--batch-size", type=int, default=100, help="Qdrant insert batch size")
    args = parser.parse_args()

    # Initialize clients
    qdrant_client = get_qdrant_client()
    embedding_client = get_embedding_client()
    
    # Print connection info
    print("Successfully connected to Qdrant!")
    print(f"Qdrant URL: {os.getenv('QDRANT_URL')}")

    # Detect URL type
    url = args.url
    if is_txt(url):
        print(f"Detected .txt/markdown file: {url}")
        crawl_results = asyncio.run(crawl_markdown_file(url))
    elif is_sitemap(url):
        print(f"Detected sitemap: {url}")
        sitemap_urls = parse_sitemap(url)
        if not sitemap_urls:
            print("No URLs found in sitemap.")
            sys.exit(1)
        crawl_results = asyncio.run(crawl_batch(sitemap_urls, max_concurrent=args.max_concurrent))
    else:
        print(f"Detected regular URL: {url}")
        crawl_results = asyncio.run(crawl_recursive_internal_links([url], max_depth=args.max_depth, max_concurrent=args.max_concurrent))

    # Chunk and collect metadata
    ids, documents, metadatas = [], [], []
    chunk_idx = 0
    for doc in crawl_results:
        url = doc['url']
        md = doc['markdown']
        chunks = smart_chunk_markdown(md, max_len=args.chunk_size)
        for chunk in chunks:
            ids.append(chunk_idx)
            documents.append(chunk)
            meta = extract_section_info(chunk)
            meta["document"] = chunk  # Include the actual document content
            meta["chunk_index"] = chunk_idx
            meta["source"] = url
            metadatas.append(meta)
            chunk_idx += 1

    if not documents:
        print("No documents found to insert.")
        sys.exit(1)

    print(f"Inserting {len(documents)} chunks into Qdrant collection '{args.collection}'...")

    insert_documents_to_qdrant(
        qdrant_client, 
        embedding_client,
        args.collection, 
        args.embedding_model,
        args.embedding_dim,
        ids, 
        documents, 
        metadatas, 
        batch_size=args.batch_size
    )

    print(f"Successfully added {len(documents)} chunks to Qdrant collection '{args.collection}'.")

if __name__ == "__main__":
    main()