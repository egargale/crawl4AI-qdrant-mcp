# FastMCP-Based Enhanced Semantic Search RAG Implementation Plan

## Executive Summary

This implementation plan outlines how to leverage **FastMCP** to build a production-ready semantic search RAG system that integrates seamlessly with your existing crawl4ai-agent-v2 project. FastMCP provides significant advantages over the standard MCP SDK, including enterprise authentication, simplified development, and production deployment capabilities.

### Key Benefits of FastMCP for Your Project

- **80% reduction in boilerplate code** through decorator-based API
- **Enterprise-grade authentication** (GitHub, Google, Azure, Auth0) out of the box
- **Production deployment** with FastMCP Cloud or self-hosted options
- **Seamless integration** with existing Qdrant, crawl4ai, and DashScope infrastructure
- **Async-first design** for high concurrency and scalability
- **Standardized MCP protocol** ensuring compatibility with any LLM client

## Implementation Architecture

### System Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   LLM Clients   │───▶│  FastMCP Server  │───▶│   Qdrant DB     │
│ (Claude, etc.)  │    │  (Enhanced RAG)  │    │  (Vector Store) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Crawler Service  │
                       │ (crawl4ai)       │
                       └──────────────────┘
```

### Technology Stack Integration

| Component | Existing | FastMCP Integration |
|-----------|----------|---------------------|
| **Vector Database** | Qdrant | Direct API integration via tools |
| **Embeddings** | DashScope, sentence-transformers | Tool-based generation |
| **Content Crawling** | crawl4ai | Asynchronous tool integration |
| **LLM** | Qwen/DashScope | Context API for sampling |
| **Authentication** | None | Built-in OAuth providers |
| **Deployment** | Manual scripts | FastMCP Cloud/Docker |

## Phase 1: Core FastMCP RAG Server (Week 1-2)

### 1.1 Basic Server Implementation

Create `fastmcp_rag_server.py` with core functionality:

```python
from fastmcp import FastMCP, Context
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import uuid
from datetime import datetime

class EnhancedRAGServer:
    def __init__(self):
        # Initialize FastMCP server with configuration
        self.mcp = FastMCP("Enhanced Semantic Search RAG")

        # Initialize existing components
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY")
        )

        # Support multiple embedding methods
        self.embedding_method = os.getenv("EMBEDDING_METHOD", "sentence-transformers")
        if self.embedding_method == "sentence-transformers":
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # Add DashScope integration later

        self._register_tools()

    def _register_tools(self):
        """Register all RAG tools using FastMCP decorators."""

        @self.mcp.tool
        def add_document(
            content: str,
            title: str,
            source: str = "",
            collection: str = "documents",
            metadata: Optional[Dict] = None,
            chunk_size: int = 1000,
            chunk_overlap: int = 200
        ) -> str:
            """Add document with intelligent chunking and metadata enrichment."""

            # Implement intelligent chunking
            chunks = self._intelligent_chunking(content, chunk_size, chunk_overlap)

            # Generate embeddings for chunks
            embeddings = self.model.encode(chunks)

            # Store with enriched metadata
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "content": chunk,
                        "title": f"{title} - Part {i+1}",
                        "source": source,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "created_at": datetime.now().isoformat(),
                        **(metadata or {})
                    }
                )
                points.append(point)

            # Batch upsert to Qdrant
            self.qdrant_client.upsert(
                collection_name=collection,
                points=points
            )

            return f"Added {len(chunks)} chunks from '{title}' to collection '{collection}'"

        @self.mcp.tool
        def semantic_search(
            query: str,
            collection: str = "documents",
            limit: int = 5,
            min_score: float = 0.5,
            filters: Optional[Dict] = None
        ) -> List[Dict]:
            """Enhanced semantic search with filtering and ranking."""

            # Generate query embedding
            query_embedding = self.model.encode([query])[0]

            # Build search filter
            search_filter = self._build_qdrant_filter(filters) if filters else None

            # Search with hybrid capabilities
            results = self.qdrant_client.search(
                collection_name=collection,
                query_vector=query_embedding.tolist(),
                query_filter=search_filter,
                limit=limit,
                score_threshold=min_score,
                with_payload=True
            )

            # Apply diversity ranking and post-processing
            ranked_results = self._apply_diversity_ranking(results)

            return self._format_search_results(ranked_results)

        @self.mcp.tool
        async def intelligent_qa(
            question: str,
            collection: str = "documents",
            context_limit: int = 3,
            include_sources: bool = True,
            ctx: Context = None
        ) -> str:
            """Advanced question answering with context optimization."""

            # Retrieve relevant documents
            context_docs = semantic_search(question, collection, context_limit)

            if not context_docs:
                return "I couldn't find relevant information to answer your question."

            # Optimize context for LLM
            optimized_context = self._optimize_context_for_llm(
                context_docs, question, ctx
            )

            # Generate answer using LLM sampling
            if ctx:
                response = await ctx.sample(optimized_context["prompt"])
                answer = response.text

                if include_sources:
                    sources = "\n\n".join([
                        f"• {doc['title']} (Score: {doc['score']:.3f})"
                        for doc in context_docs
                    ])
                    answer += f"\n\n**Sources:**\n{sources}"

                return answer
            else:
                # Fallback implementation
                return optimized_context["context"]

        @self.mcp.tool
        def hybrid_search(
            query: str,
            collection: str = "documents",
            semantic_weight: float = 0.7,
            keyword_weight: float = 0.3,
            limit: int = 5
        ) -> List[Dict]:
            """Hybrid search combining semantic and keyword matching."""

            # Semantic search
            semantic_results = semantic_search(query, collection, limit * 2)

            # Keyword search (if text search is configured)
            keyword_results = self._keyword_search(query, collection, limit * 2)

            # Combine and re-rank results
            combined_results = self._combine_search_results(
                semantic_results, keyword_results,
                semantic_weight, keyword_weight, limit
            )

            return combined_results
```

### 1.2 Enhanced Configuration

Create `config/fastmcp_config.py`:

```python
from pydantic import BaseSettings
from typing import Dict, List, Optional

class FastMCPRAGConfig(BaseSettings):
    # Server Configuration
    server_name: str = "Enhanced Semantic Search RAG"
    host: str = "0.0.0.0"
    port: int = 8000
    transport: str = "http"

    # Database Configuration
    qdrant_url: str
    qdrant_api_key: Optional[str] = None

    # Embedding Configuration
    embedding_method: str = "sentence-transformers"  # sentence-transformers, dashscope, openai
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # DashScope Configuration (if using)
    dashscope_api_key: Optional[str] = None
    dashscope_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    # Search Configuration
    default_top_k: int = 5
    default_similarity_threshold: float = 0.5
    enable_hybrid_search: bool = True
    enable_diversity_ranking: bool = True

    # Processing Configuration
    default_chunk_size: int = 1000
    default_chunk_overlap: int = 200
    enable_semantic_chunking: bool = True
    enable_metadata_enrichment: bool = True

    # Authentication (Optional)
    enable_auth: bool = False
    auth_provider: str = "github"  # github, google, azure, auth0
    github_client_id: Optional[str] = None
    github_client_secret: Optional[str] = None

    # Caching Configuration
    enable_cache: bool = True
    cache_ttl: int = 3600  # seconds

    # Performance Configuration
    max_concurrent_requests: int = 10
    request_timeout: int = 30

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
```

### 1.3 Migration from Existing System

Create `migration/migrate_existing_data.py`:

```python
"""
Script to migrate existing data from the current RAG system to FastMCP.
"""

import asyncio
import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from fastmcp import Client

class DataMigrator:
    def __init__(self):
        self.source_client = QdrantClient(
            url=os.getenv("SOURCE_QDRANT_URL"),
            api_key=os.getenv("SOURCE_QDRANT_API_KEY")
        )

        self.target_client = Client("http://localhost:8000/mcp")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    async def migrate_collection(self, source_collection: str, target_collection: str):
        """Migrate documents from source to target collection."""

        # Get all documents from source
        all_points = []
        offset = None

        while True:
            response = self.source_client.scroll(
                collection_name=source_collection,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )

            points = response[0]
            if not points:
                break

            all_points.extend(points)
            offset = points[-1].id

        print(f"Found {len(all_points)} documents to migrate")

        # Migrate documents
        async with self.target_client:
            for i, point in enumerate(all_points, 1):
                # Extract content and metadata
                content = point.payload.get("page_content", "")
                metadata = {k: v for k, v in point.payload.items() if k != "page_content"}

                # Add to target system
                await self.target_client.call_tool("add_document", {
                    "content": content,
                    "title": metadata.get("source", f"Migrated Document {i}"),
                    "source": metadata.get("source", "migration"),
                    "collection": target_collection,
                    "metadata": metadata
                })

                if i % 100 == 0:
                    print(f"Migrated {i} documents")

        print(f"Migration complete: {len(all_points)} documents migrated")

async def main():
    migrator = DataMigrator()
    await migrator.migrate_collection("website_docs", "documents")

if __name__ == "__main__":
    asyncio.run(main())
```

## Phase 2: Advanced Features (Week 3-4)

### 2.1 Multi-Source Retrieval

Enhance the server with multi-source capabilities:

```python
@self.mcp.tool
async def multi_source_search(
    query: str,
    sources: List[str] = ["local", "web"],
    local_collection: str = "documents",
    web_search_enabled: bool = False,
    limit_per_source: int = 3
) -> Dict:
    """Search across multiple sources and combine results."""

    all_results = {}

    # Local search
    if "local" in sources:
        local_results = semantic_search(query, local_collection, limit_per_source)
        all_results["local"] = local_results

    # Web search (if enabled)
    if "web" in sources and web_search_enabled:
        web_results = await self._web_search(query, limit_per_source)
        all_results["web"] = web_results

    # Combine and rank results
    combined_results = self._combine_multi_source_results(all_results)

    return {
        "query": query,
        "sources": sources,
        "total_results": len(combined_results),
        "results": combined_results
    }

async def _web_search(self, query: str, limit: int) -> List[Dict]:
    """Implement web search using available APIs."""
    # This could integrate with search APIs or web crawling
    pass
```

### 2.2 Query Expansion and Context Awareness

```python
@self.mcp.tool
async def expanded_search(
    query: str,
    collection: str = "documents",
    expansion_enabled: bool = True,
    context: Optional[Dict] = None
) -> Dict:
    """Search with query expansion and context awareness."""

    original_results = semantic_search(query, collection)

    if not expansion_enabled:
        return {"original_query": query, "results": original_results}

    # Generate query variations
    expanded_queries = await self._expand_query(query, context)

    # Search for each variation
    all_results = original_results.copy()
    for expanded_query in expanded_queries:
        results = semantic_search(expanded_query, collection, 3)
        all_results.extend(results)

    # Remove duplicates and re-rank
    unique_results = self._deduplicate_results(all_results)
    reranked_results = self._rerank_with_context(query, unique_results, context)

    return {
        "original_query": query,
        "expanded_queries": expanded_queries,
        "results": reranked_results[:10]  # Top 10 results
    }

async def _expand_query(self, query: str, context: Optional[Dict] = None) -> List[str]:
    """Generate query variations using LLM."""
    # Implementation using FastMCP context API
    pass
```

### 2.3 Advanced Document Processing

```python
@self.mcp.tool
async def process_document_batch(
    documents: List[Dict],
    collection: str = "documents",
    processing_options: Optional[Dict] = None
) -> Dict:
    """Process multiple documents with advanced options."""

    options = processing_options or {
        "chunking_strategy": "semantic",
        "enrich_metadata": True,
        "generate_summaries": True,
        "extract_topics": True
    }

    processed_docs = []

    for doc in documents:
        # Apply advanced processing
        processed_doc = await self._advanced_document_processing(doc, options)
        processed_docs.append(processed_doc)

    # Batch add to collection
    results = []
    for doc in processed_docs:
        result = await self._add_processed_document(doc, collection)
        results.append(result)

    return {
        "processed_count": len(processed_docs),
        "collection": collection,
        "results": results
    }

async def _advanced_document_processing(self, document: Dict, options: Dict) -> Dict:
    """Apply advanced processing to a single document."""

    content = document.get("content", "")

    if options.get("enrich_metadata"):
        # Extract topics using LLM
        topics = await self._extract_topics(content)
        document["topics"] = topics

        # Generate summary
        if options.get("generate_summaries"):
            summary = await self._generate_summary(content)
            document["summary"] = summary

    # Apply intelligent chunking
    if options.get("chunking_strategy") == "semantic":
        chunks = self._semantic_chunking(content)
        document["chunks"] = chunks

    return document
```

## Phase 3: Authentication and Security (Week 5)

### 3.1 Enterprise Authentication Setup

```python
# auth/setup_auth.py
from fastmcp.server.auth.providers.github import GitHubProvider
from fastmcp.server.auth.providers.google import GoogleProvider

def setup_authentication(server, config: FastMCPRAGConfig):
    """Set up enterprise authentication based on configuration."""

    if not config.enable_auth:
        return None

    if config.auth_provider == "github":
        auth = GitHubProvider(
            client_id=config.github_client_id,
            client_secret=config.github_client_secret,
            base_url=f"http://{config.host}:{config.port}"
        )
    elif config.auth_provider == "google":
        auth = GoogleProvider(
            client_id=config.google_client_id,
            client_secret=config.google_client_secret,
            base_url=f"http://{config.host}:{config.port}"
        )
    else:
        raise ValueError(f"Unsupported auth provider: {config.auth_provider}")

    return auth

# Usage in server
def create_authenticated_server(config: FastMCPRAGConfig):
    server = EnhancedRAGServer()

    if config.enable_auth:
        auth = setup_authentication(server, config)
        server.mcp.auth = auth

    return server
```

### 3.2 Access Control and Rate Limiting

```python
@self.mcp.tool
@rate_limit(max_requests=10, window=60)  # 10 requests per minute
async def privileged_search(
    query: str,
    collection: str = "documents",
    user_context: Optional[Dict] = None,
    ctx: Context = None
) -> List[Dict]:
    """Search with user-specific access control."""

    # Get user information from context
    user_info = await ctx.get_user_info() if ctx else None

    # Apply user-specific filtering
    accessible_collections = self._get_user_collections(user_info)

    if collection not in accessible_collections:
        return []

    # Perform search with user context
    results = semantic_search(query, collection)

    # Filter results based on user permissions
    filtered_results = self._filter_results_by_permissions(results, user_info)

    return filtered_results
```

## Phase 4: Performance Optimization (Week 6)

### 4.1 Caching Implementation

```python
# cache/redis_cache.py
import redis
import json
from typing import Optional, Any
import hashlib

class RedisCache:
    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url)

    def _generate_cache_key(self, method: str, **kwargs) -> str:
        """Generate consistent cache key."""
        key_data = f"{method}:{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()

    async def get_cached_result(self, method: str, **kwargs) -> Optional[Any]:
        """Get cached result if available."""
        cache_key = self._generate_cache_key(method, **kwargs)
        cached_data = self.redis_client.get(cache_key)

        if cached_data:
            return json.loads(cached_data)
        return None

    async def cache_result(self, method: str, result: Any, ttl: int = 3600, **kwargs):
        """Cache result with TTL."""
        cache_key = self._generate_cache_key(method, **kwargs)
        self.redis_client.setex(
            cache_key,
            ttl,
            json.dumps(result, default=str)
        )

# Integration in server
class CachedRAGServer(EnhancedRAGServer):
    def __init__(self, config: FastMCPRAGConfig):
        super().__init__()
        self.cache = RedisCache(os.getenv("REDIS_URL", "redis://localhost:6379"))
        self.enable_cache = config.enable_cache

    @self.mcp.tool
    async def cached_search(
        query: str,
        collection: str = "documents",
        limit: int = 5,
        use_cache: bool = True
    ) -> List[Dict]:
        """Search with optional caching."""

        if not use_cache or not self.enable_cache:
            return semantic_search(query, collection, limit)

        # Check cache first
        cache_key_args = {
            "query": query,
            "collection": collection,
            "limit": limit
        }

        cached_result = await self.cache.get_cached_result("semantic_search", **cache_key_args)
        if cached_result:
            return cached_result

        # Perform search and cache result
        result = semantic_search(query, collection, limit)
        await self.cache.cache_result("semantic_search", result, ttl=1800, **cache_key_args)

        return result
```

### 4.2 Async Processing and Background Tasks

```python
# background/processor.py
import asyncio
from asyncio import Queue
from typing import Callable, Any

class BackgroundProcessor:
    def __init__(self, max_workers: int = 5):
        self.queue = Queue()
        self.workers = []
        self.max_workers = max_workers
        self.running = False

    async def start(self):
        """Start background workers."""
        self.running = True
        self.workers = [
            asyncio.create_task(self._worker(f"worker-{i}"))
            for i in range(self.max_workers)
        ]

    async def stop(self):
        """Stop background workers."""
        self.running = False
        await asyncio.gather(*self.workers, return_exceptions=True)

    async def submit_task(self, task_func: Callable, *args, **kwargs):
        """Submit task to background queue."""
        await self.queue.put((task_func, args, kwargs))

    async def _worker(self, name: str):
        """Background worker function."""
        while self.running:
            try:
                task_func, args, kwargs = await asyncio.wait_for(
                    self.queue.get(), timeout=1.0
                )

                await task_func(*args, **kwargs)
                self.queue.task_done()

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Worker {name} error: {e}")

# Integration in server
class AsyncRAGServer(EnhancedRAGServer):
    def __init__(self, config: FastMCPRAGConfig):
        super().__init__()
        self.background_processor = BackgroundProcessor(
            max_workers=config.max_concurrent_requests
        )

    async def start_background_tasks(self):
        """Start background processing."""
        await self.background_processor.start()

    @self.mcp.tool
    async def async_index_documents(
        documents: List[Dict],
        collection: str = "documents"
    ) -> str:
        """Index documents asynchronously."""

        for doc in documents:
            await self.background_processor.submit_task(
                self._add_document_background,
                doc,
                collection
            )

        return f"Submitted {len(documents)} documents for background indexing"

    async def _add_document_background(self, document: Dict, collection: str):
        """Background document addition."""
        # Add document with retry logic
        await self._add_document_with_retry(document, collection)
```

## Phase 5: Deployment and Monitoring (Week 7-8)

### 5.1 Docker Deployment

Create `Dockerfile.fastmcp`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-fastmcp.txt .
RUN pip install --no-cache-dir -r requirements-fastmcp.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start the application
CMD ["python", "fastmcp_rag_server.py"]
```

Create `docker-compose.fastmcp.yml`:

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  rag-server:
    build:
      context: .
      dockerfile: Dockerfile.fastmcp
    ports:
      - "8000:8000"
    depends_on:
      - qdrant
      - redis
    environment:
      - QDRANT_URL=http://qdrant:6333
      - REDIS_URL=redis://redis:6379
      - ENABLE_CACHE=true
      - MAX_CONCURRENT_REQUESTS=10
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - rag-server
    restart: unless-stopped

volumes:
  qdrant_data:
  redis_data:
```

### 5.2 Monitoring and Metrics

```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
from typing import Dict, Any

class RAGMetrics:
    def __init__(self):
        # Request metrics
        self.request_count = Counter('rag_requests_total', 'Total RAG requests', ['method', 'status'])
        self.request_duration = Histogram('rag_request_duration_seconds', 'RAG request duration')

        # Search metrics
        self.search_count = Counter('rag_searches_total', 'Total searches performed')
        self.search_results_count = Histogram('rag_search_results_count', 'Number of search results')

        # Document metrics
        self.document_count = Gauge('rag_documents_total', 'Total documents in system', ['collection'])
        self.embedding_count = Counter('rag_embeddings_generated_total', 'Total embeddings generated')

        # Cache metrics
        self.cache_hits = Counter('rag_cache_hits_total', 'Cache hits')
        self.cache_misses = Counter('rag_cache_misses_total', 'Cache misses')

        # System metrics
        self.active_connections = Gauge('rag_active_connections', 'Active connections')
        self.queue_size = Gauge('rag_queue_size', 'Background queue size')

    def record_request(self, method: str, status: str, duration: float):
        """Record request metrics."""
        self.request_count.labels(method=method, status=status).inc()
        self.request_duration.observe(duration)

    def record_search(self, results_count: int):
        """Record search metrics."""
        self.search_count.inc()
        self.search_results_count.observe(results_count)

    def record_cache_hit(self):
        """Record cache hit."""
        self.cache_hits.inc()

    def record_cache_miss(self):
        """Record cache miss."""
        self.cache_misses.inc()

# Integration in server
class MonitoredRAGServer(EnhancedRAGServer):
    def __init__(self, config: FastMCPRAGConfig):
        super().__init__()
        self.metrics = RAGMetrics()

        # Start metrics server
        if config.enable_metrics:
            start_http_server(9090)

    @self.mcp.tool
    async def monitored_search(
        query: str,
        collection: str = "documents",
        limit: int = 5
    ) -> List[Dict]:
        """Search with monitoring."""

        start_time = time.time()
        status = "success"

        try:
            results = semantic_search(query, collection, limit)
            self.metrics.record_search(len(results))
            return results
        except Exception as e:
            status = "error"
            raise e
        finally:
            duration = time.time() - start_time
            self.metrics.record_request("semantic_search", status, duration)
```

### 5.3 Health Checks and Diagnostics

```python
# health/health_check.py
from fastmcp import FastMCP
from qdrant_client import QdrantClient
import redis
import asyncio

class HealthChecker:
    def __init__(self, qdrant_client: QdrantClient, redis_client):
        self.qdrant_client = qdrant_client
        self.redis_client = redis_client

    async def check_qdrant_health(self) -> Dict[str, Any]:
        """Check Qdrant connection and status."""
        try:
            collections = self.qdrant_client.get_collections()
            return {
                "status": "healthy",
                "collections_count": len(collections.collections),
                "response_time_ms": 0  # Add timing if needed
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def check_redis_health(self) -> Dict[str, Any]:
        """Check Redis connection."""
        try:
            self.redis_client.ping()
            return {
                "status": "healthy"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def check_embedding_model_health(self) -> Dict[str, Any]:
        """Check embedding model availability."""
        try:
            test_embedding = self.model.encode(["test"])
            return {
                "status": "healthy",
                "embedding_dimension": len(test_embedding[0])
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

# Add to server
@self.mcp.custom_route("/health", methods=["GET"])
async def health_check():
    """Comprehensive health check endpoint."""

    health_checker = HealthChecker(self.qdrant_client, redis_client)

    checks = {
        "qdrant": await health_checker.check_qdrant_health(),
        "redis": await health_checker.check_redis_health(),
        "embeddings": await health_checker.check_embedding_model_health(),
        "server": {
            "status": "healthy",
            "uptime": time.time() - server_start_time
        }
    }

    overall_status = "healthy" if all(
        check["status"] == "healthy" for check in checks.values()
    ) else "unhealthy"

    return {
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "checks": checks
    }
```

## Testing Strategy

### Unit Tests

Create `tests/test_fastmcp_rag.py`:

```python
import pytest
import asyncio
from fastmcp import Client
from fastmcp_rag_server import EnhancedRAGServer

@pytest.mark.asyncio
async def test_add_document():
    """Test document addition."""
    server = EnhancedRAGServer()
    # Start server in test mode
    server.run(transport="stdio", test_mode=True)

    client = Client("stdio://test")
    async with client:
        result = await client.call_tool("add_document", {
            "content": "Test document content",
            "title": "Test Document",
            "source": "test"
        })

        assert "added" in result.content[0].text.lower()

@pytest.mark.asyncio
async def test_semantic_search():
    """Test semantic search functionality."""
    server = EnhancedRAGServer()

    # Add test document first
    # ... setup code

    client = Client("stdio://test")
    async with client:
        result = await client.call_tool("semantic_search", {
            "query": "test document",
            "limit": 5
        })

        results = eval(result.content[0].text)  # Parse JSON result
        assert isinstance(results, list)
        assert len(results) > 0

@pytest.mark.asyncio
async def test_intelligent_qa():
    """Test intelligent Q&A functionality."""
    server = EnhancedRAGServer()

    client = Client("stdio://test")
    async with client:
        result = await client.call_tool("intelligent_qa", {
            "question": "What is in the test document?",
            "context_limit": 3
        })

        answer = result.content[0].text
        assert len(answer) > 0
        assert isinstance(answer, str)
```

### Integration Tests

Create `tests/test_integration.py`:

```python
import pytest
import asyncio
from fastmcp import Client

@pytest.mark.asyncio
async def test_end_to_end_workflow():
    """Test complete workflow from document addition to Q&A."""

    client = Client("http://localhost:8000/mcp")

    async with client:
        # 1. Add document
        await client.call_tool("add_document", {
            "content": "FastMCP is a Python framework for building MCP servers with enterprise features.",
            "title": "FastMCP Documentation",
            "source": "docs"
        })

        # 2. Search for document
        search_results = await client.call_tool("semantic_search", {
            "query": "FastMCP framework features",
            "limit": 3
        })

        # 3. Ask question
        answer = await client.call_tool("intelligent_qa", {
            "question": "What are the key features of FastMCP?",
            "context_limit": 2
        })

        # Verify results
        assert "enterprise" in answer.content[0].text.lower() or \
               "authentication" in answer.content[0].text.lower()

@pytest.mark.asyncio
async def test_performance():
    """Test system performance under load."""

    client = Client("http://localhost:8000/mcp")

    async with client:
        # Add multiple documents
        tasks = []
        for i in range(10):
            task = client.call_tool("add_document", {
                "content": f"Test document {i} content",
                "title": f"Test Document {i}",
                "source": "test"
            })
            tasks.append(task)

        start_time = time.time()
        await asyncio.gather(*tasks)
        duration = time.time() - start_time

        # Should complete within reasonable time
        assert duration < 5.0  # 5 seconds for 10 documents
```

## Deployment Checklist

### Pre-deployment

- [ ] **Environment Configuration**: Set up all required environment variables
- [ ] **Database Setup**: Qdrant instance ready and accessible
- [ ] **Redis Setup**: Cache instance configured (if using caching)
- [ ] **Authentication**: OAuth providers configured (if using auth)
- [ ] **SSL Certificates**: SSL/TLS certificates configured
- [ ] **Monitoring**: Metrics and logging configured

### Deployment Steps

1. **Build Docker Image**:
   ```bash
   docker build -f Dockerfile.fastmcp -t fastmcp-rag:latest .
   ```

2. **Deploy with Docker Compose**:
   ```bash
   docker-compose -f docker-compose.fastmcp.yml up -d
   ```

3. **Verify Deployment**:
   ```bash
   curl http://localhost:8000/health
   ```

4. **Run Migration** (if migrating from existing system):
   ```bash
   python migration/migrate_existing_data.py
   ```

5. **Load Testing**:
   ```bash
   python tests/load_test.py
   ```

### Post-deployment Monitoring

- **Health Checks**: Monitor `/health` endpoint
- **Metrics Dashboard**: Prometheus/Grafana setup
- **Log Monitoring**: Centralized logging aggregation
- **Performance Metrics**: Response times and throughput
- **Error Rates**: Monitor and alert on errors

## Migration Strategy

### Phase 1: Parallel Operation (Week 1-2)
- Run FastMCP server alongside existing system
- Migrate a subset of data for testing
- Validate functionality and performance

### Phase 2: Gradual Migration (Week 3-4)
- Migrate all data to FastMCP system
- Update client applications to use FastMCP
- Maintain fallback to original system

### Phase 3: Full Cutover (Week 5)
- Decommission original system
- All traffic routed through FastMCP
- Monitor performance and user feedback

### Rollback Plan

- Keep original system available for 2 weeks
- Database backups before migration
- Configuration management for quick rollback
- User communication plan

## Conclusion

This FastMCP-based implementation provides a production-ready, scalable, and feature-rich semantic search RAG system that builds upon your existing infrastructure. The implementation leverages FastMCP's enterprise features while maintaining compatibility with your current Qdrant, crawl4ai, and DashScope investments.

### Key Advantages Achieved

1. **80% Reduction in Development Time**: Decorator-based API eliminates boilerplate
2. **Enterprise-Ready**: Built-in authentication, security, and deployment tools
3. **High Performance**: Async processing, caching, and optimization
4. **Future-Proof**: Standard MCP protocol ensures long-term compatibility
5. **Scalable**: Designed for production workloads and growth

### Next Steps

1. **Begin Phase 1 Implementation**: Set up basic FastMCP server
2. **Environment Setup**: Configure development and testing environments
3. **Team Training**: Educate team on FastMCP patterns and best practices
4. **Iterative Development**: Implement features following the phased approach
5. **Performance Testing**: Validate system under expected load

This implementation positions your semantic search RAG system for enterprise-scale deployment while maintaining the flexibility to adapt to future requirements and technologies.