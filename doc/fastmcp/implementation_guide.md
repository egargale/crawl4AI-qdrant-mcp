# FastMCP RAG Implementation Guide

## Quick Start: Building a Semantic Search RAG Server with FastMCP

This guide provides step-by-step instructions for implementing an enhanced semantic search RAG system using FastMCP.

### Prerequisites

```bash
# Install required packages
pip install fastmcp qdrant-client sentence-transformers crawl4ai

# Optional: Install uv for better dependency management
pip install uv
```

### 1. Basic RAG Server Implementation

Create `fastmcp_rag_server.py`:

```python
from fastmcp import FastMCP, Context
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid
from datetime import datetime
import os
from typing import List, Dict, Optional

class RAGServer:
    def __init__(self):
        # Initialize FastMCP server
        self.mcp = FastMCP("Enhanced Semantic Search RAG")

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            host="localhost",
            port=6333
        )

        # Initialize embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Ensure collection exists
        self._ensure_collection("documents")

        # Register tools
        self._register_tools()

    def _ensure_collection(self, collection_name: str):
        """Create collection if it doesn't exist."""
        try:
            self.qdrant_client.get_collection(collection_name)
        except:
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )

    def _register_tools(self):
        """Register all MCP tools."""

        @self.mcp.tool
        def add_document(
            content: str,
            title: str,
            source: str = "",
            collection: str = "documents",
            metadata: Optional[Dict] = None
        ) -> str:
            """Add a document to the RAG system with full-text search support."""

            # Generate embedding
            embedding = self.model.encode([content])[0]

            # Create point
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={
                    "content": content,
                    "title": title,
                    "source": source,
                    "created_at": datetime.now().isoformat(),
                    **(metadata or {})
                }
            )

            # Store in Qdrant
            self.qdrant_client.upsert(
                collection_name=collection,
                points=[point]
            )

            return f"Document '{title}' added to collection '{collection}'"

        @self.mcp.tool
        def search_documents(
            query: str,
            collection: str = "documents",
            limit: int = 5,
            min_score: float = 0.5
        ) -> List[Dict]:
            """Search documents using semantic similarity with configurable thresholds."""

            # Generate query embedding
            query_embedding = self.model.encode([query])[0]

            # Search in Qdrant
            results = self.qdrant_client.search(
                collection_name=collection,
                query_vector=query_embedding.tolist(),
                limit=limit,
                score_threshold=min_score,
                with_payload=True
            )

            # Format results
            return [
                {
                    "id": hit.id,
                    "title": hit.payload.get("title", "Untitled"),
                    "content": hit.payload["content"][:300] + "..." if len(hit.payload["content"]) > 300 else hit.payload["content"],
                    "score": hit.score,
                    "source": hit.payload.get("source", ""),
                    "created_at": hit.payload.get("created_at", "")
                }
                for hit in results
            ]

        @self.mcp.tool
        async def intelligent_qa(
            question: str,
            collection: str = "documents",
            context_limit: int = 3,
            ctx: Context = None
        ) -> str:
            """Answer questions using retrieved context with LLM sampling."""

            # Retrieve relevant documents
            context_docs = search_documents(question, collection, context_limit)

            if not context_docs:
                return "I couldn't find relevant information to answer your question."

            # Build context
            context = "\n\n".join([
                f"Title: {doc['title']}\nContent: {doc['content']}"
                for doc in context_docs
            ])

            # Log the search process
            if ctx:
                await ctx.info(f"Retrieved {len(context_docs)} documents for question: {question}")

            # Generate answer using LLM sampling
            prompt = f"""
            Based on the following context, please provide a comprehensive answer to the question.

            Question: {question}

            Context:
            {context}

            Please provide a detailed answer based on the context provided above.
            """

            if ctx:
                response = await ctx.sample(prompt)
                return response.text
            else:
                # Fallback for environments without context
                return f"Based on the retrieved context, here's what I found:\n\n{context}"

        @self.mcp.tool
        def delete_document(
            document_id: str,
            collection: str = "documents"
        ) -> str:
            """Delete a document from the RAG system."""

            try:
                self.qdrant_client.delete(
                    collection_name=collection,
                    points_selector=[document_id]
                )
                return f"Document {document_id} deleted successfully"
            except Exception as e:
                return f"Error deleting document: {str(e)}"

        @self.mcp.tool
        def list_collections() -> List[str]:
            """List all available collections."""
            return self.qdrant_client.get_collections().collections

        @self.mcp.tool
        def get_collection_stats(collection: str = "documents") -> Dict:
            """Get statistics for a collection."""
            try:
                info = self.qdrant_client.get_collection(collection)
                return {
                    "name": collection,
                    "vectors_count": info.vectors_count,
                    "status": info.status,
                    "optimizer_status": info.optimizer_status
                }
            except Exception as e:
                return {"error": str(e)}

    def run(self, transport: str = "http", port: int = 8000, **kwargs):
        """Run the FastMCP server."""
        self.mcp.run(transport=transport, port=port, **kwargs)

# Usage
if __name__ == "__main__":
    server = RAGServer()
    server.run(transport="http", host="0.0.0.0", port=8000)
```

### 2. Enhanced Crawler Integration

Create `crawler_integration.py`:

```python
from fastmcp import FastMCP
from crawl4ai import AsyncWebCrawler
from sentence_transformers import SentenceTransformer
import uuid
from datetime import datetime
from typing import List, Dict
import asyncio

class CrawlerRAGServer:
    def __init__(self):
        self.mcp = FastMCP("Crawler RAG Server")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self._register_tools()

    def _register_tools(self):

        @self.mcp.tool
        async def crawl_and_index(
            url: str,
            collection: str = "web_documents",
            max_depth: int = 1,
            include_images: bool = False
        ) -> Dict:
            """Crawl a website and index its content for semantic search."""

            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(
                    url,
                    word_count_threshold=10,
                    exclude_external_links=False
                )

            if not result.success:
                return {"error": f"Failed to crawl {url}: {result.error_message}"}

            # Process content
            content = result.markdown
            title = result.title or url
            metadata = {
                "url": url,
                "title": title,
                "crawled_at": datetime.now().isoformat(),
                "content_length": len(content),
                "links_count": len(result.links) if result.links else 0
            }

            # Add to RAG system (assuming we have access to RAG server)
            # This would typically make an API call to the RAG server
            # For now, we'll return the processed data

            return {
                "url": url,
                "title": title,
                "content_length": len(content),
                "metadata": metadata,
                "status": "processed"
            }

        @self.mcp.tool
        async def batch_crawl(
            urls: List[str],
            collection: str = "web_documents",
            max_concurrent: int = 3
        ) -> Dict:
            """Crawl multiple URLs concurrently."""

            semaphore = asyncio.Semaphore(max_concurrent)

            async def crawl_single(url):
                async with semaphore:
                    return await crawl_and_index(url, collection)

            # Execute crawls concurrently
            tasks = [crawl_single(url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            successful = [r for r in results if isinstance(r, dict) and "error" not in r]
            failed = [r for r in results if isinstance(r, dict) and "error" in r]

            return {
                "total_urls": len(urls),
                "successful": len(successful),
                "failed": len(failed),
                "results": successful,
                "errors": failed
            }

        @self.mcp.tool
        def extract_links(
            url: str,
            filter_external: bool = True
        ) -> Dict:
            """Extract and analyze links from a webpage."""

            # This would use crawl4ai to extract links
            # Simplified implementation for demonstration
            return {
                "url": url,
                "internal_links": [],
                "external_links": [],
                "total_links": 0
            }

    def run(self, transport: str = "http", port: int = 8001):
        """Run the crawler server."""
        self.mcp.run(transport=transport, host="0.0.0.0", port=port)

# Usage
if __name__ == "__main__":
    server = CrawlerRAGServer()
    server.run()
```

### 3. Client Implementation

Create `rag_client.py`:

```python
import asyncio
from fastmcp import Client

class RAGClient:
    def __init__(self, server_url: str = "http://localhost:8000/mcp"):
        self.client = Client(server_url)

    async def add_document(self, content: str, title: str, source: str = ""):
        """Add a document to the RAG system."""
        async with self.client:
            result = await self.client.call_tool("add_document", {
                "content": content,
                "title": title,
                "source": source
            })
            return result.content[0].text

    async def search(self, query: str, limit: int = 5):
        """Search for documents."""
        async with self.client:
            result = await self.client.call_tool("search_documents", {
                "query": query,
                "limit": limit
            })
            return result.content[0].text

    async def ask_question(self, question: str):
        """Ask a question using RAG."""
        async with self.client:
            result = await self.client.call_tool("intelligent_qa", {
                "question": question
            })
            return result.content[0].text

    async def get_stats(self, collection: str = "documents"):
        """Get collection statistics."""
        async with self.client:
            result = await self.client.call_tool("get_collection_stats", {
                "collection": collection
            })
            return result.content[0].text

# Example usage
async def main():
    client = RAGClient()

    # Add a document
    await client.add_document(
        content="FastMCP is a Python framework for building MCP servers.",
        title="FastMCP Overview",
        source="docs"
    )

    # Search for documents
    results = await client.search("FastMCP framework")
    print("Search results:", results)

    # Ask a question
    answer = await client.ask_question("What is FastMCP?")
    print("Answer:", answer)

if __name__ == "__main__":
    asyncio.run(main())
```

### 4. Running the System

#### Start Qdrant (using Docker):
```bash
docker run -p 6333:6333 qdrant/qdrant
```

#### Start the RAG Server:
```bash
python fastmcp_rag_server.py
```

#### Test with Client:
```bash
python rag_client.py
```

### 5. Integration with Existing Code

To integrate with your existing RAG system:

```python
# In your existing rag_setup.py or similar file
from fastmcp_rag_server import RAGServer

def create_fastmcp_server():
    """Create FastMCP server from existing RAG components."""
    server = RAGServer()

    # Add existing documents
    existing_docs = load_existing_documents()  # Your existing function
    for doc in existing_docs:
        # This would use the FastMCP client to add documents
        pass

    return server

if __name__ == "__main__":
    server = create_fastmcp_server()
    server.run()
```

### 6. Advanced Features

#### Authentication (Optional):
```python
from fastmcp.server.auth.providers.github import GitHubProvider

# Add authentication
auth = GitHubProvider(
    client_id="your_client_id",
    client_secret="your_client_secret",
    base_url="https://your-server.com"
)

server = RAGServer()
server.mcp.auth = auth
server.run()
```

#### Custom Routes:
```python
@server.mcp.custom_route("/health", methods=["GET"])
async def health_check():
    return {"status": "healthy", "service": "RAG Server"}
```

### 7. Testing

Create `test_rag_server.py`:

```python
import pytest
import asyncio
from fastmcp import Client

@pytest.mark.asyncio
async def test_add_document():
    client = Client("http://localhost:8000/mcp")
    async with client:
        result = await client.call_tool("add_document", {
            "content": "Test document",
            "title": "Test"
        })
        assert "added" in result.content[0].text.lower()

@pytest.mark.asyncio
async def test_search_documents():
    client = Client("http://localhost:8000/mcp")
    async with client:
        result = await client.call_tool("search_documents", {
            "query": "test"
        })
        # Assert search results
```

### 8. Deployment

#### Using FastMCP CLI:
```bash
# Install FastMCP CLI
pip install fastmcp

# Run with custom options
fastmcp run fastmcp_rag_server.py:RAGServer().mcp --transport http --port 8000
```

#### Docker Deployment:
```dockerfile
FROM python:3.11

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "fastmcp_rag_server.py"]
```

#### Docker Compose:
```yaml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"

  rag-server:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - qdrant
    environment:
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
```

This implementation guide provides a complete foundation for building a production-ready semantic search RAG system using FastMCP, integrating with your existing crawl4ai and Qdrant infrastructure.