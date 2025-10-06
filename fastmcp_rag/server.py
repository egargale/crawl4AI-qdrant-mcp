"""
FastMCP RAG Server Core Implementation

This module provides the core FastMCP server implementation for the
enhanced semantic search RAG system.
"""

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastmcp import FastMCP, Context
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from .config import FastMCPRAGConfig


class EnhancedRAGServer:
    """
    Enhanced RAG Server built with FastMCP.

    This server provides semantic search, document management, and
    intelligent Q&A capabilities using FastMCP for protocol handling
    and Qdrant for vector storage.
    """

    def __init__(self, config: Optional[FastMCPRAGConfig] = None):
        """Initialize the FastMCP RAG server."""
        # Load configuration
        self.config = config or FastMCPRAGConfig()

        # Set up logging
        self._setup_logging()

        # Initialize FastMCP server
        self.mcp = FastMCP(self.config.server_name)

        # Initialize components
        self.qdrant_client = None
        self.embedding_model = None

        # Server state
        self.server_start_time = time.time()
        self.is_initialized = False

        # Initialize server components
        self._initialize_components()

        # Register tools
        self._register_tools()

        # Register custom routes (commented out as FastMCP doesn't support custom_route in current version)
        # self._register_custom_routes()

        self.logger.info("FastMCP RAG Server initialized successfully")

    def _setup_logging(self):
        """Set up logging configuration."""
        log_level = getattr(logging, self.config.log_level.upper())

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Configure root logger
        logging.basicConfig(level=log_level)

        # Set up file logging if specified
        if self.config.log_file:
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setFormatter(formatter)
            logging.getLogger().addHandler(file_handler)

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized at level: {self.config.log_level}")

    def _initialize_components(self):
        """Initialize server components (Qdrant, embedding model, etc.)."""
        try:
            # Initialize Qdrant client
            self._initialize_qdrant_client()

            # Initialize embedding model
            self._initialize_embedding_model()

            self.is_initialized = True
            self.logger.info("All server components initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize server components: {e}")
            raise

    def _initialize_qdrant_client(self):
        """Initialize Qdrant client with configuration."""
        try:
            self.qdrant_client = QdrantClient(
                url=self.config.qdrant_url,
                api_key=self.config.qdrant_api_key,
                timeout=self.config.qdrant_timeout
            )

            # Test connection
            collections = self.qdrant_client.get_collections()
            self.logger.info(f"Connected to Qdrant at {self.config.qdrant_url}")
            self.logger.info(f"Available collections: {[c.name for c in collections.collections]}")

        except Exception as e:
            self.logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def _initialize_embedding_model(self):
        """Initialize embedding model based on configuration."""
        try:
            if self.config.embedding_method == "sentence-transformers":
                self.embedding_model = SentenceTransformer(self.config.embedding_model)
                self.logger.info(f"Loaded sentence-transformers model: {self.config.embedding_model}")

                # Test embedding generation
                test_embedding = self.embedding_model.encode(["test"])
                actual_dimension = len(test_embedding[0])

                if actual_dimension != self.config.embedding_dimension:
                    self.logger.warning(
                        f"Embedding dimension mismatch: expected {self.config.embedding_dimension}, "
                        f"got {actual_dimension}"
                    )
                    # Update config to match actual dimension
                    self.config.embedding_dimension = actual_dimension

            elif self.config.embedding_method in ["dashscope", "openai"]:
                # These will be implemented in later phases
                self.logger.info(f"Embedding method {self.config.embedding_method} configured (to be implemented)")

            else:
                raise ValueError(f"Unsupported embedding method: {self.config.embedding_method}")

        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {e}")
            raise

    def _ensure_collection(self, collection_name: str):
        """Ensure collection exists in Qdrant."""
        try:
            # Try to get collection
            self.qdrant_client.get_collection(collection_name)
            self.logger.debug(f"Collection '{collection_name}' already exists")

        except Exception as e:
            # Collection doesn't exist, create it
            self.logger.info(f"Creating collection '{collection_name}'")

            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.config.embedding_dimension,
                    distance=Distance.COSINE
                )
            )

            self.logger.info(f"Collection '{collection_name}' created successfully")

    def _register_tools(self):
        """Register all FastMCP tools."""

        @self.mcp.tool()
        def add_document(
            content: str,
            title: str,
            source: str = "",
            collection: str = "documents",
            metadata: Optional[Dict] = None
        ) -> str:
            """
            Add a document to the RAG system.

            Args:
                content: Document content text
                title: Document title
                source: Document source/URL
                collection: Target collection name
                metadata: Additional metadata dictionary

            Returns:
                Success message with document details
            """
            try:
                # Validate input
                if not content or not content.strip():
                    return "Error: Document content cannot be empty"

                if not title or not title.strip():
                    return "Error: Document title cannot be empty"

                if len(content) > self.config.max_document_size:
                    return f"Error: Document size ({len(content)}) exceeds maximum allowed ({self.config.max_document_size})"

                # Ensure collection exists
                self._ensure_collection(collection)

                # Generate embedding
                if self.embedding_model is not None:
                    embedding = self.embedding_model.encode([content])[0]
                else:
                    return "Error: Embedding model not initialized"

                # Create point
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "content": content,
                        "title": title,
                        "source": source,
                        "created_at": datetime.now().isoformat(),
                        "content_length": len(content),
                        **(metadata or {})
                    }
                )

                # Store in Qdrant
                self.qdrant_client.upsert(
                    collection_name=collection,
                    points=[point]
                )

                self.logger.info(f"Document '{title}' added to collection '{collection}'")
                return f"Successfully added document '{title}' to collection '{collection}' (ID: {point.id})"

            except Exception as e:
                error_msg = f"Error adding document: {str(e)}"
                self.logger.error(error_msg)
                return error_msg

        @self.mcp.tool()
        def search_documents(
            query: str,
            collection: str = "documents",
            limit: int = 5,
            min_score: float = 0.5
        ) -> List[Dict]:
            """
            Search documents using semantic similarity.

            Args:
                query: Search query text
                collection: Collection to search in
                limit: Maximum number of results
                min_score: Minimum similarity score threshold

            Returns:
                List of search results with metadata
            """
            try:
                # Validate input
                if not query or not query.strip():
                    return []

                if self.embedding_model is None:
                    return [{"error": "Embedding model not initialized"}]

                # Ensure collection exists
                self._ensure_collection(collection)

                # Generate query embedding
                query_embedding = self.embedding_model.encode([query])[0]

                # Search in Qdrant
                results = self.qdrant_client.search(
                    collection_name=collection,
                    query_vector=query_embedding.tolist(),
                    limit=limit,
                    score_threshold=min_score,
                    with_payload=True
                )

                # Format results
                formatted_results = []
                for hit in results:
                    formatted_results.append({
                        "id": hit.id,
                        "title": hit.payload.get("title", "Untitled"),
                        "content": hit.payload["content"][:300] + "..." if len(hit.payload["content"]) > 300 else hit.payload["content"],
                        "score": hit.score,
                        "source": hit.payload.get("source", ""),
                        "created_at": hit.payload.get("created_at", ""),
                        "content_length": hit.payload.get("content_length", 0)
                    })

                self.logger.info(f"Search for '{query}' returned {len(formatted_results)} results")
                return formatted_results

            except Exception as e:
                error_msg = f"Error searching documents: {str(e)}"
                self.logger.error(error_msg)
                return [{"error": error_msg}]

        @self.mcp.tool()
        def list_collections() -> List[str]:
            """
            List all available collections in Qdrant.

            Returns:
                List of collection names
            """
            try:
                collections = self.qdrant_client.get_collections()
                collection_names = [c.name for c in collections.collections]
                self.logger.info(f"Listed {len(collection_names)} collections")
                return collection_names

            except Exception as e:
                error_msg = f"Error listing collections: {str(e)}"
                self.logger.error(error_msg)
                return []

        @self.mcp.tool()
        def get_collection_stats(collection: str = "documents") -> Dict:
            """
            Get statistics for a specific collection.

            Args:
                collection: Collection name

            Returns:
                Dictionary with collection statistics
            """
            try:
                info = self.qdrant_client.get_collection(collection)

                stats = {
                    "name": collection,
                    "vectors_count": info.vectors_count,
                    "status": str(info.status),
                    "optimizer_status": str(info.optimizer_status),
                    "points_count": info.points_count,
                    "indexed_vectors_count": info.indexed_vectors_count
                }

                self.logger.info(f"Retrieved stats for collection '{collection}'")
                return stats

            except Exception as e:
                error_msg = f"Error getting collection stats: {str(e)}"
                self.logger.error(error_msg)
                return {"error": error_msg, "name": collection}

        @self.mcp.tool()
        def delete_document(
            document_id: str,
            collection: str = "documents"
        ) -> str:
            """
            Delete a document from the RAG system.

            Args:
                document_id: Document ID to delete
                collection: Collection containing the document

            Returns:
                Success/error message
            """
            try:
                self.qdrant_client.delete(
                    collection_name=collection,
                    points_selector=[document_id]
                )

                self.logger.info(f"Document {document_id} deleted from collection '{collection}'")
                return f"Document {document_id} deleted successfully from collection '{collection}'"

            except Exception as e:
                error_msg = f"Error deleting document: {str(e)}"
                self.logger.error(error_msg)
                return error_msg

        @self.mcp.tool()
        async def intelligent_qa(
            question: str,
            collection: str = "documents",
            context_limit: int = 3,
            include_sources: bool = True,
            ctx: Context = None
        ) -> str:
            """
            Answer questions using retrieved context with LLM integration.

            Args:
                question: User question to answer
                collection: Collection to search in
                context_limit: Number of documents to retrieve
                include_sources: Whether to include source information
                ctx: FastMCP context for LLM sampling

            Returns:
                Answer string with sources (if enabled)
            """
            try:
                if not question or not question.strip():
                    return "Error: Question cannot be empty"

                # First, search for relevant documents
                search_results = search_documents(question, collection, context_limit, 0.3)

                if not search_results or isinstance(search_results[0].get("error", None), str):
                    return f"I couldn't find relevant information to answer your question about '{question}'."

                # Format context for LLM
                context_parts = []
                for i, doc in enumerate(search_results, 1):
                    context_parts.append(
                        f"Document {i}:\n"
                        f"Title: {doc['title']}\n"
                        f"Content: {doc['content']}\n"
                        f"Source: {doc['source']}\n"
                        f"Relevance Score: {doc['score']:.3f}"
                    )

                context_text = "\n\n".join(context_parts)

                # Build prompt for LLM
                prompt = f"""You are a helpful assistant that answers questions based on the provided context. Please read the following information carefully and provide a comprehensive answer to the question.

Question: {question}

Context:
{context_text}

Instructions:
1. Base your answer primarily on the provided context
2. If the context doesn't contain the information, say so clearly
3. Be specific and provide details from the context
4. Keep your answer focused and relevant to the question
5. Do not make up information not present in the context

Please provide your answer:"""

                # Log the search process
                if ctx:
                    await ctx.info(f"Retrieved {len(search_results)} documents for question: {question}")
                    await ctx.debug(f"Question: {question}")
                    await ctx.debug(f"Retrieved {len(search_results)} documents with scores: {[r['score'] for r in search_results]}")

                # Try to use LLM sampling via FastMCP context
                if ctx:
                    try:
                        response = await ctx.sample(prompt)
                        answer = response.text

                        # Add sources if requested
                        if include_sources:
                            sources = "\n\n**Sources:**\n"
                            for doc in search_results:
                                sources += f"• {doc['title']} (Score: {doc['score']:.3f}) - {doc['source']}\n"
                            answer += sources

                        return answer

                    except Exception as e:
                        self.logger.warning(f"LLM sampling failed, using fallback: {e}")
                        # Fall back to simple response

                # Fallback: Return the context directly
                fallback_response = f"Based on the retrieved information, here's what I found about '{question}':\n\n"

                for doc in search_results:
                    fallback_response += f"\nFrom '{doc['title']}':\n{doc['content']}\n"

                if include_sources:
                    fallback_response += "\n**Sources:**\n"
                    for doc in search_results:
                        fallback_response += f"• {doc['title']} (Score: {doc['score']:.3f})\n"

                return fallback_response

            except Exception as e:
                error_msg = f"Error in intelligent QA: {str(e)}"
                self.logger.error(error_msg)
                return f"I encountered an error while trying to answer your question: {error_msg}"

        @self.mcp.tool()
        def server_info() -> Dict:
            """
            Get server information and status.

            Returns:
                Dictionary with server information
            """
            uptime = time.time() - self.server_start_time

            info = {
                "server_name": self.config.server_name,
                "version": "1.0.0",
                "uptime_seconds": round(uptime, 2),
                "uptime_formatted": f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s",
                "is_initialized": self.is_initialized,
                "embedding_method": self.config.embedding_method,
                "embedding_model": self.config.embedding_model,
                "qdrant_url": self.config.qdrant_url,
                "llm_provider": self.config.llm_provider,
                "llm_model": self.config.llm_model,
                "features": {
                    "semantic_search": True,
                    "document_management": True,
                    "intelligent_qa": True,
                    "authentication": self.config.enable_auth,
                    "caching": self.config.enable_cache,
                    "metrics": self.config.enable_metrics
                }
            }

            return info

    def _register_custom_routes(self):
        """Register custom HTTP routes for the server."""

        @self.mcp.custom_route("/health", methods=["GET"])
        async def health_check():
            """
            Health check endpoint for monitoring.

            Returns:
                Health status information
            """
            try:
                # Check components
                qdrant_healthy = self.qdrant_client is not None
                embedding_healthy = self.embedding_model is not None

                overall_status = "healthy" if (qdrant_healthy and embedding_healthy and self.is_initialized) else "unhealthy"

                uptime = time.time() - self.server_start_time

                health_data = {
                    "status": overall_status,
                    "timestamp": datetime.now().isoformat(),
                    "uptime_seconds": round(uptime, 2),
                    "components": {
                        "qdrant": "healthy" if qdrant_healthy else "unhealthy",
                        "embedding_model": "healthy" if embedding_healthy else "unhealthy",
                        "server": "healthy" if self.is_initialized else "initializing"
                    }
                }

                return health_data

            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
                return {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }

        @self.mcp.custom_route("/metrics", methods=["GET"])
        async def metrics_endpoint():
            """
            Metrics endpoint for Prometheus monitoring.

            Returns:
                Basic metrics in Prometheus format
            """
            try:
                # Get collection stats
                collections = self.qdrant_client.get_collections()
                total_documents = 0

                for collection in collections.collections:
                    try:
                        info = self.qdrant_client.get_collection(collection.name)
                        total_documents += info.points_count
                    except:
                        pass

                # Generate Prometheus metrics
                metrics = [
                    f"# HELP rag_server_uptime_seconds Server uptime in seconds",
                    f"# TYPE rag_server_uptime_seconds counter",
                    f"rag_server_uptime_seconds {time.time() - self.server_start_time}",
                    "",
                    f"# HELP rag_documents_total Total number of documents",
                    f"# TYPE rag_documents_total gauge",
                    f"rag_documents_total {total_documents}",
                    "",
                    f"# HELP rag_collections_total Total number of collections",
                    f"# TYPE rag_collections_total gauge",
                    f"rag_collections_total {len(collections.collections)}"
                ]

                return "\n".join(metrics)

            except Exception as e:
                self.logger.error(f"Metrics endpoint failed: {e}")
                return f"# Error generating metrics: {str(e)}"

    def run(self, **kwargs):
        """
        Run the FastMCP server.

        Args:
            **kwargs: Additional arguments passed to FastMCP.run()
        """
        if not self.is_initialized:
            raise RuntimeError("Server not properly initialized")

        # Use config defaults if not provided
        run_kwargs = {
            "transport": self.config.transport,
            "host": self.config.host,
            "port": self.config.port,
            **kwargs
        }

        self.logger.info(f"Starting FastMCP RAG Server on {self.config.host}:{self.config.port}")
        self.logger.info(f"Transport: {self.config.transport}")
        self.logger.info(f"Embedding method: {self.config.embedding_method}")
        self.logger.info(f"Qdrant URL: {self.config.qdrant_url}")

        try:
            self.mcp.run(**run_kwargs)
        except KeyboardInterrupt:
            self.logger.info("Server stopped by user")
        except Exception as e:
            self.logger.error(f"Server error: {e}")
            raise


def create_server_from_env() -> EnhancedRAGServer:
    """
    Create EnhancedRAGServer instance from environment configuration.

    Returns:
        Configured server instance
    """
    from .config import load_config_from_env

    config = load_config_from_env()
    return EnhancedRAGServer(config)


def main():
    """Main entry point for the FastMCP RAG server."""
    import argparse

    parser = argparse.ArgumentParser(description="FastMCP Enhanced Semantic Search RAG Server")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--host", default=None, help="Server host (overrides config)")
    parser.add_argument("--port", type=int, default=None, help="Server port (overrides config)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    try:
        # Load configuration
        if args.config:
            # Load from file (to be implemented)
            config = FastMCPRAGConfig()
        else:
            config = load_config_from_env()

        # Override with command line arguments
        if args.host:
            config.host = args.host
        if args.port:
            config.port = args.port
        if args.debug:
            config.debug = True
            config.log_level = "DEBUG"

        # Create and run server
        server = EnhancedRAGServer(config)
        server.run()

    except Exception as e:
        print(f"Failed to start server: {e}")
        if "--debug" in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()