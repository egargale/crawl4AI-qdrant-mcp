"""
Tests for FastMCP RAG Server core functionality.

These tests verify the basic functionality of the FastMCP RAG server
including document management, search, and configuration.
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock

from fastmcp_rag.server import EnhancedRAGServer
from fastmcp_rag.config import FastMCPRAGConfig


class TestFastMCPRAGConfig:
    """Test FastMCP RAG configuration."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = FastMCPRAGConfig(
            qdrant_url="http://localhost:6333"
        )

        assert config.server_name == "Enhanced Semantic Search RAG"
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.embedding_method == "sentence-transformers"
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.default_top_k == 5
        assert config.enable_auth is False

    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid embedding method
        with pytest.raises(ValueError):
            FastMCPRAGConfig(
                qdrant_url="http://localhost:6333",
                embedding_method="invalid_method"
            )

        # Test invalid transport
        with pytest.raises(ValueError):
            FastMCPRAGConfig(
                qdrant_url="http://localhost:6333",
                transport="invalid_transport"
            )

    def test_embedding_config(self):
        """Test embedding configuration extraction."""
        config = FastMCPRAGConfig(
            qdrant_url="http://localhost:6333",
            embedding_method="sentence-transformers",
            embedding_model="all-MiniLM-L6-v2"
        )

        embedding_config = config.get_embedding_config()
        assert embedding_config["method"] == "sentence-transformers"
        assert embedding_config["model"] == "all-MiniLM-L6-v2"
        assert embedding_config["dimension"] == 384

    def test_llm_config(self):
        """Test LLM configuration extraction."""
        config = FastMCPRAGConfig(
            qdrant_url="http://localhost:6333",
            llm_provider="dashscope",
            llm_model="qwen-turbo"
        )

        llm_config = config.get_llm_config()
        assert llm_config["provider"] == "dashscope"
        assert llm_config["model"] == "qwen-turbo"


class TestEnhancedRAGServer:
    """Test Enhanced RAG Server functionality."""

    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        return FastMCPRAGConfig(
            qdrant_url="http://localhost:6333",
            embedding_method="sentence-transformers",
            embedding_model="all-MiniLM-L6-v2",
            enable_auth=False,
            enable_cache=False,
            debug=True
        )

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create mock Qdrant client."""
        mock_client = Mock()
        mock_client.get_collections.return_value.collections = []
        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_client.create_collection.return_value = None
        mock_client.upsert.return_value = None
        mock_client.search.return_value = []
        return mock_client

    @pytest.fixture
    def mock_embedding_model(self):
        """Create mock embedding model."""
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]  # Mock embedding
        return mock_model

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create mock Qdrant client."""
        mock_client = Mock()
        # Mock the get_collections response
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        # Mock collection_exists to return False by default
        mock_client.collection_exists.return_value = False
        # Mock create_collection
        mock_client.create_collection.return_value = None
        # Mock upsert
        mock_client.upsert.return_value = None
        # Mock search
        mock_client.search.return_value = []
        return mock_client

    @patch('fastmcp_rag.server.QdrantClient')
    @patch('fastmcp_rag.server.SentenceTransformer')
    def test_server_initialization(self, mock_transformer, mock_qdrant_class, test_config, mock_qdrant_client):
        """Test server initialization."""
        # Setup mocks
        mock_qdrant_class.return_value = mock_qdrant_client
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_transformer.return_value = mock_model

        # Create server
        server = EnhancedRAGServer(test_config)

        # Verify initialization
        assert server.config == test_config
        assert server.qdrant_client == mock_qdrant_client
        assert server.embedding_model == mock_model
        assert server.is_initialized is True

        # Verify Qdrant client was initialized correctly
        mock_qdrant_class.assert_called_once_with(
            url=test_config.qdrant_url,
            api_key=test_config.qdrant_api_key,
            timeout=test_config.qdrant_timeout
        )

        # Verify embedding model was loaded
        mock_transformer.assert_called_once_with(test_config.embedding_model)

    @patch('fastmcp_rag.server.QdrantClient')
    @patch('fastmcp_rag.server.SentenceTransformer')
    def test_add_document_tool(self, mock_transformer, mock_qdrant_class, test_config):
        """Test add_document tool functionality."""
        # Setup mocks
        mock_client = Mock()
        mock_qdrant_class.return_value = mock_client
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_transformer.return_value = mock_model

        # Create server
        server = EnhancedRAGServer(test_config)

        # Get the add_document tool
        add_document_tool = None
        for tool in server.mcp._tools:
            if tool.name == "add_document":
                add_document_tool = tool
                break

        assert add_document_tool is not None

        # Test adding a document
        result = add_document_tool.call({
            "content": "This is a test document about FastMCP.",
            "title": "Test Document",
            "source": "test_source"
        })

        assert "Successfully added document" in result
        assert "Test Document" in result

        # Verify Qdrant upsert was called
        mock_client.upsert.assert_called_once()

    @patch('fastmcp_rag.server.QdrantClient')
    @patch('fastmcp_rag.server.SentenceTransformer')
    def test_search_documents_tool(self, mock_transformer, mock_qdrant_class, test_config):
        """Test search_documents tool functionality."""
        # Setup mocks
        mock_client = Mock()
        mock_qdrant_class.return_value = mock_client
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_transformer.return_value = mock_model

        # Mock search results
        mock_hit = Mock()
        mock_hit.id = "test_id"
        mock_hit.score = 0.8
        mock_hit.payload = {
            "title": "Test Document",
            "content": "This is test content about FastMCP framework.",
            "source": "test_source",
            "created_at": "2024-01-01T00:00:00",
            "content_length": 50
        }
        mock_client.search.return_value = [mock_hit]

        # Create server
        server = EnhancedRAGServer(test_config)

        # Get the search_documents tool
        search_tool = None
        for tool in server.mcp._tools:
            if tool.name == "search_documents":
                search_tool = tool
                break

        assert search_tool is not None

        # Test searching
        results = search_tool.call({
            "query": "FastMCP framework",
            "collection": "documents",
            "limit": 5,
            "min_score": 0.5
        })

        assert len(results) == 1
        assert results[0]["title"] == "Test Document"
        assert results[0]["score"] == 0.8
        assert "FastMCP framework" in results[0]["content"]

        # Verify Qdrant search was called
        mock_client.search.assert_called_once()

    @patch('fastmcp_rag.server.QdrantClient')
    @patch('fastmcp_rag.server.SentenceTransformer')
    def test_list_collections_tool(self, mock_transformer, mock_qdrant_class, test_config):
        """Test list_collections tool functionality."""
        # Setup mocks
        mock_client = Mock()
        mock_qdrant_class.return_value = mock_client
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_transformer.return_value = mock_model

        # Mock collections
        mock_collection1 = Mock()
        mock_collection1.name = "documents"
        mock_collection2 = Mock()
        mock_collection2.name = "test_collection"
        mock_client.get_collections.return_value.collections = [mock_collection1, mock_collection2]

        # Create server
        server = EnhancedRAGServer(test_config)

        # Get the list_collections tool
        list_tool = None
        for tool in server.mcp._tools:
            if tool.name == "list_collections":
                list_tool = tool
                break

        assert list_tool is not None

        # Test listing collections
        collections = list_tool.call({})

        assert len(collections) == 2
        assert "documents" in collections
        assert "test_collection" in collections

    @patch('fastmcp_rag.server.QdrantClient')
    @patch('fastmcp_rag.server.SentenceTransformer')
    def test_server_info_tool(self, mock_transformer, mock_qdrant_class, test_config, mock_qdrant_client):
        """Test server_info tool functionality."""
        # Setup mocks
        mock_qdrant_class.return_value = mock_qdrant_client
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_transformer.return_value = mock_model

        # Create server
        server = EnhancedRAGServer(test_config)

        # Get the server_info tool
        info_tool = None
        for tool in server.mcp._tools:
            if tool.name == "server_info":
                info_tool = tool
                break

        assert info_tool is not None

        # Test getting server info
        info = info_tool.call({})

        assert info["server_name"] == test_config.server_name
        assert info["embedding_method"] == test_config.embedding_method
        assert info["embedding_model"] == test_config.embedding_model
        assert info["qdrant_url"] == test_config.qdrant_url
        assert info["features"]["semantic_search"] is True
        assert info["features"]["document_management"] is True
        assert info["features"]["intelligent_qa"] is True
        assert info["is_initialized"] is True

    @patch('fastmcp_rag.server.QdrantClient')
    @patch('fastmcp_rag.server.SentenceTransformer')
    def test_intelligent_qa_tool_fallback(self, mock_transformer, mock_qdrant_class, test_config):
        """Test intelligent_qa tool fallback functionality."""
        # Setup mocks
        mock_client = Mock()
        mock_qdrant_class.return_value = mock_client
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_transformer.return_value = mock_model

        # Mock search results
        mock_hit = Mock()
        mock_hit.id = "test_id"
        mock_hit.score = 0.8
        mock_hit.payload = {
            "title": "Test Document",
            "content": "FastMCP is a framework for building MCP servers.",
            "source": "test_source",
            "created_at": "2024-01-01T00:00:00",
            "content_length": 50
        }
        mock_client.search.return_value = [mock_hit]

        # Create server
        server = EnhancedRAGServer(test_config)

        # Get the intelligent_qa tool
        qa_tool = None
        for tool in server.mcp._tools:
            if tool.name == "intelligent_qa":
                qa_tool = tool
                break

        assert qa_tool is not None

        # Test Q&A without context (fallback mode)
        result = qa_tool.call({
            "question": "What is FastMCP?",
            "collection": "documents",
            "context_limit": 3,
            "include_sources": True
        })

        assert "Based on the retrieved information" in result
        assert "FastMCP" in result
        assert "Sources:" in result


class TestServerHealth:
    """Test server health and metrics endpoints."""

    @patch('fastmcp_rag.server.QdrantClient')
    @patch('fastmcp_rag.server.SentenceTransformer')
    def test_health_check(self, mock_transformer, mock_qdrant_class):
        """Test health check endpoint."""
        config = FastMCPRAGConfig(
            qdrant_url="http://localhost:6333",
            embedding_method="sentence-transformers"
        )

        # Setup mocks
        mock_client = Mock()
        mock_qdrant_class.return_value = mock_client
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_transformer.return_value = mock_model

        # Create server
        server = EnhancedRAGServer(config)

        # Find health check route
        health_route = None
        for route in server.mcp._custom_routes:
            if route[0] == "/health":
                health_route = route[1]
                break

        assert health_route is not None

        # Test health check
        health_data = asyncio.run(health_route())

        assert health_data["status"] == "healthy"
        assert "timestamp" in health_data
        assert "uptime_seconds" in health_data
        assert "components" in health_data
        assert health_data["components"]["qdrant"] == "healthy"
        assert health_data["components"]["embedding_model"] == "healthy"


@pytest.mark.asyncio
async def test_end_to_end_workflow():
    """Test end-to-end workflow with real components (if available)."""
    # This test would require actual Qdrant and embedding model
    # For now, we'll skip it in CI environments
    if not os.getenv("RUN_INTEGRATION_TESTS"):
        pytest.skip("Integration tests disabled")

    config = FastMCPRAGConfig(
        qdrant_url="http://localhost:6333",
        embedding_method="sentence-transformers",
        enable_cache=False
    )

    try:
        server = EnhancedRAGServer(config)

        # Test adding a document
        add_result = server.mcp._tools[0].call({
            "content": "FastMCP is a Python framework for building MCP servers with enterprise features.",
            "title": "FastMCP Overview",
            "source": "test"
        })
        assert "Successfully added" in add_result

        # Test searching
        search_results = server.mcp._tools[1].call({
            "query": "FastMCP framework",
            "limit": 5
        })
        assert len(search_results) >= 0

        # Test Q&A
        qa_result = await server.mcp._tools[6].call({
            "question": "What is FastMCP?",
            "context_limit": 3
        })
        assert len(qa_result) > 0

    except Exception as e:
        # If services are not available, skip the test
        if "connection" in str(e).lower() or "timeout" in str(e).lower():
            pytest.skip(f"Integration test skipped: {e}")
        else:
            raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])