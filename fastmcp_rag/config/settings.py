"""
FastMCP RAG Configuration Settings

This module provides comprehensive configuration management for the
FastMCP-based semantic search RAG system using Pydantic settings.
"""

import os
from typing import Dict, List, Optional
from pydantic import Field, field_validator, ConfigDict
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class FastMCPRAGConfig(BaseSettings):
    """
    Comprehensive configuration for FastMCP RAG system.

    This configuration class manages all settings for the enhanced
    semantic search RAG server, including database connections,
    embedding models, authentication, and performance tuning.
    """

    # Server Configuration
    server_name: str = Field(default="Enhanced Semantic Search RAG", description="Server name for identification")
    host: str = Field(default="0.0.0.0", description="Server host address")
    port: int = Field(default=8000, description="Server port")
    transport: str = Field(default="http", description="Transport protocol (http, stdio)")
    debug: bool = Field(default=False, description="Enable debug mode")

    # Database Configuration
    qdrant_url: str = Field(..., description="Qdrant server URL")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant API key (optional for local)")
    qdrant_timeout: int = Field(default=30, description="Qdrant connection timeout in seconds")

    # Embedding Configuration
    embedding_method: str = Field(
        default="sentence-transformers",
        description="Embedding method: sentence-transformers, dashscope, openai"
    )
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Embedding model name")
    embedding_dimension: int = Field(default=384, description="Embedding vector dimension")
    embedding_batch_size: int = Field(default=32, description="Batch size for embedding generation")

    # DashScope Configuration (if using)
    dashscope_api_key: Optional[str] = Field(default=None, description="DashScope API key")
    dashscope_url: str = Field(
        default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        description="DashScope API URL"
    )

    # OpenAI Configuration (if using)
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_base_url: str = Field(default="https://api.openai.com/v1", description="OpenAI API base URL")

    # LLM Configuration for Q&A
    llm_provider: str = Field(default="dashscope", description="LLM provider: dashscope, openai")
    llm_model: str = Field(default="qwen-turbo", description="LLM model name")
    llm_temperature: float = Field(default=0.7, description="LLM temperature for generation")
    llm_max_tokens: int = Field(default=2000, description="Maximum tokens for LLM generation")

    # Search Configuration
    default_top_k: int = Field(default=5, description="Default number of search results")
    default_similarity_threshold: float = Field(default=0.5, description="Default similarity threshold")
    enable_hybrid_search: bool = Field(default=True, description="Enable hybrid search capabilities")
    enable_diversity_ranking: bool = Field(default=True, description="Enable result diversity ranking")
    enable_query_expansion: bool = Field(default=True, description="Enable query expansion")

    # Document Processing Configuration
    default_chunk_size: int = Field(default=1000, description="Default chunk size for documents")
    default_chunk_overlap: int = Field(default=200, description="Default chunk overlap")
    enable_semantic_chunking: bool = Field(default=True, description="Enable semantic chunking")
    enable_metadata_enrichment: bool = Field(default=True, description="Enable automatic metadata enrichment")
    max_document_size: int = Field(default=1000000, description="Maximum document size in characters")

    # Authentication Configuration
    enable_auth: bool = Field(default=False, description="Enable authentication")
    auth_provider: str = Field(default="github", description="Auth provider: github, google, azure, auth0")
    github_client_id: Optional[str] = Field(default=None, description="GitHub OAuth client ID")
    github_client_secret: Optional[str] = Field(default=None, description="GitHub OAuth client secret")
    google_client_id: Optional[str] = Field(default=None, description="Google OAuth client ID")
    google_client_secret: Optional[str] = Field(default=None, description="Google OAuth client secret")

    # Caching Configuration
    enable_cache: bool = Field(default=True, description="Enable result caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    redis_url: str = Field(default="redis://localhost:6379", description="Redis URL for caching")

    # Performance Configuration
    max_concurrent_requests: int = Field(default=10, description="Maximum concurrent requests")
    request_timeout: int = Field(default=30, description="Request timeout in seconds")
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(default=9090, description="Prometheus metrics port")

    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default=None, description="Log file path (optional)")

    model_config = ConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore"  # Ignore extra fields from environment
    )

    @field_validator("embedding_method")
    @classmethod
    def validate_embedding_method(cls, v):
        """Validate embedding method."""
        allowed_methods = ["sentence-transformers", "dashscope", "openai"]
        if v not in allowed_methods:
            raise ValueError(f"embedding_method must be one of: {allowed_methods}")
        return v

    @field_validator("transport")
    @classmethod
    def validate_transport(cls, v):
        """Validate transport protocol."""
        allowed_transports = ["http", "stdio", "websocket"]
        if v not in allowed_transports:
            raise ValueError(f"transport must be one of: {allowed_transports}")
        return v

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, v):
        """Validate LLM provider."""
        allowed_providers = ["dashscope", "openai"]
        if v not in allowed_providers:
            raise ValueError(f"llm_provider must be one of: {allowed_providers}")
        return v

    @field_validator("auth_provider")
    @classmethod
    def validate_auth_provider(cls, v):
        """Validate authentication provider."""
        allowed_providers = ["github", "google", "azure", "auth0"]
        if v not in allowed_providers:
            raise ValueError(f"auth_provider must be one of: {allowed_providers}")
        return v

    @field_validator("embedding_dimension")
    @classmethod
    def validate_embedding_dimension(cls, v, info):
        """Validate embedding dimension based on model."""
        model = info.data.get("embedding_model", "")

        # Known model dimensions
        model_dimensions = {
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-v4": 1536,
        }

        if model in model_dimensions and v != model_dimensions[model]:
            print(f"Warning: Embedding dimension {v} may not match model '{model}' expected dimension {model_dimensions[model]}")

        return v

    def get_embedding_config(self) -> Dict:
        """Get embedding configuration as dictionary."""
        if self.embedding_method == "sentence-transformers":
            return {
                "method": "sentence-transformers",
                "model": self.embedding_model,
                "dimension": self.embedding_dimension,
                "batch_size": self.embedding_batch_size
            }
        elif self.embedding_method == "dashscope":
            return {
                "method": "dashscope",
                "api_key": self.dashscope_api_key,
                "url": self.dashscope_url,
                "model": self.embedding_model,
                "dimension": self.embedding_dimension
            }
        elif self.embedding_method == "openai":
            return {
                "method": "openai",
                "api_key": self.openai_api_key,
                "base_url": self.openai_base_url,
                "model": self.embedding_model,
                "dimension": self.embedding_dimension
            }

    def get_llm_config(self) -> Dict:
        """Get LLM configuration as dictionary."""
        if self.llm_provider == "dashscope":
            return {
                "provider": "dashscope",
                "api_key": self.dashscope_api_key,
                "base_url": self.dashscope_url,
                "model": self.llm_model,
                "temperature": self.llm_temperature,
                "max_tokens": self.llm_max_tokens
            }
        elif self.llm_provider == "openai":
            return {
                "provider": "openai",
                "api_key": self.openai_api_key,
                "base_url": self.openai_base_url,
                "model": self.llm_model,
                "temperature": self.llm_temperature,
                "max_tokens": self.llm_max_tokens
            }

    def validate_required_config(self) -> List[str]:
        """Validate required configuration and return list of missing items."""
        missing = []

        # Check required based on configuration
        if self.embedding_method == "dashscope" and not self.dashscope_api_key:
            missing.append("dashscope_api_key required for DashScope embeddings")

        if self.embedding_method == "openai" and not self.openai_api_key:
            missing.append("openai_api_key required for OpenAI embeddings")

        if self.llm_provider == "dashscope" and not self.dashscope_api_key:
            missing.append("dashscope_api_key required for DashScope LLM")

        if self.llm_provider == "openai" and not self.openai_api_key:
            missing.append("openai_api_key required for OpenAI LLM")

        if self.enable_auth:
            if self.auth_provider == "github":
                if not self.github_client_id or not self.github_client_secret:
                    missing.append("github_client_id and github_client_secret required for GitHub auth")
            elif self.auth_provider == "google":
                if not self.google_client_id or not self.google_client_secret:
                    missing.append("google_client_id and google_client_secret required for Google auth")

        return missing


def create_default_config() -> FastMCPRAGConfig:
    """Create default configuration for development."""
    return FastMCPRAGConfig(
        qdrant_url="http://localhost:6333",
        embedding_method="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        llm_provider="dashscope",
        enable_auth=False,
        enable_cache=False,
        debug=True
    )


def load_config_from_env() -> FastMCPRAGConfig:
    """Load configuration from environment variables."""
    try:
        config = FastMCPRAGConfig()

        # Validate required configuration
        missing = config.validate_required_config()
        if missing:
            print("Configuration warnings:")
            for warning in missing:
                print(f"  - {warning}")

        return config

    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Using default configuration for development...")
        return create_default_config()