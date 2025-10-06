"""
FastMCP-based Enhanced Semantic Search RAG System

This package provides a production-ready semantic search RAG system
built with FastMCP, integrating with Qdrant, sentence-transformers,
and existing crawl4ai infrastructure.
"""

__version__ = "1.0.0"
__author__ = "Enhanced RAG Team"

from .config import FastMCPRAGConfig
from .server import EnhancedRAGServer

__all__ = ["FastMCPRAGConfig", "EnhancedRAGServer"]