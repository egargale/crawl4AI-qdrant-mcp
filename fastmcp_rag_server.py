#!/usr/bin/env python3
"""
FastMCP Enhanced Semantic Search RAG Server

This is the main entry point for the FastMCP-based RAG server.
Run this script to start the server with default or custom configuration.
"""

import sys
import os
import argparse
import logging

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastmcp_rag import FastMCPRAGConfig, EnhancedRAGServer
from fastmcp_rag.config.settings import load_config_from_env


def setup_logging(debug: bool = False):
    """Set up basic logging for the application."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main entry point for the FastMCP RAG server."""
    parser = argparse.ArgumentParser(
        description="FastMCP Enhanced Semantic Search RAG Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with default configuration
  python fastmcp_rag_server.py

  # Start with custom host and port
  python fastmcp_rag_server.py --host 127.0.0.1 --port 8080

  # Start with debug mode
  python fastmcp_rag_server.py --debug

  # Start with stdio transport (for MCP clients)
  python fastmcp_rag_server.py --transport stdio

  # Start with SSE transport (Server-Sent Events)
  python fastmcp_rag_server.py --transport sse

Environment Variables:
  See .env.fastmcp.template for available configuration options.
  Key variables:
    QDRANT_URL              - Qdrant server URL (required)
    EMBEDDING_METHOD        - sentence-transformers, dashscope, openai
    DASHSCOPE_API_KEY       - For DashScope embeddings/LLM
    OPENAI_API_KEY          - For OpenAI embeddings/LLM
    ENABLE_AUTH             - Enable authentication
    ENABLE_CACHE            - Enable Redis caching
        """
    )

    # Server configuration arguments
    parser.add_argument("--host", default=None, help="Server host address (overrides config)")
    parser.add_argument("--port", type=int, default=None, help="Server port (overrides config)")
    parser.add_argument("--transport", choices=["stdio", "sse"],
                       default=None, help="Transport protocol (overrides config)")

    # Feature flags
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--no-auth", action="store_true", help="Disable authentication")
    parser.add_argument("--no-metrics", action="store_true", help="Disable metrics")

    # LLM and embedding options
    parser.add_argument("--embedding-method",
                       choices=["sentence-transformers", "dashscope", "openai"],
                       default=None, help="Embedding method")
    parser.add_argument("--llm-provider", choices=["dashscope", "openai"],
                       default=None, help="LLM provider for Q&A")

    # Configuration file
    parser.add_argument("--config", help="Path to configuration file (future feature)")

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.debug)

    logger = logging.getLogger(__name__)

    try:
        # Load configuration from environment
        config = load_config_from_env()

        # Override configuration with command line arguments
        if args.host:
            config.host = args.host
        if args.port:
            config.port = args.port
        if args.transport:
            config.transport = args.transport
        if args.debug:
            config.debug = True
            config.log_level = "DEBUG"
        if args.no_cache:
            config.enable_cache = False
        if args.no_auth:
            config.enable_auth = False
        if args.no_metrics:
            config.enable_metrics = False
        if args.embedding_method:
            config.embedding_method = args.embedding_method
        if args.llm_provider:
            config.llm_provider = args.llm_provider

        # Validate configuration
        missing_config = config.validate_required_config()
        if missing_config:
            logger.error("Configuration errors:")
            for error in missing_config:
                logger.error(f"  - {error}")
            logger.error("Please set the required environment variables or configuration.")
            sys.exit(1)

        # Create server instance
        logger.info("Initializing FastMCP RAG Server...")
        logger.info(f"Configuration:")
        logger.info(f"  - Host: {config.host}")
        logger.info(f"  - Port: {config.port}")
        logger.info(f"  - Transport: {config.transport}")
        logger.info(f"  - Embedding Method: {config.embedding_method}")
        logger.info(f"  - Embedding Model: {config.embedding_model}")
        logger.info(f"  - LLM Provider: {config.llm_provider}")
        logger.info(f"  - Qdrant URL: {config.qdrant_url}")
        logger.info(f"  - Authentication: {config.enable_auth}")
        logger.info(f"  - Caching: {config.enable_cache}")
        logger.info(f"  - Metrics: {config.enable_metrics}")

        server = EnhancedRAGServer(config)

        # Start server
        logger.info("Starting FastMCP RAG Server...")
        server.run()

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()