#!/usr/bin/env python3
"""
Predefined prompts for common search queries in the MCP server.
These can be exposed as slash commands in Qwen Code.
"""

from typing import List, Dict, Any
import asyncio

# Import our search tool
from mcp_manuali_search_tool import search_manuali_collection


class ManualiSearchPrompts:
    """Predefined prompts for searching the manuali collection."""
    
    @staticmethod
    async def search_code_examples(query: str = "", limit: int = 5) -> str:
        """
        Search for code examples in the manuali collection.
        
        Args:
            query: Specific code topic to search for
            limit: Number of results to return
            
        Returns:
            Formatted search results
        """
        search_query = f"code example {query}" if query else "code example"
        return await search_manuali_collection(search_query, limit, "code")
    
    @staticmethod
    async def search_function_definitions(query: str = "", limit: int = 5) -> str:
        """
        Search for function definitions in the manuali collection.
        
        Args:
            query: Specific function or method to search for
            limit: Number of results to return
            
        Returns:
            Formatted search results
        """
        search_query = f"function definition {query}" if query else "function definition"
        return await search_manuali_collection(search_query, limit, "function")
    
    @staticmethod
    async def search_api_usage(query: str = "", limit: int = 5) -> str:
        """
        Search for API usage examples in the manuali collection.
        
        Args:
            query: Specific API or endpoint to search for
            limit: Number of results to return
            
        Returns:
            Formatted search results
        """
        search_query = f"API usage {query}" if query else "API usage"
        return await search_manuali_collection(search_query, limit, "api")
    
    @staticmethod
    async def search_oauth_providers(provider: str = "", limit: int = 5) -> str:
        """
        Search for OAuth provider configuration examples.
        
        Args:
            provider: Specific OAuth provider (e.g., "GitHub", "Google")
            limit: Number of results to return
            
        Returns:
            Formatted search results
        """
        search_query = f"OAuth provider {provider}" if provider else "OAuth provider"
        return await search_manuali_collection(search_query, limit, "code")
    
    @staticmethod
    async def search_fastmcp_tools(tool_name: str = "", limit: int = 5) -> str:
        """
        Search for FastMCP tool examples.
        
        Args:
            tool_name: Specific tool name to search for
            limit: Number of results to return
            
        Returns:
            Formatted search results
        """
        search_query = f"FastMCP tool {tool_name}" if tool_name else "FastMCP tool"
        return await search_manuali_collection(search_query, limit, "code")


# Example MCP server integration
def create_mcp_server_with_prompts():
    """
    Example of how to integrate these prompts into an MCP server.
    This would typically be part of your main MCP server setup.
    """
    try:
        from mcp_server_qdrant.mcp_server import QdrantMCPServer
        from mcp_server_qdrant.settings import (
            EmbeddingProviderSettings,
            QdrantSettings,
            ToolSettings,
        )
        
        # Create MCP server
        mcp = QdrantMCPServer(
            tool_settings=ToolSettings(),
            qdrant_settings=QdrantSettings(),
            embedding_provider_settings=EmbeddingProviderSettings(),
        )
        
        # Register predefined prompts as tools
        mcp.tool(
            ManualiSearchPrompts.search_code_examples,
            name="search-code-examples",
            description="Search for code examples in the manuali collection"
        )
        
        mcp.tool(
            ManualiSearchPrompts.search_function_definitions,
            name="search-function-definitions",
            description="Search for function definitions in the manuali collection"
        )
        
        mcp.tool(
            ManualiSearchPrompts.search_api_usage,
            name="search-api-usage",
            description="Search for API usage examples in the manuali collection"
        )
        
        mcp.tool(
            ManualiSearchPrompts.search_oauth_providers,
            name="search-oauth-providers",
            description="Search for OAuth provider configuration examples"
        )
        
        mcp.tool(
            ManualiSearchPrompts.search_fastmcp_tools,
            name="search-fastmcp-tools",
            description="Search for FastMCP tool examples"
        )
        
        return mcp
        
    except ImportError:
        print("MCP server components not available")
        return None


# Standalone execution for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test predefined search prompts")
    parser.add_argument("prompt_type", choices=["code", "function", "api", "oauth", "tool"], 
                       help="Type of prompt to test")
    parser.add_argument("query", nargs="?", default="", help="Search query")
    parser.add_argument("--limit", type=int, default=3, help="Number of results")
    
    args = parser.parse_args()
    
    async def main():
        if args.prompt_type == "code":
            results = await ManualiSearchPrompts.search_code_examples(args.query, args.limit)
        elif args.prompt_type == "function":
            results = await ManualiSearchPrompts.search_function_definitions(args.query, args.limit)
        elif args.prompt_type == "api":
            results = await ManualiSearchPrompts.search_api_usage(args.query, args.limit)
        elif args.prompt_type == "oauth":
            results = await ManualiSearchPrompts.search_oauth_providers(args.query, args.limit)
        elif args.prompt_type == "tool":
            results = await ManualiSearchPrompts.search_fastmcp_tools(args.query, args.limit)
        
        print(results)
    
    asyncio.run(main())