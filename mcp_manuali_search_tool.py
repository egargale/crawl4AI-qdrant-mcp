#!/usr/bin/env python3
"""
MCP tool for searching the 'manuali' collection for code examples and functions.
"""

import os
import sys
from typing import List, Optional
from dotenv import load_dotenv

# Add the mcp-server-qdrant src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mcp-server-qdrant', 'src'))

from qdrant_client import QdrantClient
from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider


def safe_print(text):
    """Safely print text with Unicode encoding handling."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback to ASCII encoding with error handling
        print(text.encode('ascii', errors='ignore').decode('ascii'))


class ManualiSearchTool:
    """Tool for searching the 'manuali' collection for code examples and functions."""
    
    def __init__(self):
        """Initialize the search tool."""
        load_dotenv()
        self.collection_name = "manuali"
        self.client = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))
        self.embedding_provider = FastEmbedProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    async def search_code_examples(self, query: str, limit: int = 5) -> List[dict]:
        """
        Search the 'manuali' collection for code examples and functions.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of search results with content and metadata
        """
        try:
            # Generate embedding for the query
            query_vector = await self.embedding_provider.embed_query(query)
            
            # Query the collection
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True
            )
            
            # Format results
            results = []
            for point in search_result.points:
                result = {
                    'id': point.id,
                    'score': point.score,
                    'content': '',
                    'source': '',
                    'metadata': {}
                }
                
                if hasattr(point, 'payload') and point.payload:
                    # Extract document content
                    if 'document' in point.payload:
                        result['content'] = str(point.payload['document'])
                    
                    # Extract source
                    if 'source' in point.payload:
                        result['source'] = point.payload['source']
                    
                    # Extract other metadata
                    result['metadata'] = {k: v for k, v in point.payload.items() 
                                        if k not in ['document', 'source']}
                
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error searching collection: {e}")
            return []
    
    def format_results(self, results: List[dict], max_content_length: int = 300) -> str:
        """
        Format search results for display.
        
        Args:
            results: List of search results
            max_content_length: Maximum length of content to display
            
        Returns:
            Formatted string of results
        """
        if not results:
            return "No results found."
        
        output = [f"Found {len(results)} results:"]
        
        for i, result in enumerate(results, 1):
            output.append(f"\n--- Result {i} ---")
            output.append(f"ID: {result['id']}")
            output.append(f"Score: {result['score']:.4f}")
            output.append(f"Source: {result['source']}")
            
            # Format content
            content = result['content']
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            # Handle Unicode safely
            try:
                clean_content = content.encode('utf-8', errors='ignore').decode('utf-8')
                output.append(f"Content: {clean_content}")
            except Exception:
                output.append(f"Content: {content[:100]}...")
            
            # Add metadata if present
            if result['metadata']:
                output.append(f"Metadata: {result['metadata']}")
        
        return "\n".join(output)


# Example usage as an MCP tool
async def search_manuali_collection(
    query: str,
    limit: int = 5,
    content_type: str = "code"
) -> str:
    """
    Search the 'manuali' collection for code examples and functions.
    
    Args:
        query: Search query string
        limit: Maximum number of results to return (default: 5)
        content_type: Type of content to search for (default: "code")
        
    Returns:
        Formatted search results
    """
    # Adjust query based on content type
    if content_type == "code":
        query = f"code example {query}"
    elif content_type == "function":
        query = f"function definition {query}"
    elif content_type == "api":
        query = f"API usage {query}"
    
    # Create search tool
    search_tool = ManualiSearchTool()
    
    # Perform search
    results = await search_tool.search_code_examples(query, limit)
    
    # Format and return results
    return search_tool.format_results(results)


# Standalone script execution
if __name__ == "__main__":
    import asyncio
    import argparse
    
    parser = argparse.ArgumentParser(description="Search the 'manuali' collection for code examples and functions")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    parser.add_argument("--type", choices=["code", "function", "api"], default="code", 
                       help="Type of content to search for")
    
    args = parser.parse_args()
    
    async def main():
        results = await search_manuali_collection(args.query, args.limit, args.type)
        safe_print(results)
    
    asyncio.run(main())