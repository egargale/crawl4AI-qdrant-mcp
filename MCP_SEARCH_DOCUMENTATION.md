# Using MCP Server to Search Code Examples and Functions

## Overview

This document explains how to use MCP (Model Context Protocol) server tools to search for code examples and functions in the "manuali" collection.

## Discovering Available Tools

To discover available tools from your MCP server:

1. **In Qwen Code**, use the `/mcp` command to see a list of available tools
2. Look for tools related to searching or querying documentation
3. The `search_manuali_collection` tool should appear in the list

## Using the Search Tool

### Basic Usage

Once discovered, you can use the search tool by:

1. Selecting the `search_manuali_collection` tool
2. Providing a query parameter with your search term
3. Optionally specifying the number of results to return

### Example Queries

Here are some example queries you can use:

- **Code Examples**: "FastMCP tool example", "function decorator usage"
- **API Usage**: "API endpoint definition", "authentication setup"
- **Configuration**: "OAuth provider configuration", "server initialization"

### Parameters

The search tool accepts the following parameters:

- `query` (required): Search query string
- `limit` (optional, default: 5): Maximum number of results to return
- `content_type` (optional, default: "code"): Type of content to search for ("code", "function", or "api")

## Predefined Prompts

The MCP server may expose predefined prompts as slash commands:

- `/search-code`: Search for code examples
- `/search-functions`: Search for function definitions
- `/search-api`: Search for API usage examples

## Example Results

When you search, you'll get results like:

```
Found 3 results:

--- Result 1 ---
ID: 223
Score: 0.1813
Source: https://gofastmcp.com/patterns/cli
Content: # Install with specific Python version
fastmcp install claude-desktop server.py --python 3.11

--- Result 2 ---
ID: 93
Score: 0.1730
Source: https://gofastmcp.com/integrations/claude-desktop
Content: # Create a proxy to a remote server
proxy = FastMCP.as_proxy(
  "https://example.com/mcp/sse", 
  name="Remote Server Proxy"
)
```

## Tips for Effective Searches

1. **Be Specific**: Use specific terms like "FastMCP tool" rather than generic terms
2. **Use Code-related Terms**: Include terms like "example", "function", "API", "implementation"
3. **Check Multiple Sources**: The "manuali" collection contains documentation from multiple sources
4. **Adjust Result Count**: Use the `limit` parameter to get more or fewer results

## Troubleshooting

If you don't see the search tool:

1. Ensure your MCP server is properly configured
2. Check that the `mcp-server-qdrant` component is running
3. Verify that the "manuali" collection exists in Qdrant
4. Confirm that your environment variables are set correctly