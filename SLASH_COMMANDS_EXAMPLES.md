# Slash Commands for Code Searches

This document provides examples of slash commands that can be used in Qwen Code to search for code examples and functions in the "manuali" collection.

## Code Search Commands

### /search-code
Search for general code examples.

**Usage:**
```
/search-code [topic]
```

**Examples:**
```
/search-code FastMCP tool
/search-code OAuth provider
/search-code function decorator
```

### /search-functions
Search for function definitions and method implementations.

**Usage:**
```
/search-functions [function_name]
```

**Examples:**
```
/search-functions tool registration
/search-functions server initialization
/search-functions API endpoint
```

### /search-api
Search for API usage examples and endpoint definitions.

**Usage:**
```
/search-api [api_name]
```

**Examples:**
```
/search-api authentication
/search-api proxy configuration
/search-api server setup
```

### /search-tools
Search for specific tool examples and usage patterns.

**Usage:**
```
/search-tools [tool_name]
```

**Examples:**
```
/search-tools FastMCP CLI
/search-tools GitHubProvider
/search-tools Auth0 integration
```

### /search-oauth
Search for OAuth provider configuration examples.

**Usage:**
```
/search-oauth [provider]
```

**Examples:**
```
/search-oauth GitHub
/search-oauth Google
/search-oauth Azure
```

## Advanced Search Commands

### /search-code-exact
Search for exact code snippets or specific implementations.

**Usage:**
```
/search-code-exact [exact_phrase]
```

**Examples:**
```
/search-code-exact "@mcp.tool"
/search-code-exact "FastMCP("
/search-code-exact "as_proxy"
```

### /search-best
Get the best matching results for a query.

**Usage:**
```
/search-best [query]
```

**Examples:**
```
/search-best server initialization example
/search-best tool registration pattern
/search-best OAuth configuration
```

## Result Formatting Commands

### /search-with-context
Get search results with additional context.

**Usage:**
```
/search-with-context [query]
```

**Examples:**
```
/search-with-context FastMCP tool with authentication
/search-with-context server setup with proxy
```

### /search-detailed
Get detailed search results with full metadata.

**Usage:**
```
/search-detailed [query]
```

**Examples:**
```
/search-detailed complete FastMCP server example
/search-detailed full OAuth provider setup
```

## Integration Examples

### Example 1: Finding FastMCP Tool Usage
```
User: /search-code FastMCP tool example
Assistant: Found 3 results:
          
          --- Result 1 ---
          ID: 223
          Score: 0.1813
          Source: https://gofastmcp.com/patterns/cli
          Content: # Install with specific Python version
          fastmcp install claude-desktop server.py --python 3.11
```

### Example 2: Finding Function Definitions
```
User: /search-functions server initialization
Assistant: Found 2 results:
          
          --- Result 1 ---
          ID: 407
          Score: 0.1460
          Source: https://gofastmcp.com/servers/server
          Content: # Create a basic server instance
          mcp = FastMCP(name="MyAssistantServer")
```

### Example 3: Finding OAuth Configuration
```
User: /search-oauth GitHub
Assistant: Found 2 results:
          
          --- Result 1 ---
          ID: 152
          Score: 0.1716
          Source: https://gofastmcp.com/integrations/github
          Content: # The GitHubProvider handles GitHub's token format and validation
          auth_provider = GitHubProvider(
            client_id="Ov23liAbcDefGhiJkLmN",
            client_secret="github_pat_...",
            base_url="http://localhost:8000"
          )
```

## Tips for Using Slash Commands

1. **Be Specific**: Use specific terms in your queries for better results
2. **Combine Terms**: Combine multiple terms to narrow down results
3. **Use Quotes**: Use quotes around exact phrases you want to find
4. **Check Sources**: Look at the source URLs to understand the context of examples
5. **Iterate**: If you don't find what you're looking for, try rephrasing your query

## Customizing Search Results

You can customize the number of results returned by adding a limit parameter:

```
/search-code FastMCP tool --limit 10
```

This will return up to 10 results instead of the default 5.