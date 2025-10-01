# FastMCP Integration Examples from Qdrant Collection

This document summarizes the FastMCP integration examples found in the "manuali" collection in Qdrant.

## 1. Claude Desktop Integration

### Installation Commands
```bash
# Install with specific Python version
fastmcp install claude-desktop server.py --python 3.11

# Install with explicit config file
fastmcp install claude-desktop my-config.fastmcp.json
```

### Proxy Setup
```python
# Create a proxy to a remote server
proxy = FastMCP.as_proxy(
  "https://example.com/mcp/sse", 
  name="Remote Server Proxy"
)

if __name__ == "__main__":
  proxy.run() # Runs via STDIO for Claude Desktop
```

### Authenticated Proxy
```python
from fastmcp import FastMCP, Client
from fastmcp.client.auth import BearerAuth

# For authenticated remote servers
auth_client = Client(
  "https://example.com/mcp/sse",
  auth=BearerAuth(token="your-token")
)

proxy = FastMCP.as_proxy(auth_client, name="Authenticated Proxy")
```

## 2. OpenAPI Integration

### Server Creation from OpenAPI Spec
```python
# Create the MCP server from OpenAPI spec
mcp = FastMCP.from_openapi(
  openapi_spec=openapi_spec,
  client=client,
  name="My API Server"
)

if __name__ == "__main__":
  mcp.run()
```

### Authentication with OpenAPI
```python
import httpx
from fastmcp import FastMCP

# If your API requires authentication, configure it on the HTTP client
client = httpx.Client(
  base_url="https://api.example.com",
  headers={"Authorization": "Bearer your-token"}
)
```

## 3. Proxy Server Examples

### Basic Proxy
```python
# Get a proxy server
proxy = FastMCP.as_proxy("backend_server.py")
```

### Remote Proxy
```python
# Create a proxy to a remote server
proxy = FastMCP.as_proxy(
  "https://example.com/mcp/sse", 
  name="Remote Server Proxy"
)
```

### Config-Based Proxy
```python
# Create a proxy to the configured server (auto-creates ProxyClient)
proxy = FastMCP.as_proxy(config, name="Config-Based Proxy")
```

### Shared Session Proxy
```python
# Create and connect a client
async with Client("backend_server.py") as connected_client:
  # This proxy will reuse the connected session for all requests
  proxy = FastMCP.as_proxy(connected_client)
```

## 4. Authentication Integrations

### Auth0 Provider
```bash
# Use the Auth0 provider
FASTMCP_SERVER_AUTH=fastmcp.server.auth.providers.auth0.Auth0Provider
```

### GitHub Provider
```python
from fastmcp import FastMCP
from fastmcp.server.auth.providers.github import GitHubProvider

# GitHub OAuth provider
auth_provider = GitHubProvider(
  client_id="your_client_id",
  client_secret="your_client_secret",
  base_url="http://localhost:8000"
)

mcp = FastMCP(name="GitHub Secured App", auth=auth_provider)
```

### General Authentication Setup
```python
# Authentication automatically configured from environment
mcp = FastMCP(name="My Server")

@mcp.tool
def protected_tool(data: str) -> str:
  """This tool is now protected by OAuth."""
  return f"Processed: {data}"

if __name__ == "__main__":
  mcp.run(transport="http", port=8000)
```

## 5. CLI Commands

### Development Server
```bash
# Run dev server with specific Python version
fastmcp dev server.py --python 3.11

# Run dev server with requirements file
fastmcp dev server.py --with-requirements requirements.txt
```

### Server Execution
```bash
# Override port from config file
fastmcp run fastmcp.json --port 8080
```

### Inspection
```bash
# Output FastMCP format to stdout
fastmcp inspect server.py --format fastmcp

# Output MCP protocol format to stdout
fastmcp inspect server.py --format mcp
```

## Key Integration Patterns

1. **Proxy Pattern**: Use `FastMCP.as_proxy()` to create proxies for remote servers or local backends
2. **Authentication Pattern**: Configure authentication providers through environment variables or direct instantiation
3. **OpenAPI Pattern**: Generate MCP servers directly from OpenAPI specifications
4. **CLI Pattern**: Use command-line tools for installation, development, and deployment
5. **Configuration Pattern**: Use config files to manage complex server setups

These examples demonstrate the flexibility of FastMCP in integrating with various systems and services.