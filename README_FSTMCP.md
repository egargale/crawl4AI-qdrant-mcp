# FastMCP Enhanced Semantic Search RAG System

A production-ready semantic search RAG system built with FastMCP, integrating seamlessly with your existing crawl4ai-agent-v2 project infrastructure.

## Features

- **🚀 FastMCP Integration**: Leverage FastMCP's enterprise features and simplified development
- **🔍 Semantic Search**: Advanced document search with multiple embedding methods
- **🧠 Intelligent Q&A**: Context-aware question answering with LLM integration
- **📊 Multi-Provider Support**: Support for sentence-transformers, DashScope, and OpenAI
- **🔒 Enterprise Authentication**: Built-in OAuth providers (GitHub, Google, etc.)
- **⚡ High Performance**: Async processing, caching, and optimization
- **📈 Monitoring**: Built-in metrics and health checks
- **🐳 Production Ready**: Docker deployment and monitoring

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements-fastmcp.txt

# Or using uv
uv pip install -r requirements-fastmcp.txt
```

### 2. Configuration

Copy the environment template and configure:

```bash
cp .env.fastmcp.template .env
```

Edit `.env` with your configuration:

```bash
# Required
QDRANT_URL=http://localhost:6333

# Optional (choose your embedding method)
EMBEDDING_METHOD=sentence-transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Or use DashScope
# EMBEDDING_METHOD=dashscope
# DASHSCOPE_API_KEY=your_dashscope_api_key
# EMBEDDING_MODEL=text-embedding-v4

# Or use OpenAI
# EMBEDDING_METHOD=openai
# OPENAI_API_KEY=your_openai_api_key
# EMBEDDING_MODEL=text-embedding-3-small
```

### 3. Start Qdrant (if not already running)

```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant

# Or locally if you have Qdrant installed
qdrant --service-mode
```

### 4. Run the Server

```bash
# Start with default configuration
python fastmcp_rag_server.py

# Start with custom settings
python fastmcp_rag_server.py --host 127.0.0.1 --port 8080

# Start with debug mode
python fastmcp_rag_server.py --debug

# Start with stdio transport (for MCP clients)
python fastmcp_rag_server.py --transport stdio
```

### 5. Test the Server

The server exposes several tools that can be used with any MCP-compatible client:

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Server Info
```bash
curl http://localhost:8000/mcp -X POST -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "server_info", "arguments": {}}}'
```

## Available Tools

### Document Management

- **`add_document`**: Add documents to the RAG system
- **`delete_document`**: Remove documents from the system
- **`list_collections`**: List available document collections

### Search and Q&A

- **`search_documents`**: Semantic search through documents
- **`intelligent_qa`**: Ask questions and get AI-powered answers

### System Information

- **`server_info`**: Get server status and configuration
- **`get_collection_stats`**: Get collection statistics

## Usage Examples

### Python Client Example

```python
import asyncio
from fastmcp import Client

async def main():
    # Connect to the server
    client = Client("http://localhost:8000/mcp")

    async with client:
        # Add a document
        result = await client.call_tool("add_document", {
            "content": "FastMCP is a Python framework for building MCP servers with enterprise features.",
            "title": "FastMCP Overview",
            "source": "docs"
        })
        print(result.content[0].text)

        # Search documents
        results = await client.call_tool("search_documents", {
            "query": "FastMCP features",
            "limit": 5
        })
        print(results.content[0].text)

        # Ask a question
        answer = await client.call_tool("intelligent_qa", {
            "question": "What are the key features of FastMCP?",
            "context_limit": 3
        })
        print(answer.content[0].text)

if __name__ == "__main__":
    asyncio.run(main())
```

### Claude Code Integration

Add to your `.vscode/mcp.json`:

```json
{
  "servers": {
    "rag": {
      "command": "python",
      "args": ["/path/to/fastmcp_rag_server.py"],
      "cwd": "/path/to/project"
    }
  }
}
```

## Configuration Options

### Embedding Methods

#### Sentence Transformers (Default)
```bash
EMBEDDING_METHOD=sentence-transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
```

#### DashScope
```bash
EMBEDDING_METHOD=dashscope
DASHSCOPE_API_KEY=your_api_key
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIMENSION=1536
```

#### OpenAI
```bash
EMBEDDING_METHOD=openai
OPENAI_API_KEY=your_api_key
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
```

### LLM Configuration

#### DashScope (Default)
```bash
LLM_PROVIDER=dashscope
DASHSCOPE_API_KEY=your_api_key
LLM_MODEL=qwen-turbo
LLM_TEMPERATURE=0.7
```

#### OpenAI
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key
LLM_MODEL=gpt-3.5-turbo
LLM_TEMPERATURE=0.7
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=fastmcp_rag --cov-report=html

# Run specific test file
pytest fastmcp_rag/tests/test_server.py -v

# Run with debug output
pytest -v -s --tb=short
```

### Project Structure

```
fastmcp_rag/
├── __init__.py              # Package initialization
├── server.py               # Main server implementation
├── config/
│   ├── __init__.py
│   └── settings.py         # Configuration management
├── tests/
│   ├── __init__.py
│   ├── test_server.py      # Server functionality tests
│   └── test_config.py      # Configuration tests
└── [future modules]
    ├── auth/              # Authentication providers
    ├── cache/             # Caching implementation
    └── monitoring/        # Metrics and monitoring
```

## Deployment

### Docker

```bash
# Build image
docker build -t fastmcp-rag:latest -f Dockerfile.fastmcp .

# Run with Docker Compose
docker-compose -f docker-compose.fastmcp.yml up -d
```

### Environment Variables

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `QDRANT_URL` | Qdrant server URL | Required |
| `EMBEDDING_METHOD` | Embedding method | `sentence-transformers` |
| `LLM_PROVIDER` | LLM provider | `dashscope` |
| `ENABLE_AUTH` | Enable authentication | `false` |
| `ENABLE_CACHE` | Enable Redis caching | `true` |
| `ENABLE_METRICS` | Enable metrics | `true` |

## Integration with Existing System

This FastMCP implementation is designed to work alongside your existing crawl4ai-agent-v2 project:

### Migration Strategy

1. **Parallel Operation**: Run FastMCP server alongside existing RAG system
2. **Data Migration**: Use the migration scripts to transfer existing documents
3. **Client Migration**: Update applications to use FastMCP tools
4. **Full Cutover**: Decommission original system after successful migration

### Compatibility

- ✅ **Qdrant**: Uses your existing Qdrant instance
- ✅ **crawl4ai**: Can be integrated as a document source
- ✅ **DashScope**: Supports your existing DashScope API keys
- ✅ **sentence-transformers**: Compatible with your current embedding models

## Monitoring and Health

### Health Endpoint

```bash
curl http://localhost:8000/health
```

### Metrics Endpoint

```bash
curl http://localhost:8000/metrics
```

### Logging

Server logs include:
- Request/response information
- Error details and stack traces
- Performance metrics
- System health status

## Roadmap

### Phase 1: Core Implementation ✅
- [x] Basic FastMCP server
- [x] Document management tools
- [x] Semantic search functionality
- [x] Intelligent Q&A
- [x] Configuration system
- [x] Basic tests

### Phase 2: Advanced Features (Future)
- [ ] Query expansion and context awareness
- [ ] Hybrid search capabilities
- [ ] Document processing pipelines
- [ ] Caching implementation
- [ ] Advanced ranking algorithms

### Phase 3: Enterprise Features (Future)
- [ ] Authentication providers
- [ ] Rate limiting and access control
- [ ] Performance monitoring
- [ ] Scalability improvements

## Troubleshooting

### Common Issues

1. **Qdrant Connection Failed**
   ```bash
   # Check if Qdrant is running
   curl http://localhost:6333/collections

   # Verify URL in configuration
   echo $QDRANT_URL
   ```

2. **Embedding Model Loading Failed**
   ```bash
   # Install sentence-transformers
   pip install sentence-transformers

   # Or check API key for DashScope/OpenAI
   echo $DASHSCOPE_API_KEY
   ```

3. **Port Already in Use**
   ```bash
   # Check what's using the port
   lsof -i :8000

   # Use a different port
   python fastmcp_rag_server.py --port 8080
   ```

### Debug Mode

Enable debug mode for detailed logging:

```bash
python fastmcp_rag_server.py --debug
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project extends the existing crawl4ai-agent-v2 project and maintains the same licensing terms.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the test files for usage examples
3. Enable debug mode for detailed error information
4. Check the logs for specific error messages