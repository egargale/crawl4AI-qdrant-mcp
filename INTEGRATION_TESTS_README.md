# MCP Server Qdrant Integration Tests

This directory contains integration tests that verify the compatibility between the `insert_docs_qdrant.py` script and the `mcp-server-qdrant` find functionality.

## Test Files

1. **`test_mcp_integration_simple.py`** - Simple integration test using FastEmbed for both insertion and retrieval
2. **`test_mcp_integration_comprehensive.py`** - Comprehensive test with multiple documents and various search queries
3. **`test_e2e_integration.py`** - End-to-end test demonstrating the full workflow

## Key Findings

### Compatibility Requirements

For documents inserted by `insert_docs_qdrant.py` to be correctly found by `mcp-server-qdrant`, both systems must use the same embedding method:

1. **OpenAI Embeddings**: Both systems use OpenAI-compatible embeddings
2. **FastEmbed**: Both systems use FastEmbed library with the same model

### Vector Naming Convention

The `mcp-server-qdrant` FastEmbedProvider uses a specific naming convention for vector names in Qdrant collections:
- Format: `fast-{model_name_part}`
- Example: For model `sentence-transformers/all-MiniLM-L6-v2`, the vector name is `fast-all-minilm-l6-v2`

### Test Results

All integration tests are passing, demonstrating that:

1. Documents inserted using FastEmbed can be successfully found by mcp-server-qdrant
2. Semantic search works correctly across both systems
3. The end-to-end workflow functions as expected

## Running the Tests

```bash
# Run simple integration test
python test_mcp_integration_simple.py

# Run comprehensive integration test
python test_mcp_integration_comprehensive.py

# Run end-to-end integration test
python test_e2e_integration.py
```

## Important Notes

- For production use, ensure both systems use the same embedding method
- The tests automatically clean up test collections after running
- Environment variables must be configured (QDRANT_URL, QDRANT_API_KEY, etc.)