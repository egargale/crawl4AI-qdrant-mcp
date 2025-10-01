# Qdrant Document Retrieval Script

A simple Python script for retrieving documents from Qdrant collections using semantic search with embeddings.

## Features

- **Simple Interface**: Easy-to-use command-line interface for document retrieval
- **Semantic Search**: Uses embeddings to find semantically similar documents
- **Flexible Configuration**: Supports custom collections, embedding models, and search parameters
- **Score Filtering**: Optional similarity score threshold filtering
- **Multiple Output Formats**: Preview mode or full content display
- **Programmatic Usage**: Can be imported and used in other Python scripts

## Prerequisites

- Python 3.7+
- Qdrant server running and accessible
- OpenAI-compatible embedding API (e.g., DashScope, OpenAI)
- Environment variables configured (see setup section)

## Setup

1. **Install Dependencies**:
   ```bash
   pip install qdrant-client openai python-dotenv
   ```

2. **Configure Environment Variables**:
   Create a `.env` file in the same directory with:
   ```env
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_api_key
   DASHSCOPE_API_KEY=your_embedding_api_key
   DASHSCOPE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1  # Optional
   ```

## Usage

### Command Line Interface

Basic usage:
```bash
python retrieve_docs_qdrant.py "your search query"
```

Advanced usage with options:
```bash
python retrieve_docs_qdrant.py "how to install python" \
    --collection docs \
    --embedding-model text-embedding-v3 \
    --embedding-dim 1024 \
    --n-results 10 \
    --score-threshold 0.7 \
    --full-content
```

### Command Line Arguments

- `query` (required): The search query text
- `--collection`: Qdrant collection name (default: "docs")
- `--embedding-model`: Embedding model name (default: "text-embedding-v3")
- `--embedding-dim`: Embedding dimension (default: 1024)
- `--n-results`: Number of results to return (default: 5)
- `--score-threshold`: Minimum similarity score (0-1) to filter results
- `--full-content`: Show full document content instead of preview

### Programmatic Usage

You can also use the retrieval functions in your own Python code:

```python
from retrieve_docs_qdrant import (
    get_qdrant_client,
    get_embedding_client,
    retrieve_documents,
    format_results
)

# Initialize clients
qdrant_client = get_qdrant_client()
embedding_client = get_embedding_client()

# Retrieve documents
results = retrieve_documents(
    qdrant_client=qdrant_client,
    embedding_client=embedding_client,
    collection_name="docs",
    model_name="text-embedding-v3",
    embedding_dim=1024,
    query_text="your search query",
    n_results=5,
    score_threshold=0.7
)

# Format and display results
formatted_output = format_results(results, show_full_content=False)
print(formatted_output)
```

## Output Format

The script returns results in the following format:

```
Found X relevant documents:

Result 1:
  ID: 123
  Score: 0.8923
  Metadata:
    source: https://example.com/page1
    chunk_index: 0
  Content Preview:
    This is the beginning of the document content...

Result 2:
  ID: 456
  Score: 0.8234
  Metadata:
    source: https://example.com/page2
    chunk_index: 1
  Content Preview:
    Another document with relevant information...
```

## Testing

Run the test script to verify your setup:
```bash
python test_retrieval.py
```

## Integration with Existing Scripts

This retrieval script works seamlessly with the existing Qdrant integration scripts:

- **insert_docs_qdrant.py**: Use this to insert documents into Qdrant
- **rag_agent_qdrant.py**: Use this for RAG-based question answering
- **retrieve_docs_qdrant.py**: Use this for simple document retrieval (this script)

## Error Handling

The script includes comprehensive error handling for:
- Missing environment variables
- Connection issues with Qdrant
- Invalid collection names
- Embedding generation failures
- Empty search results

## Performance Tips

1. **Batch Processing**: For multiple queries, consider batching them to reduce API calls
2. **Score Threshold**: Use score thresholds to filter out low-quality results
3. **Result Limits**: Set appropriate `n_results` limits to avoid overwhelming responses
4. **Embedding Caching**: Consider caching embeddings for repeated queries

##