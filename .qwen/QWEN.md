# Qwen Code Context for crawl4AI-agent-v2

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system using Qdrant as the vector database and Crawl4AI for web crawling. The system allows users to crawl web content, process it into semantic chunks, store it in Qdrant with embeddings, and then retrieve relevant information using semantic search to answer questions.

### Core Components

1. **Web Crawling**: Uses Crawl4AI to extract content from URLs, sitemaps, or text files
2. **Document Processing**: Intelligently chunks documents while preserving semantic structure
3. **Vector Storage**: Stores document chunks in Qdrant with embeddings for semantic search
4. **Retrieval System**: Implements semantic search functionality to find relevant documents
5. **RAG Agent**: Uses Pydantic AI with Qwen models to answer questions based on retrieved context

### Technologies Used

- **Python 3.7+**
- **Qdrant**: Vector database for storing and retrieving document embeddings
- **Crawl4AI**: Web crawling library for content extraction
- **OpenAI/DashScope API**: For generating text embeddings and language model responses
- **Pydantic AI**: Framework for building AI agents with structured outputs
- **Environment Configuration**: Using python-dotenv for configuration management

## Project Structure

```
├── .env                      # Environment configuration file
├── insert_docs_qdrant.py     # Script to crawl URLs and insert content into Qdrant
├── qdrant_test.py            # Test script for Qdrant functionality
├── rag_agent_qdrant.py       # RAG agent implementation using Pydantic AI
├── requirements.txt          # Python dependencies
├── RETRIEVAL_README.md       # Documentation for retrieval functionality
├── retrieve_docs_qdrant.py   # Document retrieval script with CLI interface
├── test_retrieval.py         # Test script for retrieval functionality
├── crawl4AI-examples/        # Example scripts for Crawl4AI usage
│   ├── 1-crawl_single_page.py
│   ├── 2-crawl_docs_sequential.py
│   ├── 3-crawl_sitemap_in_parallel.py
│   ├── 4-crawl_llms_txt.py
│   ├── 5-crawl_site_recursively.py
│   ├── 6-crawl_docs_FAST.py
│   └── test.py
```

## Setup and Configuration

### Prerequisites

1. Python 3.7+
2. Qdrant server (cloud or self-hosted)
3. DashScope API key (for embeddings and LLM)
4. OpenAI API key (for GPT models, if needed)

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables in `.env`:
```env
# DashScope configuration for embeddings and LLM
DASHSCOPE_API_KEY=your_dashscope_api_key
DASHSCOPE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1

# Qwen model configuration
MODEL_CHOICE=qwen-turbo
LLM_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1

# Qdrant configuration
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
```

## Usage

### 1. Inserting Documents into Qdrant

The `insert_docs_qdrant.py` script crawls web content and inserts it into Qdrant:

```bash
# Crawl a single page or recursively crawl internal links
python insert_docs_qdrant.py https://example.com/page

# Crawl a sitemap
python insert_docs_qdrant.py https://example.com/sitemap.xml

# Crawl a text/markdown file
python insert_docs_qdrant.py https://example.com/document.txt

# Advanced usage with custom parameters
python insert_docs_qdrant.py https://example.com \
    --collection my_docs \
    --embedding-model text-embedding-v3 \
    --embedding-dim 1024 \
    --chunk-size 1000 \
    --max-depth 3 \
    --max-concurrent 10
```

### 2. Retrieving Documents

The `retrieve_docs_qdrant.py` script provides a simple interface for document retrieval:

```bash
# Basic usage
python retrieve_docs_qdrant.py "search query"

# Advanced usage
python retrieve_docs_qdrant.py "how to install python" \
    --collection docs \
    --embedding-model text-embedding-v3 \
    --embedding-dim 1024 \
    --n-results 10 \
    --score-threshold 0.7 \
    --full-content
```

### 3. Running the RAG Agent

The `rag_agent_qdrant.py` script runs a Pydantic AI agent that answers questions using retrieved context:

```bash
python rag_agent_qdrant.py --question "Your question here"
```

### 4. Testing

Run the test scripts to verify functionality:

```bash
# Test retrieval functionality
python test_retrieval.py

# Test Qdrant connection and basic operations
python qdrant_test.py
```

## Development Workflow

### Adding New Content

1. Use `insert_docs_qdrant.py` to crawl and process new content
2. The script automatically handles:
   - Content detection (regular page, sitemap, text file)
   - Intelligent document chunking
   - Embedding generation
   - Qdrant collection management
   - Batch insertion of documents

### Querying Content

1. Use `retrieve_docs_qdrant.py` for simple document retrieval
2. Use `rag_agent_qdrant.py` for question-answering with context
3. Both scripts support filtering by similarity score and limiting results

### Programmatic Usage

The retrieval functionality can be used programmatically:

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

## Key Features

### Intelligent Document Chunking

The system uses hierarchical chunking that:
- Preserves document structure by splitting on headers (#, ##, ###)
- Maintains semantic coherence within chunks
- Respects maximum length constraints (default: 1000 characters)

### Flexible Content Sources

Supports crawling:
- Regular web pages
- Sitemaps (XML format)
- Text/markdown files
- Recursive crawling of internal links

### Semantic Search

- Uses cosine similarity for document matching
- Supports score threshold filtering
- Returns relevance scores with results

### RAG Implementation

- Integrates with Qwen models via DashScope
- Uses Pydantic AI for structured agent development
- Provides context-aware question answering

## Troubleshooting

### Common Issues

1. **Missing Environment Variables**: Ensure all required variables are set in `.env`
2. **Qdrant Connection Errors**: Verify QDRANT_URL and QDRANT_API_KEY are correct
3. **Embedding Generation Failures**: Check DASHSCOPE_API_KEY validity
4. **Empty Search Results**: Try lowering the score threshold or increasing n_results

### Error Handling

All scripts include comprehensive error handling for:
- Missing environment variables
- Connection issues with Qdrant
- Invalid collection names
- Embedding generation failures
- Empty search results

## Performance Optimization

1. **Batch Processing**: For multiple queries, consider batching them to reduce API calls
2. **Score Thresholds**: Use score thresholds to filter out low-quality results
3. **Result Limits**: Set appropriate n_results limits to avoid overwhelming responses
4. **Embedding Caching**: Consider caching embeddings for repeated queries
5. **Collection Management**: The system automatically handles dimension mismatches by recreating collections when needed