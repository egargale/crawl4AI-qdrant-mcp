# crawl4AI-agent-v2 Project Context

## Project Overview

This is a Python-based project that implements a Retrieval-Augmented Generation (RAG) system using:
- **crawl4ai** for website crawling and content extraction
- **Qdrant** as the vector database for storing and retrieving document embeddings
- **DashScope/Qwen** for embeddings and language model capabilities
- **LangChain** for orchestration of the RAG pipeline

The project is designed to crawl websites, extract and process content, store it in a vector database with semantic embeddings, and then provide question-answering capabilities over that content.

## Core Components

### 1. Website Downloader (`website_downloader.py`)
Downloads entire websites recursively and saves pages in markdown format. Features:
- LLM-based extraction for intelligent content filtering and summarization
- Outputs content in both markdown and JSONL formats for RAG use
- Supports concurrent crawling with configurable depth
- Uses crawl4ai with browser automation

### 2. RAG Setup (`rag_setup.py`)
Processes downloaded content and sets up the RAG pipeline:
- Loads markdown or JSONL files from a directory
- Splits documents into chunks for better retrieval
- Creates embeddings using DashScope's optimized text embedding models
- Stores documents in Qdrant vector database

### 3. RAG Query (`rag_query.py`)
Provides querying capabilities over the stored documents:
- Semantic search using DashScope embeddings
- Question answering with Qwen LLM
- Supports both search-only mode and full RAG mode
- Uses LangChain for orchestration

### 4. RAG Agent (`rag_agent_qdrant.py`)
Implements a Pydantic AI agent with RAG capabilities:
- Custom agent with tools for document retrieval
- Integration with DashScope/Qwen models
- Configurable embedding methods (OpenAI or FastEmbed)

### 5. Document Retrieval (`retrieve_docs_qdrant.py`)
Simple interface for querying documents from Qdrant:
- Semantic search with configurable parameters
- Support for multiple embedding methods
- Formatted output of retrieval results

## Key Technologies

- **crawl4ai**: Advanced web crawling with LLM-based content extraction
- **Qdrant**: Vector database for semantic search
- **DashScope/Qwen**: Alibaba's language models and embeddings
- **LangChain**: Framework for building applications with LLMs
- **Pydantic AI**: Type-safe AI agent framework
- **FastEmbed**: Optional local embedding generation

## Environment Variables

The project requires the following environment variables (see `.env.template`):
- `DASHSCOPE_API_KEY`: API key for DashScope services
- `QDRANT_URL`: URL of the Qdrant instance
- `QDRANT_API_KEY`: API key for Qdrant (if required)

## Usage Patterns

### 1. Download Website Content
```bash
python website_downloader.py https://example.com -o downloaded_content --max-depth 2
```

### 2. Process and Store Content in Qdrant
```bash
python rag_setup.py downloaded_content --collection my_docs
```

### 3. Query the Content
```bash
# Question answering mode
python rag_query.py "What is this website about?" --collection my_docs

# Semantic search only
python rag_query.py "key features" --collection my_docs --search-only
```

### 4. Use the RAG Agent
```bash
python rag_agent_qdrant.py --question "What are the main features?" --collection my_docs
```

### 5. Simple Document Retrieval
```bash
python retrieve_docs_qdrant.py "search query" --collection my_docs
```

## Development Conventions

- Python 3.13+ is required
- Dependencies are managed with `uv` (see `pyproject.toml`)
- All scripts support command-line arguments for configuration
- Environment variables are loaded from `.env` file
- Code follows standard Python conventions with type hints
- Error handling is implemented throughout

## Common Workflows

1. **Setup**: Create `.env` file with API keys, ensure Qdrant is running
2. **Crawl**: Use `website_downloader.py` to download website content
3. **Process**: Use `rag_setup.py` to process and store content in Qdrant
4. **Query**: Use `rag_query.py` or `rag_agent_qdrant.py` to ask questions about the content

## Project Structure
```
├── website_downloader.py     # Website crawling and content extraction
├── rag_setup.py             # Process and store content in Qdrant
├── rag_query.py             # Query content with RAG
├── rag_agent_qdrant.py      # Pydantic AI agent with RAG
├── retrieve_docs_qdrant.py  # Simple document retrieval
├── custom_dashscope_embeddings.py  # Custom DashScope embeddings
├── main.py                  # Main entry point/demo
├── .env.template           # Template for environment variables
├── pyproject.toml          # Project dependencies and metadata
└── downloaded_website/     # Default output directory for crawled content (when using website_downloader)
```