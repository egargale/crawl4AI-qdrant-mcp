# crawl4AI Agent v2

A Retrieval-Augmented Generation (RAG) system using Qdrant as the vector database and Crawl4AI for web crawling.

## Overview

This project implements a complete RAG pipeline that allows you to:
1. Crawl web content using Crawl4AI
2. Process and chunk documents intelligently
3. Store document embeddings in Qdrant
4. Retrieve relevant information using semantic search
5. Answer questions using a Pydantic AI agent

## Prerequisites

- Python 3.13+
- Qdrant server (cloud or self-hosted)
- DashScope API key for embeddings and LLM
- UV package manager

## Setup

1. Install dependencies using UV:
   ```bash
   uv sync
   ```

2. Configure environment variables in `.env`:
   ```env
   DASHSCOPE_API_KEY=your_dashscope_api_key
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_api_key
   ```

## Usage

### Initialize the project
```bash
uv init
uv add -r requirements.txt
uv sync
```

### Run the main demo
```bash
uv run main.py
```

### Insert documents into Qdrant
```bash
uv run insert_docs_qdrant.py <URL>
```

### Retrieve documents
```bash
uv run retrieve_docs_qdrant.py "search query"
```

### Run the RAG agent
```bash
uv run rag_agent_qdrant.py --question "Your question here"
```

## Project Structure

- `insert_docs_qdrant.py`: Crawl and insert documents into Qdrant
- `retrieve_docs_qdrant.py`: Document retrieval with CLI interface
- `rag_agent_qdrant.py`: RAG agent implementation using Pydantic AI
- `test_retrieval.py`: Test script for retrieval functionality
- `crawl4AI-examples/`: Example scripts for Crawl4AI usage

## Dependencies

All dependencies are managed through UV and defined in `pyproject.toml`. The project includes:
- crawl4ai for web crawling
- qdrant-client for vector storage
- openai for embeddings and LLM interactions
- pydantic-ai for agent implementation
- And many other supporting libraries