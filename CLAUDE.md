# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv (recommended)
uv sync

# Alternative: Install with pip
pip install -r requirements.txt

# Install Playwright dependencies (required for crawl4ai)
playwright install-deps
playwright install chromium

# Set up environment variables
cp .env.template .env
# Edit .env with your API keys
```

### Core Project Scripts
```bash
# 1. Crawl websites and extract content
python website_downloader.py https://example.com -o output_dir --max-depth 2

# 2. Process and store content in Qdrant
python rag_setup.py output_dir --collection my_docs

# 3. Query content with RAG
python rag_query.py "What is this about?" --collection my_docs

# 4. Use the RAG agent with Pydantic AI
python rag_agent_qdrant.py --question "What are the main features?" --collection my_docs

# 5. Simple document retrieval
python retrieve_docs_qdrant.py "search query" --collection my_docs

# Test project setup
python test_setup.py

# Run example usage
python example_usage.py
```

### Testing
- Use `test_setup.py` to verify environment configuration and dependencies
- No formal test suite is configured - check `test_setup.py` for validation patterns

## Architecture Overview

This is a Python-based RAG (Retrieval-Augmented Generation) system with these core components:

### Main Processing Pipeline
1. **Website Crawling** (`website_downloader.py`): Uses crawl4ai with LLM-based content extraction
2. **Content Processing** (`rag_setup.py`): Processes crawled content and stores in Qdrant with embeddings
3. **Query & Retrieval** (`rag_query.py`, `retrieve_docs_qdrant.py`): Semantic search and question answering
4. **AI Agent** (`rag_agent_qdrant.py`): Pydantic AI agent with retrieval tools

### Key Dependencies
- **crawl4ai**: Advanced web crawling with browser automation
- **Qdrant**: Vector database for semantic search
- **DashScope/Qwen**: LLM and embeddings provider
- **LangChain**: Orchestration framework
- **Pydantic AI**: Agent framework
- **FastEmbed**: Alternative embedding provider

### Data Flow
1. Websites are crawled and content extracted using LLMs
2. Content is processed and chunked, then stored in Qdrant with embeddings
3. Queries use semantic search to retrieve relevant documents
4. LLM generates answers based on retrieved context

## Configuration

### Required Environment Variables
Set these in `.env` file (copy from `.env.template`):
- `DASHSCOPE_API_KEY`: For Qwen LLM and embeddings
- `QDRANT_URL`: Qdrant server URL (http://localhost:6333 for local)
- `QDRANT_API_KEY`: Qdrant API key (optional for local instances)

### Default Settings
- Default Qdrant collection: `website_docs`
- Default output directory: `downloaded_website`
- Default crawl depth: 3
- Default concurrent requests: 5

## File Structure Notes

- **Core Scripts**: Standalone executables for each pipeline stage
- **custom_dashscope_embeddings.py**: Custom embeddings implementation for DashScope
- **main.py**: Entry point demonstrating project setup
- **No package structure**: All code is in root directory with direct script execution
- **Python 3.13+ required**: Uses modern Python features and dependencies

## Common Usage Patterns

### Website Processing Workflow
```bash
# Crawl a website
python website_downloader.py https://docs.example.com -o docs --max-depth 2

# Store in Qdrant
python rag_setup.py docs --collection docs_v1

# Query the content
python rag_query.py "How do I configure X?" --collection docs_v1
```

### Agent-based Interaction
```bash
# Use with FastEmbed instead of DashScope
python rag_agent_qdrant.py --question "What are the key features?" --collection docs --embedding-method fastembed
```