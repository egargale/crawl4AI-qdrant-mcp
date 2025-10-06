# FastMCP RAG System Makefile

.PHONY: help install test lint format clean run dev docker-build docker-run

# Default target
help:
	@echo "FastMCP RAG System - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  install      Install dependencies"
	@echo "  dev          Start development server with debug"
	@echo "  test         Run tests"
	@echo "  lint         Run linting"
	@echo "  format       Format code"
	@echo "  clean        Clean temporary files"
	@echo ""
	@echo "Production:"
	@echo "  run          Start production server"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run with Docker Compose"
	@echo ""
	@echo "Utilities:"
	@echo "  check        Check configuration"
	@echo "  health       Check server health"
	@echo "  docs         Generate documentation"

# Development commands
install:
	@echo "Installing FastMCP RAG dependencies..."
	pip install -r requirements-fastmcp.txt
	@echo "Dependencies installed successfully!"

dev:
	@echo "Starting FastMCP RAG server in development mode..."
	python fastmcp_rag_server.py --debug --host 127.0.0.1

run:
	@echo "Starting FastMCP RAG server..."
	python fastmcp_rag_server.py

# Testing commands
test:
	@echo "Running FastMCP RAG tests..."
	pytest fastmcp_rag/tests/ -v --cov=fastmcp_rag --cov-report=term-missing

test-integration:
	@echo "Running integration tests..."
	RUN_INTEGRATION_TESTS=true pytest fastmcp_rag/tests/ -v -m integration

test-all: test test-integration

# Code quality commands
lint:
	@echo "Running linting..."
	ruff check fastmcp_rag/
	mypy fastmcp_rag/

format:
	@echo "Formatting code..."
	ruff format fastmcp_rag/
	black fastmcp_rag/

format-check:
	@echo "Checking code formatting..."
	ruff format --check fastmcp_rag/
	black --check fastmcp_rag/

# Cleanup commands
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .pytest_cache/
	@echo "Cleanup completed!"

# Docker commands
docker-build:
	@echo "Building FastMCP RAG Docker image..."
	docker build -t fastmcp-rag:latest -f Dockerfile.fastmcp .

docker-run:
	@echo "Starting FastMCP RAG with Docker Compose..."
	docker-compose -f docker-compose.fastmcp.yml up -d

docker-stop:
	@echo "Stopping FastMCP RAG Docker containers..."
	docker-compose -f docker-compose.fastmcp.yml down

docker-logs:
	@echo "Showing FastMCP RAG Docker logs..."
	docker-compose -f docker-compose.fastmcp.yml logs -f

# Configuration and utilities
check:
	@echo "Checking FastMCP RAG configuration..."
	python -c "
from fastmcp_rag.config import load_config_from_env
try:
    config = load_config_from_env()
    missing = config.validate_required_config()
    if missing:
        print('❌ Configuration errors:')
        for error in missing:
            print(f'  - {error}')
        exit(1)
    else:
        print('✅ Configuration is valid')
        print(f'📊 Server: {config.server_name}')
        print(f'🔗 Qdrant: {config.qdrant_url}')
        print(f'🧠 Embeddings: {config.embedding_method} ({config.embedding_model})')
        print(f'🤖 LLM: {config.llm_provider} ({config.llm_model})')
except Exception as e:
    print(f'❌ Configuration error: {e}')
    exit(1)
"

health:
	@echo "Checking FastMCP RAG server health..."
	curl -s http://localhost:8000/health | python -m json.tool || echo "❌ Server not running or unhealthy"

docs:
	@echo "Generating FastMCP RAG documentation..."
	@echo "Documentation available in README_FSTMCP.md"

# Quick start commands
quick-start: install check
	@echo "🚀 Quick starting FastMCP RAG..."
	@echo "1. Make sure Qdrant is running on http://localhost:6333"
	@echo "2. Configure your .env file (copy from .env.fastmcp.template)"
	@echo "3. Starting server..."
	python fastmcp_rag_server.py --debug

# Development workflow
dev-setup: install
	@echo "Setting up development environment..."
	cp .env.fastmcp.template .env
	@echo "✅ Development environment ready!"
	@echo "📝 Please edit .env with your configuration"
	@echo "🔍 Then run: make dev"

# Production workflow
prod-setup: check
	@echo "Production environment ready!"
	@echo "🚀 Run: make run or make docker-run"

# CI/CD commands
ci: lint format-check test
	@echo "✅ CI checks passed!"

# Full test suite
full-check: ci check test-all
	@echo "✅ All checks passed!"