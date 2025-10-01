#!/usr/bin/env python3
"""
Debug version of the RAG agent to see what's being retrieved.
"""

import os
import sys
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any
import asyncio

import dotenv
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI, OpenAI
from qdrant_client import QdrantClient

# Load environment variables from .env file
dotenv.load_dotenv()

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set.")
    print("Please create a .env file with your OpenAI API key or set it in your environment.")
    sys.exit(1)


@dataclass
class RAGDeps:
    """Dependencies for the RAG agent."""
    qdrant_client: QdrantClient
    collection_name: str
    embedding_model: str
    embedding_dim: int


# Create the RAG agent with custom LLM configuration
llm_base_url = os.getenv("LLM_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
model_choice = os.getenv("MODEL_CHOICE", "qwen-plus")

# Configure the OpenAI model with DashScope provider
qwen_model = OpenAIModel(
    model_name=model_choice,
    provider=OpenAIProvider(
        base_url=llm_base_url,
        api_key=os.getenv("DASHSCOPE_API_KEY")
    )
)

agent = Agent(
    model=qwen_model,
    deps_type=RAGDeps,
    system_prompt="You are a helpful assistant that answers questions based on the provided documentation. "
                  "Use the retrieve tool to get relevant information from the documentation before answering. "
                  "If the documentation doesn't contain the answer, clearly state that the information isn't available "
                  "in the current documentation and provide your best general knowledge response."
)


def get_qdrant_client() -> QdrantClient:
    """Initialize and return a Qdrant client."""
    QDRANT_URL = os.getenv('QDRANT_URL')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
    
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in the .env file")
    
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def get_embedding_client() -> OpenAI:
    """Initialize and return an OpenAI-compatible embedding client."""
    return OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    )


def generate_embeddings(client: OpenAI, texts: List[str], model_name: str, dimensions: int) -> List[List[float]]:
    """Generate embeddings for a list of texts."""
    all_embeddings = []
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(
            model=model_name,
            input=batch,
            dimensions=dimensions
        )
        all_embeddings.extend([data.embedding for data in resp.data])
    return all_embeddings


def query_collection(
    client: QdrantClient,
    embedding_client: OpenAI,
    collection_name: str,
    model_name: str,
    embedding_dim: int,
    query_text: str,
    n_results: int = 5,
) -> Dict[str, Any]:
    """Query a Qdrant collection for similar documents.
    
    Args:
        client: Qdrant client
        embedding_client: OpenAI-compatible embedding client
        collection_name: Name of the collection
        model_name: Name of the embedding model to use
        embedding_dim: Dimension of the embedding model
        query_text: Text to search for
        n_results: Number of results to return
        
    Returns:
        Query results containing documents, metadatas, distances, and ids
    """
    # Generate embedding for the query
    query_embedding = generate_embeddings(embedding_client, [query_text], model_name, embedding_dim)[0]
    
    # Query the collection
    search_result = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=n_results,
        with_payload=True
    )
    
    # Debug: Print what we're getting from Qdrant
    print(f"DEBUG: Found {len(search_result.points)} points from Qdrant")
    for i, point in enumerate(search_result.points):
        print(f"DEBUG: Point {i}: ID={point.id}, Score={point.score}")
        print(f"DEBUG: Point {i} Payload keys: {list(point.payload.keys())}")
        print(f"DEBUG: Point {i} Document content preview: {str(point.payload.get('document', ''))[:100]}...")
    
    # Format results similar to ChromaDB format for compatibility
    return {
        "ids": [[result.id for result in search_result.points]],
        "documents": [[result.payload.get("document", "") for result in search_result.points]],
        "metadatas": [[result.payload for result in search_result.points]],
        "distances": [[1 - result.score for result in search_result.points]]  # Convert score to distance
    }


def format_results_as_context(query_results: Dict[str, Any]) -> str:
    """Format query results as a context string for the agent.
    
    Args:
        query_results: Results from a Qdrant query
        
    Returns:
        Formatted context string
    """
    context = "CONTEXT INFORMATION:\n\n"
    
    print(f"DEBUG: Formatting {len(query_results['documents'][0])} documents as context")
    
    for i, (doc, metadata, distance) in enumerate(zip(
        query_results["documents"][0],
        query_results["metadatas"][0],
        query_results["distances"][0]
    )):
        print(f"DEBUG: Document {i}: Content preview: {str(doc)[:100]}...")
        # Add document information
        context += f"Document {i+1} (Relevance: {1 - distance:.2f}):\n"
        
        # Add metadata if available
        if metadata:
            for key, value in metadata.items():
                # Skip the document content as it's already included separately
                if key != "document":
                    context += f"{key}: {value}\n"
        
        # Add document content
        context += f"Content: {doc}\n\n"
    
    return context


@agent.tool
async def retrieve(context: RunContext[RAGDeps], search_query: str, n_results: int = 5) -> str:
    """Retrieve relevant documents from Qdrant based on a search query.
    
    Args:
        context: The run context containing dependencies.
        search_query: The search query to find relevant documents.
        n_results: Number of results to return (default: 5).
        
    Returns:
        Formatted context information from the retrieved documents.
    """
    print(f"DEBUG: Retrieving documents for query: '{search_query}'")
    # Get Qdrant client and embedding client
    embedding_client = get_embedding_client()
    
    # Query the collection
    query_results = query_collection(
        context.deps.qdrant_client,
        embedding_client,
        context.deps.collection_name,
        context.deps.embedding_model,
        context.deps.embedding_dim,
        search_query,
        n_results=n_results
    )
    
    # Format the results as context
    context_str = format_results_as_context(query_results)
    print(f"DEBUG: Returning context:\n{context_str}")
    return context_str


async def run_rag_agent(
    question: str,
    collection_name: str = "docs",
    embedding_model: str = "text-embedding-v3",
    embedding_dim: int = 1024,
    n_results: int = 5
) -> str:
    """Run the RAG agent to answer a question about Sierrachart.
    
    Args:
        question: The question to answer.
        collection_name: Name of the Qdrant collection to use.
        embedding_model: Name of the embedding model to use.
        embedding_dim: Dimension of the embedding model.
        n_results: Number of results to return from the retrieval.
        
    Returns:
        The agent's response.
    """
    # Create dependencies
    deps = RAGDeps(
        qdrant_client=get_qdrant_client(),
        collection_name=collection_name,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim
    )
    
    # Run the agent
    result = await agent.run(question, deps=deps)
    
    return result.output


def main():
    """Main function to parse arguments and run the RAG agent."""
    parser = argparse.ArgumentParser(description="Run a Pydantic AI agent with RAG using Qdrant")
    parser.add_argument("--question", help="The question to answer about Sierrachart")
    parser.add_argument("--collection", default="docs", help="Name of the Qdrant collection")
    parser.add_argument("--embedding-model", default="text-embedding-v3", help="Name of the embedding model to use")
    parser.add_argument("--embedding-dim", type=int, default=1024, help="Dimension of the embedding model")
    parser.add_argument("--n-results", type=int, default=5, help="Number of results to return from the retrieval")
    
    args = parser.parse_args()
    
    # Run the agent
    response = asyncio.run(run_rag_agent(
        args.question,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        embedding_dim=args.embedding_dim,
        n_results=args.n_results
    ))
    
    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    main()