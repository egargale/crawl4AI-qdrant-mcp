#!/usr/bin/env python3
"""
Test the Pydantic AI agent directly with context.
"""

import os
import sys
import asyncio
from dataclasses import dataclass

import dotenv
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Load environment variables from .env file
dotenv.load_dotenv()

@dataclass
class RAGDeps:
    """Dependencies for the RAG agent."""
    dummy: str = "dummy"

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

test_context = """CONTEXT INFORMATION:

Document 1 (Relevance: 0.87):
headers: ### sc.AddAndManageSingleTextDrawingForStudy()
char_count: 1068
word_count: 164
chunk_index: 0
source: https://www.sierrachart.com/index.php?page=doc/ACSIL_Members_Functions.html
Content: ### sc.AddAndManageSingleTextDrawingForStudy()

The sc.AddAndManageSingleTextDrawingForStudy() function adds or manages a single text drawing for a study. This function is used to display text on a chart that is associated with a specific study instance.

This function is particularly useful for displaying study-specific information such as parameter values, calculated values, or status messages directly on the chart where the study is applied.

Parameters:
- DrawingText: A string that specifies the text to be displayed.
- LineNumber: An integer that specifies the line number for the text drawing.
- HorizontalPosition: An integer that specifies the horizontal position of the text.
- VerticalPosition: A float that specifies the vertical position of the text.
- Color: An integer that specifies the color of the text.
- FontSize: An integer that specifies the font size of the text.
- FontBold: A boolean that specifies whether the text should be bold.

Returns:
- An integer handle to the text drawing, which can be used to modify or delete the drawing later."""

@agent.tool
async def retrieve(context: RunContext[RAGDeps], search_query: str, n_results: int = 5) -> str:
    """Mock retrieve tool that returns predefined context."""
    print(f"DEBUG: Retrieve tool called with query: {search_query}")
    return test_context

async def test_agent():
    """Test the agent with a specific question."""
    question = "explain function AddAndManageSingleTextDrawingForStudy"
    
    # Create dependencies
    deps = RAGDeps()
    
    # Run the agent
    print(f"Asking agent: {question}")
    result = await agent.run(question, deps=deps)
    
    print("\nResponse:")
    print(result.output)

if __name__ == "__main__":
    asyncio.run(test_agent())