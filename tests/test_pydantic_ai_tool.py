#!/usr/bin/env python3
"""
Simple test of Pydantic AI with a tool to see if it works.
"""

import os
import asyncio
from dataclasses import dataclass
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

@dataclass
class TestDeps:
    """Test dependencies."""
    pass

# Create the agent with custom LLM configuration
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
    deps_type=TestDeps,
    system_prompt="You are a helpful assistant. When asked about the weather, use the get_weather tool to get information.",
    retries=3
)

@agent.tool
async def get_weather(context: RunContext[TestDeps], city: str) -> str:
    """Get the weather for a city."""
    print(f"DEBUG: get_weather tool called with city: {city}")
    if city.lower() == "london":
        return "The weather in London is sunny with a temperature of 22°C."
    elif city.lower() == "paris":
        return "The weather in Paris is cloudy with a temperature of 18°C."
    else:
        return f"I don't have weather information for {city}."

async def test_agent():
    """Test the agent."""
    question = "What is the weather in London?"
    
    # Create dependencies
    deps = TestDeps()
    
    print(f"DEBUG: Asking agent: {question}")
    # Run the agent
    result = await agent.run(question, deps=deps)
    
    print("\nResponse:")
    print(result.output)

if __name__ == "__main__":
    asyncio.run(test_agent())