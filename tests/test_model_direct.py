#!/usr/bin/env python3
"""
Direct test of the Qwen model with context.
"""

import os
import asyncio
from openai import AsyncOpenAI
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

async def test_model():
    """Test the model directly with context."""
    
    # Configure the OpenAI model with DashScope provider
    client = AsyncOpenAI(
        base_url=os.getenv("LLM_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
        api_key=os.getenv("DASHSCOPE_API_KEY")
    )
    
    model_choice = os.getenv("MODEL_CHOICE", "qwen-plus")
    
    context = """CONTEXT INFORMATION:

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

    question = "explain function AddAndManageSingleTextDrawingForStudy"
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided documentation. Use the context information to answer the question. If the documentation doesn't contain the answer, clearly state that the information isn't available in the current documentation."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
    
    print(f"Asking model: {question}")
    print(f"Using context:\n{context}")
    print("\n" + "="*50 + "\n")
    
    response = await client.chat.completions.create(
        model=model_choice,
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )
    
    print("Response:")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    asyncio.run(test_model())