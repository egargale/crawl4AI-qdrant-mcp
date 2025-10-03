# DashScope Setup Guide

This guide will help you set up DashScope API keys for use with the crawl4AI-agent-v2 project.

## What is DashScope?

DashScope is Alibaba Cloud's platform that provides access to Qwen (Tongyi) series models, including:
- Language models (Qwen-Turbo, Qwen-Plus, Qwen-Max, Qwen-Long)
- Text embedding models (text-embedding-v1, text-embedding-v2, text-embedding-v4)
- Other AI services

## Getting Started with DashScope

1. **Sign up for Alibaba Cloud**
   - Visit [Alibaba Cloud](https://www.alibabacloud.com/)
   - Create an account or sign in to your existing account

2. **Enable DashScope Service**
   - Go to the [DashScope console](https://dashscope.console.aliyun.com/)
   - Enable the DashScope service if it's not already enabled

3. **Get Your API Key**
   - In the DashScope console, navigate to "API Key Management"
   - Click "Create New API Key"
   - Copy the generated API key and save it securely

## Configuration for crawl4AI-agent-v2

Update your `.env` file with your DashScope API key:

```env
DASHSCOPE_API_KEY=your_actual_dashscope_api_key_here
```

Note: You can also set it as an environment variable in your system:
- Linux/Mac: `export DASHSCOPE_API_KEY=your_actual_dashscope_api_key_here`
- Windows: `set DASHSCOPE_API_KEY=your_actual_dashscope_api_key_here`

## Model Selection

The project uses these models by default:
- For embeddings: `text-embedding-v4` (latest version)
- For language model: `qwen-turbo` (can be changed via `MODEL_CHOICE` environment variable)

Available models:
- `qwen-turbo`: Balanced choice for most applications
- `qwen-plus`: More powerful with better reasoning
- `qwen-max`: Most capable model for complex tasks
- `qwen-long`: Optimized for handling long context

To change the model, set the `MODEL_CHOICE` environment variable:
```env
MODEL_CHOICE=qwen-plus
```

## Testing Your Setup

You can test your DashScope setup with a simple Python script:

```python
import os
from openai import OpenAI

# Initialize client with DashScope endpoint
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

# Test embedding
response = client.embeddings.create(
    model="text-embedding-v4",
    input=["Hello, world!"]
)
print("Embedding dimension:", len(response.data[0].embedding))

# Test language model
completion = client.chat.completions.create(
    model="qwen-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]
)
print("Response:", completion.choices[0].message.content)
```

## Pricing Information

DashScope offers competitive pricing with a generous free tier:
- Free quota for new users
- Pay-as-you-go pricing for additional usage
- Volume discounts for high usage

Check the latest [pricing information](https://dashscope.console.aliyun.com/billing) in the DashScope console.

## Troubleshooting

1. **Authentication Error**: Verify your API key is correct and properly set
2. **Model Not Found**: Check that you're using a supported model name
3. **Rate Limiting**: If you hit rate limits, consider implementing exponential backoff
4. **Quota Exhausted**: Check your usage in the DashScope console and upgrade if needed

For more information, visit the [DashScope documentation](https://help.aliyun.com/zh/dashscope/).