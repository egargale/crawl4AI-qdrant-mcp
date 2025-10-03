# Getting Started with crawl4AI-agent-v2

This guide will help you set up and run the crawl4AI-agent-v2 project from scratch.

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.13 or higher
- Git
- Docker (for local Qdrant setup)
- An Alibaba Cloud account (for DashScope API access)

## Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/crawl4AI-agent-v2.git
cd crawl4AI-agent-v2
```

## Step 2: Set Up Python Environment

It's recommended to use a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Alternatively, if you're using `uv` (faster dependency management):

```bash
# Install uv if you haven't already
pip install uv

# Install project dependencies
uv sync
```

## Step 3: Install Playwright Dependencies

crawl4ai requires Playwright for browser automation:

```bash
playwright install-deps
playwright install chromium
```

## Step 4: Set Up Qdrant

You have two options for Qdrant:

### Option A: Local Qdrant with Docker (Development)

1. Run Qdrant in a Docker container:
```bash
docker run -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

2. Verify Qdrant is running at http://localhost:6333

### Option B: Qdrant Cloud (Production)

1. Sign up at https://cloud.qdrant.io/
2. Create a new cluster
3. Get your cluster URL and API key from the dashboard

For detailed instructions, see [QDRANT_SETUP.md](QDRANT_SETUP.md).

## Step 5: Set Up DashScope

1. Sign up for Alibaba Cloud if you haven't already
2. Enable DashScope service
3. Create an API key in the DashScope console

For detailed instructions, see [DASHSCOPE_SETUP.md](DASHSCOPE_SETUP.md).

## Step 6: Configure Environment Variables

1. Copy the template environment file:
```bash
cp .env.template .env
```

2. Edit the `.env` file with your actual API keys:

For local Qdrant:
```env
DASHSCOPE_API_KEY=your_actual_dashscope_api_key_here
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
```

For Qdrant Cloud:
```env
DASHSCOPE_API_KEY=your_actual_dashscope_api_key_here
QDRANT_URL=https://YOUR-CLUSTER-URL.qdrant.tech:6333
QDRANT_API_KEY=your_qdrant_api_key_here
```

## Step 7: Test Your Setup

Run the main script to verify everything is working:

```bash
python main.py
```

You should see output showing that the environment variables are loaded correctly.

## Step 8: Quick Start Example

Let's walk through a complete example:

1. **Crawl a website**:
```bash
python website_downloader.py https://example.com -o example_content --max-depth 2
```

2. **Process and store content**:
```bash
python rag_setup.py example_content --collection example_docs
```

3. **Ask a question**:
```bash
python rag_query.py "What is this website about?" --collection example_docs
```

## Next Steps

- Explore the different components and their options
- Check out the detailed usage examples in [README.md](README.md)
- Customize the LLM extraction prompts in `website_downloader.py`
- Experiment with different embedding models and parameters

## Troubleshooting

If you encounter issues:

1. **Import errors**: Make sure all dependencies are installed correctly
2. **Connection errors**: Verify Qdrant is running and accessible
3. **Authentication errors**: Double-check your API keys in the `.env` file
4. **Playwright errors**: Reinstall Playwright dependencies

For more detailed troubleshooting, refer to the specific setup guides:
- [QDRANT_SETUP.md](QDRANT_SETUP.md)
- [DASHSCOPE_SETUP.md](DASHSCOPE_SETUP.md)

## Need Help?

- Check the [README.md](README.md) for detailed documentation
- File an issue on GitHub if you encounter bugs
- Refer to the official documentation for the underlying technologies:
  - [crawl4ai](https://github.com/unclecode/crawl4ai)
  - [Qdrant](https://qdrant.tech/documentation/)
  - [DashScope](https://help.aliyun.com/zh/dashscope/)