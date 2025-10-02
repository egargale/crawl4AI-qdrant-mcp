# https://docs.crawl4ai.com

## Extracted Content

### Block 1

```json
{
  "title": "\ud83d\ude80\ud83e\udd16 Crawl4AI: Open-Source LLM-Friendly Web Crawler & Scraper",
  "summary": "Crawl4AI is a fast, open-source web crawler and scraper designed for large language models, AI agents, and data pipelines. It offers adaptive crawling, structured extraction, and high performance with support for asynchronous operations.",
  "main_content": "Crawl4AI is the #1 trending GitHub repository, actively maintained by a vibrant community. It delivers blazing-fast, AI-ready web crawling tailored for large language models, AI agents, and data pipelines. Fully open source, flexible, and built for real-time performance, Crawl4AI empowers developers with unmatched speed, precision, and deployment ease.\n\nNew: Adaptive Web Crawling - Crawl4AI now features intelligent adaptive crawling that knows when to stop! Using advanced information foraging algorithms, it determines when sufficient information has been gathered to answer your query.\n\nQuick Start:\nHere's a quick example to show you how easy it is to use Crawl4AI with its asynchronous capabilities:\n\nimport asyncio\nfrom crawl4ai import AsyncWebCrawler\n\nasync def main():\n    # Create an instance of AsyncWebCrawler\n    async with AsyncWebCrawler() as crawler:\n        # Run the crawler on a URL\n        result = await crawler.arun(url=\"https://crawl4ai.com\")\n        # Print the extracted content\n        print(result.markdown)\n\n# Run the async main function\nasyncio.run(main())\n\nWhat Does Crawl4AI Do?\nCrawl4AI is a feature-rich crawler and scraper that aims to:\n1. Generate Clean Markdown: Perfect for RAG pipelines or direct ingestion into LLMs.\n2. Structured Extraction: Parse repeated patterns with CSS, XPath, or LLM-based extraction.\n3. Advanced Browser Control: Hooks, proxies, stealth modes, session re-use\u2014fine-grained control.\n4. High Performance: Parallel crawling, chunk-based extraction, real-time use cases.\n5. Open Source: No forced API keys, no paywalls\u2014everyone can access their data.\n\nCore Philosophies:\n- Democratize Data: Free to use, transparent, and highly configurable.\n- LLM Friendly: Minimally processed, well-structured text, images, and metadata, so AI models can easily consume it.\n\nDocumentation Structure:\nTo help you get started, we\u2019ve organized our docs into clear sections:\n* Setup & Installation: Basic instructions to install Crawl4AI via pip or Docker.\n* Quick Start: A hands-on introduction showing how to do your first crawl, generate Markdown, and do a simple extraction.\n* Core: Deeper guides on single-page crawling, advanced browser/crawler parameters, content filtering, and caching.\n* Advanced: Explore link & media handling, lazy loading, hooking & authentication, proxies, session management, and more.\n* Extraction: Detailed references for no-LLM (CSS, XPath) vs. LLM-based strategies, chunking, and clustering approaches.\n* API Reference: Find the technical specifics of each class and method, including `AsyncWebCrawler`, `arun()`, and `CrawlResult`.\n\nHow You Can Support:\n* Star & Fork: If you find Crawl4AI helpful, star the repo on GitHub or fork it to add your own features.\n* File Issues: Encounter a bug or missing feature? Let us know by filing an issue, so we can improve.\n* Pull Requests: Whether it\u2019s a small fix, a big feature, or better docs\u2014contributions are always welcome.\n* Join Discord: Come chat about web scraping, crawling tips, or AI workflows with the community.\n* Spread the Word: Mention Crawl4AI in your blog posts, talks, or on social media.\n\nOur mission: to empower everyone\u2014students, researchers, entrepreneurs, data scientists\u2014to access, parse, and shape the world\u2019s data with speed, cost-efficiency, and creative freedom.",
  "key_points": [
    "Crawl4AI is a fast, open-source web crawler and scraper for AI and data pipelines.",
    "It supports adaptive crawling using advanced information foraging algorithms.",
    "Features include clean markdown generation, structured extraction, and advanced browser control.",
    "The tool is asynchronous and supports parallel crawling for high performance.",
    "Documentation includes setup, quick start, core concepts, advanced features, and API reference.",
    "Community support includes GitHub stars, issues, pull requests, and Discord participation.",
    "The project emphasizes democratizing data access and being LLM-friendly."
  ],
  "metadata": {
    "url": "https://docs.crawl4ai.com"
  }
}
```

