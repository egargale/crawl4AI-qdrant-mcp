"""
Website Downloader with LLM Extraction - JSONL Version
----------------------------------------------------
Downloads an entire website recursively and saves pages in markdown format.
Uses LLM-based extraction for intelligent content filtering and summarization.
Stores URLs in JSON data and outputs in JSONL format for RAG use.
"""

import os
import re
import sys
import json
import asyncio
import argparse
import urllib.parse
from pathlib import Path
from typing import Set, List, Optional
from pydantic import BaseModel, Field
from crawl4ai import (
    AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode,
    MemoryAdaptiveDispatcher, LLMConfig
)
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator


class ExtractedContent(BaseModel):
    """Model for LLM-extracted content structure."""
    title: str = Field(description="The main title or headline of the page")
    summary: str = Field(description="A concise summary of the page content")
    main_content: str = Field(description="The main content of the page, cleaned and structured")
    key_points: List[str] = Field(description="Key points or important information from the page")
    metadata: dict = Field(default_factory=dict, description="Additional metadata about the page")


class WebsiteDownloader:
    def __init__(self, base_url: str, output_dir: str = "downloaded_website",
                 max_depth: int = 3, max_concurrent: int = 5,
                 llm_provider: str = "openai/qwen-turbo",
                 llm_api_key: Optional[str] = None,
                 llm_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                 use_llm_extraction: bool = True,
                 output_format: str = "jsonl",
                 consolidate_output: bool = False):
        """
        Initialize the website downloader.
        
        Args:
            base_url: The starting URL to crawl from
            output_dir: Directory to save the downloaded pages
            max_depth: Maximum depth to crawl recursively
            max_concurrent: Maximum number of concurrent requests
            llm_provider: LLM provider string (e.g., "openai/qwen-turbo", "ollama/llama2")
            llm_api_key: API key for the LLM provider (if required)
            llm_base_url: Base URL for the LLM API endpoint
            use_llm_extraction: If True, use LLM-based extraction; if False, use standard markdown
            output_format: Output format for extracted data ("json" or "jsonl")
            consolidate_output: If True, consolidate all pages into single files
        """
        self.base_url = base_url.rstrip('/')
        self.output_dir = Path(output_dir)
        self.max_depth = max_depth
        self.max_concurrent = max_concurrent
        self.llm_provider = llm_provider
        self.llm_api_key = llm_api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        self.llm_base_url = llm_base_url
        self.use_llm_extraction = use_llm_extraction
        self.output_format = output_format
        self.consolidate_output = consolidate_output
        self.visited_urls: Set[str] = set()
        self.all_extracted_data = []  # For consolidated JSONL output
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)
        
        # Parse the base URL to get the domain for filtering
        self.parsed_base = urllib.parse.urlparse(self.base_url)
        self.base_domain = self.parsed_base.netloc
        
        # Cache for parsed URLs to improve performance
        self._parsed_urls_cache = {}
        
    def normalize_url(self, url: str) -> str:
        """Normalize URL by removing fragments and ensuring consistency."""
        return urllib.parse.urldefrag(url)[0]
    
    def is_internal_url(self, url: str) -> bool:
        """Check if URL belongs to the same domain."""
        if url in self._parsed_urls_cache:
            parsed = self._parsed_urls_cache[url]
        else:
            parsed = urllib.parse.urlparse(url)
            self._parsed_urls_cache[url] = parsed
        return parsed.netloc == self.base_domain
    
    def url_to_filename(self, url: str) -> str:
        """Convert URL to a valid filename."""
        # Remove protocol and domain
        parsed = urllib.parse.urlparse(url)
        path = parsed.path
        
        # Handle root path
        if not path or path == "/":
            return "index.md"
        
        # Clean the path
        path = path.strip("/")
        
        # Replace invalid characters
        path = re.sub(r'[<>:"/\\|?*]', '_', path)
        
        # Ensure it ends with .md
        if not path.endswith('.md'):
            path += '.md'
            
        return path
    
    def save_content(self, url: str, content: str, extracted_data: Optional[dict] = None):
        """Save the content to files in various formats."""
        try:
            filename = self.url_to_filename(url)
            filepath = self.output_dir / filename
            
            # Create subdirectories if needed
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# {url}\n\n")
                
                if self.use_llm_extraction and extracted_data:
                    # Handle different formats of extracted data
                    if isinstance(extracted_data, dict):
                        # Expected format: dictionary with title, summary, key_points, etc.
                        f.write(f"## {extracted_data.get('title', 'Untitled')}\n\n")
                        f.write(f"**Summary:** {extracted_data.get('summary', 'No summary available')}\n\n")
                        
                        if extracted_data.get('key_points'):
                            f.write("## Key Points\n\n")
                            for point in extracted_data['key_points']:
                                f.write(f"- {point}\n")
                            f.write("\n")
                        
                        f.write("## Main Content\n\n")
                        f.write(extracted_data.get('main_content', content))
                        
                        # Save metadata as JSON at the end
                        if extracted_data.get('metadata'):
                            f.write("\n\n## Metadata\n\n")
                            f.write("```json\n")
                            f.write(json.dumps(extracted_data['metadata'], indent=2))
                            f.write("\n```\n")
                    elif isinstance(extracted_data, list):
                        # Alternative format: list of extracted blocks
                        f.write("## Extracted Content\n\n")
                        for i, block in enumerate(extracted_data):
                            if isinstance(block, dict):
                                if 'content' in block:
                                    f.write(f"### Block {i+1}\n\n")
                                    if 'tags' in block and block['tags']:
                                        f.write(f"**Tags:** {', '.join(block['tags'])}\n\n")
                                    f.write(f"{block['content']}\n\n")
                                else:
                                    # If it's a dict but not in expected format, just write it as JSON
                                    f.write(f"### Block {i+1}\n\n")
                                    f.write("```json\n")
                                    f.write(json.dumps(block, indent=2))
                                    f.write("\n```\n\n")
                            else:
                                # If it's not a dict, just write it as text
                                f.write(f"### Block {i+1}\n\n")
                                f.write(f"{block}\n\n")
                    else:
                        # Fallback: just write the content as text
                        f.write("## Extracted Content\n\n")
                        f.write(str(extracted_data))
                else:
                    # Save standard markdown
                    f.write(content)
                
                print(f"Saved: {filepath}")
                
                # Also save the extracted JSON data if available
                if self.use_llm_extraction and extracted_data:
                    jsonl_filepath = filepath.with_suffix('.jsonl')
                    
                    # Handle different formats of extracted data for JSONL
                    if isinstance(extracted_data, dict):
                        # Expected format: dictionary with title, summary, key_points, etc.
                        jsonl_entry = {
                            "url": url,
                            "title": extracted_data.get('title', 'Untitled'),
                            "summary": extracted_data.get('summary', ''),
                            "main_content": extracted_data.get('main_content', ''),
                            "key_points": extracted_data.get('key_points', []),
                            "metadata": extracted_data.get('metadata', {})
                        }
                        with open(jsonl_filepath, 'a', encoding='utf-8') as jsonl_f:
                            jsonl_f.write(json.dumps(jsonl_entry) + '\n')
                        print(f"Saved JSONL: {jsonl_filepath}")
                        
                        # Also add to consolidated data for potential single file output
                        self.all_extracted_data.append(jsonl_entry)
                        
                    elif isinstance(extracted_data, list):
                        # Alternative format: list of extracted blocks
                        # Process each item in the list
                        jsonl_entries = []
                        for i, item in enumerate(extracted_data):
                            if isinstance(item, dict):
                                # If it's a dict, try to extract meaningful data
                                jsonl_entry = {
                                    "url": url,
                                    "title": item.get('title', f'Extracted Block {i+1}'),
                                    "summary": item.get('summary', ''),
                                    "main_content": item.get('content', str(item)),
                                    "key_points": item.get('key_points', []),
                                    "metadata": item.get('metadata', {})
                                }
                            else:
                                # If it's not a dict, convert to string and use as content
                                jsonl_entry = {
                                    "url": url,
                                    "title": f'Extracted Block {i+1}',
                                    "summary": '',
                                    "main_content": str(item),
                                    "key_points": [],
                                    "metadata": {}
                                }
                            jsonl_entries.append(jsonl_entry)
                        
                        # Write all entries to JSONL file
                        with open(jsonl_filepath, 'a', encoding='utf-8') as jsonl_f:
                            for entry in jsonl_entries:
                                jsonl_f.write(json.dumps(entry) + '\n')
                        print(f"Saved JSONL: {jsonl_filepath}")
                        
                        # Add all entries to consolidated data
                        self.all_extracted_data.extend(jsonl_entries)
                        
                    else:
                        # Fallback: convert to string and create a basic entry
                        jsonl_entry = {
                            "url": url,
                            "title": 'Extracted Content',
                            "summary": '',
                            "main_content": str(extracted_data),
                            "key_points": [],
                            "metadata": {}
                        }
                        with open(jsonl_filepath, 'a', encoding='utf-8') as jsonl_f:
                            jsonl_f.write(json.dumps(jsonl_entry) + '\n')
                        print(f"Saved JSONL: {jsonl_filepath}")
                        
                        # Also add to consolidated data
                        self.all_extracted_data.append(jsonl_entry)
                    
        except Exception as e:
            print(f"Error saving {url}: {str(e)}")
    
    def save_consolidated_jsonl(self):
        """Save all extracted data to a single JSONL file for RAG use."""
        if self.all_extracted_data:
            consolidated_filepath = self.output_dir / "consolidated_data.jsonl"
            with open(consolidated_filepath, 'w', encoding='utf-8') as f:
                for entry in self.all_extracted_data:
                    f.write(json.dumps(entry) + '\n')
            print(f"Saved consolidated JSONL: {consolidated_filepath}")
            print(f"Total entries: {len(self.all_extracted_data)}")
    
    async def crawl_recursive(self):
        """Crawl the website recursively and save pages in markdown format."""
        # Configure browser for headless operation
        browser_config = BrowserConfig(
            headless=True,
            verbose=False,
            extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
        )
        
        # Configure LLM extraction strategy if enabled
        extraction_strategy = None
        if self.use_llm_extraction:
            if not self.llm_api_key:
                print("Warning: No LLM API key provided. Using standard markdown extraction.")
                self.use_llm_extraction = False
            else:
                try:
                    llm_config = LLMConfig(
                        provider=self.llm_provider,
                        api_token=self.llm_api_key,
                        base_url=self.llm_base_url
                    )
                    
                    extraction_strategy = LLMExtractionStrategy(
                        llm_config=llm_config,
                        schema=ExtractedContent.model_json_schema(),
                        extraction_type="schema",
                        instruction="""Extract and structure the content from this webpage.
                        Provide a clear title, concise summary, the main content cleaned and organized,
                        and key points. Retain any code example and any framework and function explanation.
                        Focus on the most important information and remove noise like navigation, ads, and irrelevant content.""",
                        chunk_token_threshold=2000,
                        overlap_rate=0.1,
                        apply_chunking=True,
                        input_format="markdown",
                        extra_args={"temperature": 0.1, "max_tokens": 2000},
                        verbose=True
                    )
                    print(f"Using LLM extraction with provider: {self.llm_provider}")
                except Exception as e:
                    print(f"Error configuring LLM extraction: {e}")
                    print("Falling back to standard markdown extraction.")
                    self.use_llm_extraction = False
        
        # Configure crawl
        md_generator = DefaultMarkdownGenerator(
            content_source="cleaned_html",
            options={"skip_internal_links": True}
        )
        
        crawl_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            stream=False,
            markdown_generator=md_generator,
            extraction_strategy=extraction_strategy,
            excluded_tags=["nav", "footer", "header", "aside", "script", "style", "form"],
            exclude_external_links=True,
            exclude_social_media_links=True,
            exclude_domains=["adtrackers.com", "spammynews.org"],
            word_count_threshold=10
        )
        
        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=70.0,
            check_interval=1.0,
            max_session_permit=self.max_concurrent
        )
        
        # Start with the base URL
        current_urls = {self.normalize_url(self.base_url)}
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            for depth in range(self.max_depth):
                print(f"\n=== Crawling Depth {depth+1} ===")
                
                # Filter URLs we haven't visited yet
                urls_to_crawl = [
                    url for url in current_urls
                    if self.normalize_url(url) not in self.visited_urls
                ]
                
                if not urls_to_crawl:
                    print("No new URLs to crawl at this depth.")
                    break
                
                print(f"Crawling {len(urls_to_crawl)} URLs at depth {depth+1}")
                
                # Batch crawl all URLs at this depth in parallel
                results = await crawler.arun_many(
                    urls=urls_to_crawl,
                    config=crawl_config,
                    dispatcher=dispatcher
                )
                
                next_level_urls = set()
                
                for result in results:
                    norm_url = self.normalize_url(result.url)
                    self.visited_urls.add(norm_url)
                    
                    if result.success:
                        print(f"[OK] {result.url}")
                        
                        # Save the content
                        if self.use_llm_extraction and hasattr(result, 'extracted_content') and result.extracted_content:
                            try:
                                # Parse the extracted JSON content
                                extracted_data = json.loads(result.extracted_content)
                                self.save_content(result.url, result.markdown, extracted_data)
                            except (json.JSONDecodeError, Exception) as e:
                                print(f"Error parsing extracted content: {e}")
                                # Fallback to standard markdown
                                self.save_content(result.url, result.markdown)
                        else:
                            # Save standard markdown
                            self.save_content(result.url, result.markdown)
                        
                        # Collect internal links for next depth
                        if hasattr(result, 'links') and result.links:
                            for link in result.links.get("internal", []):
                                href = link.get("href", "")
                                if href and self.is_internal_url(href):
                                    next_url = self.normalize_url(href)
                                    if next_url not in self.visited_urls:
                                        next_level_urls.add(next_url)
                    else:
                        print(f"[ERROR] {result.url}: {result.error_message}")
                
                # Move to the next set of URLs for the next recursion depth
                current_urls = next_level_urls
                
                if not current_urls:
                    print("No more URLs to crawl.")
                    break
        
        print(f"\nCrawling completed. Total pages downloaded: {len(self.visited_urls)}")
        print(f"Files saved to: {self.output_dir.absolute()}")
        
        # Save consolidated JSONL if we have extracted data
        if self.use_llm_extraction and self.all_extracted_data:
            self.save_consolidated_jsonl()
        
        # Show LLM usage statistics if available (disabled to remove usage history output)
        # if self.use_llm_extraction and extraction_strategy:
        #     try:
        #         extraction_strategy.show_usage()
        #     except Exception as e:
        #         print(f"Could not show LLM usage statistics: {e}")


def print_help():
    """Print help information."""
    help_text = """
Website Downloader with LLM Extraction - JSONL Version
=====================================================

Downloads an entire website recursively and saves pages in markdown format.
Uses LLM-based extraction for intelligent content filtering and summarization.
Stores URLs in JSON data and outputs in JSONL format for RAG use.

USAGE:
    python website_downloader_jsonl.py <URL> [OPTIONS]

REQUIRED ARGUMENTS:
    URL                    The starting URL to crawl from (e.g., https://example.com)

OPTIONAL ARGUMENTS:
    -o, --output-dir DIR   Output directory to save downloaded pages (default: downloaded_website)
    -d, --max-depth NUM    Maximum depth to crawl recursively (default: 3)
    -c, --max-concurrent NUM  Maximum number of concurrent requests (default: 5)
    --llm-provider STR     LLM provider string (default: openai/qwen-turbo)
    --llm-api-key STR      API key for the LLM provider (default: OPENAI_API_KEY or DASHSCOPE_API_KEY env var)
    --llm-base-url STR     Base URL for the LLM API endpoint (default: DashScope compatible endpoint)
    --no-llm               Disable LLM extraction and use standard markdown
    -h, --help             Show this help message

EXAMPLES:
    # Download a website with LLM extraction (default)
    python website_downloader_jsonl.py https://docs.crawl4ai.com

    # Download with custom output directory and depth
    python website_downloader_jsonl.py https://example.com -o my_site -d 2

    # Download with increased concurrency for faster crawling
    python website_downloader_jsonl.py https://example.com -c 10

    # Download with different LLM provider
    python website_downloader_jsonl.py https://example.com --llm-provider ollama/llama2

    # Download with custom LLM base URL
    python website_downloader_jsonl.py https://example.com --llm-base-url https://custom-llm-endpoint.com/v1

    # Download without LLM extraction (standard markdown only)
    python website_downloader_jsonl.py https://example.com --no-llm

OUTPUT:
    The script creates markdown files in the output directory:
    
    With LLM extraction (default):
    - *.md files contain structured content with title, summary, key points, and main content
    - *.jsonl files contain the extracted data in JSONL format for RAG use
    - consolidated_data.jsonl contains all extracted data in a single file
    
    Without LLM extraction (--no-llm):
    - *.md files contain standard markdown
    
    Files are organized in a directory structure that mirrors the website's URL structure.
    The JSONL format includes the URL in each entry for easy reference and RAG applications.

ENVIRONMENT VARIABLES:
    OPENAI_API_KEY         API key for OpenAI models (if not provided via --llm-api-key)
    DASHSCOPE_API_KEY      API key for DashScope models (if not provided via --llm-api-key)
"""
    print(help_text)


async def main():
    """Main function to run the website downloader."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Download an entire website recursively and save pages in markdown format.",
        add_help=False  # We'll handle help manually to show our custom help
    )
    
    parser.add_argument("url", nargs="?", help="The starting URL to crawl from")
    parser.add_argument("-o", "--output-dir", default="downloaded_website",
                       help="Output directory to save downloaded pages (default: downloaded_website)")
    parser.add_argument("-d", "--max-depth", type=int, default=3,
                       help="Maximum depth to crawl recursively (default: 3)")
    parser.add_argument("-c", "--max-concurrent", type=int, default=5,
                       help="Maximum number of concurrent requests (default: 5)")
    parser.add_argument("--llm-provider", default="openai/qwen-turbo",
                       help="LLM provider string (default: openai/qwen-turbo)")
    parser.add_argument("--llm-api-key", default=None,
                       help="API key for the LLM provider (default: OPENAI_API_KEY or DASHSCOPE_API_KEY env var)")
    parser.add_argument("--llm-base-url", default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                       help="Base URL for the LLM API endpoint (default: DashScope compatible endpoint)")
    parser.add_argument("--no-llm", action="store_true",
                       help="Disable LLM extraction and use standard markdown")
    parser.add_argument("-h", "--help", action="store_true",
                       help="Show this help message")
    
    args = parser.parse_args()
    
    # Show help if requested or if no URL is provided
    if args.help or not args.url:
        print_help()
        return
    
    # Validate URL
    if not args.url.startswith(("http://", "https://")):
        print("Error: URL must start with http:// or https://")
        print_help()
        return
    
    # Validate parameters
    if args.max_depth < 1:
        print("Error: Max depth must be at least 1")
        return
    
    if args.max_concurrent < 1:
        print("Error: Max concurrent must be at least 1")
        return
    
    # Configuration
    config = {
        "base_url": args.url,
        "output_dir": args.output_dir,
        "max_depth": args.max_depth,
        "max_concurrent": args.max_concurrent,
        "llm_provider": args.llm_provider,
        "llm_api_key": args.llm_api_key,
        "llm_base_url": args.llm_base_url,
        "use_llm_extraction": not args.no_llm
    }
    
    print("Website Downloader with LLM Extraction")
    print("=" * 40)
    print(f"Base URL: {config['base_url']}")
    print(f"Output Directory: {config['output_dir']}")
    print(f"Max Depth: {config['max_depth']}")
    print(f"Max Concurrent: {config['max_concurrent']}")
    print(f"LLM Provider: {config['llm_provider']}")
    print(f"LLM Base URL: {config['llm_base_url']}")
    print(f"LLM Extraction: {'Enabled' if config['use_llm_extraction'] else 'Disabled'}")
    if config['use_llm_extraction'] and not config['llm_api_key']:
        print("Note: Using OPENAI_API_KEY or DASHSCOPE_API_KEY environment variable for LLM authentication")
    print()
    
    downloader = WebsiteDownloader(**config)
    await downloader.crawl_recursive()


if __name__ == "__main__":
    asyncio.run(main())