"""
Website Downloader with LLM Extraction - JSONL Version (BFS Implementation)
------------------------------------------------------------------------
Downloads an entire website using BFSDeepCrawlStrategy and saves pages in markdown format.
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
from typing import Set, List, Optional, Union
from pydantic import BaseModel, Field
from crawl4ai import (
    AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode,
    MemoryAdaptiveDispatcher, LLMConfig
)
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy


class ExtractedContent(BaseModel):
    """Model for LLM-extracted content structure."""
    title: str = Field(description="The main title or headline of the page")
    summary: str = Field(description="A concise summary of the page content")
    main_content: str = Field(description="The main content of the page, cleaned and structured")
    key_points: List[str] = Field(description="Key points or important information from the page")
    metadata: dict = Field(default_factory=dict, description="Additional metadata about the page")


class WebsiteDownloaderBFS:
    def __init__(self, base_url: str, output_dir: str = "downloaded_website",
                 max_depth: int = 3, max_pages: Optional[int] = None, max_concurrent: int = 30,
                 llm_provider: str = "openai/qwen-turbo",
                 llm_api_key: Optional[str] = None,
                 llm_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                 use_llm_extraction: bool = True,
                 json_output_only: bool = False,
                 output_format: str = "jsonl",
                 consolidate_output: bool = False):
        """
        Initialize the website downloader with BFS strategy.
        
        Args:
            base_url: The starting URL to crawl from
            output_dir: Directory to save the downloaded pages
            max_depth: Maximum depth to crawl recursively
            max_pages: Maximum number of pages to crawl (None for unlimited)
            max_concurrent: Maximum number of concurrent requests
            llm_provider: LLM provider string (e.g., "openai/qwen-turbo", "ollama/llama2")
            llm_api_key: API key for the LLM provider (if required)
            llm_base_url: Base URL for the LLM API endpoint
            use_llm_extraction: If True, use LLM-based extraction; if False, use standard markdown
            json_output_only: If True, only generate JSON files, no markdown files
            output_format: Output format for extracted data ("json" or "jsonl")
            consolidate_output: If True, consolidate all pages into single files (only applicable to jsonl format)
        """
        self.base_url = base_url.rstrip('/')
        self.output_dir = Path(output_dir)
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.max_concurrent = max_concurrent
        self.llm_provider = llm_provider
        self.llm_api_key = llm_api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.llm_base_url = llm_base_url
        self.use_llm_extraction = use_llm_extraction
        self.json_output_only = json_output_only
        self.output_format = output_format
        # Only apply consolidate_output to jsonl format
        self.consolidate_output = consolidate_output and (output_format == "jsonl")
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

            # Handle JSON-only mode
            if self.json_output_only:
                # Only save JSONL data
                if self.use_llm_extraction and extracted_data and (isinstance(extracted_data, dict) or isinstance(extracted_data, list)):
                    self._save_jsonl_data(url, extracted_data, filepath)
                    print(f"Saved JSONL: {filepath.with_suffix('.jsonl')}")
                else:
                    print(f"Warning: JSON-only mode requires LLM extraction. Skipping {url}")
                return

            # Normal mode: Save markdown files
            with open(filepath, 'w', encoding='utf-8') as f:
                if self.use_llm_extraction and extracted_data and (isinstance(extracted_data, dict) or isinstance(extracted_data, list)):
                    # Handle list of extracted objects
                    if isinstance(extracted_data, list) and extracted_data:
                        # Use the first item from the list
                        extracted_data = extracted_data[0]

                    # Write proper markdown structure from extracted data
                    f.write(f"# {extracted_data.get('title', 'Untitled')}\n\n")
                    f.write(f"**URL:** {url}\n\n")
                    f.write(f"**Summary:** {extracted_data.get('summary', 'No summary available')}\n\n")

                    # Write key points if available
                    if extracted_data.get('key_points'):
                        f.write("## Key Points\n\n")
                        for point in extracted_data['key_points']:
                            f.write(f"- {point}\n")
                        f.write("\n")

                    # Write main content
                    f.write("## Content\n\n")
                    main_content = extracted_data.get('main_content', content)
                    # Convert plain text to markdown format
                    main_content = self._text_to_markdown(main_content)
                    f.write(f"{main_content}\n\n")

                    # Write metadata if available
                    if extracted_data.get('metadata'):
                        f.write("## Metadata\n\n")
                        for key, value in extracted_data['metadata'].items():
                            f.write(f"- **{key}:** {value}\n")
                        f.write("\n")

                    # Save JSONL data alongside markdown
                    self._save_jsonl_data(url, extracted_data, filepath)
                else:
                    # Save standard markdown
                    f.write(f"# {url}\n\n")
                    f.write(content)

                print(f"Saved: {filepath}")

        except Exception as e:
            print(f"Error saving {url}: {str(e)}")

    def _text_to_markdown(self, text: str) -> str:
        """Convert plain text content to markdown format."""
        if not isinstance(text, str):
            text = str(text)

        # Convert paragraphs separated by double newlines to markdown
        lines = text.split('\n')
        markdown_lines = []
        code_block = False

        for line in lines:
            stripped = line.strip()

            # Handle code blocks
            if stripped.startswith('```'):
                code_block = not code_block
                markdown_lines.append(line)
                continue

            if code_block:
                markdown_lines.append(line)
                continue

            # Skip empty lines
            if not stripped:
                markdown_lines.append('')
                continue

            # Handle headers (if they look like headers)
            if stripped.startswith(('# ', '## ', '### ')):
                markdown_lines.append(stripped)
            # Handle bullet points
            elif stripped.startswith(('-', '*', '+')):
                markdown_lines.append(stripped)
            # Handle numbered lists
            elif stripped[0].isdigit() and len(stripped) > 1 and stripped[1] in ('.', ')'):
                markdown_lines.append(stripped)
            else:
                # Regular paragraph
                markdown_lines.append(stripped)

        return '\n'.join(markdown_lines)

    def _parse_llm_response(self, response: str) -> dict:
        """Parse LLM response and extract structured data."""
        if not response or not isinstance(response, str):
            return {}

        # Clean the response
        response = response.strip()

        # Try to parse as JSON directly
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        if response.startswith('```json'):
            # Extract JSON from markdown code block
            end_pos = response.find('```', 7)
            if end_pos != -1:
                json_str = response[7:end_pos].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
        elif response.startswith('```'):
            # Try to find JSON in any code block
            lines = response.split('\n')
            json_lines = []
            in_code_block = False

            for line in lines:
                if line.strip().startswith('```'):
                    in_code_block = not in_code_block
                    if in_code_block and 'json' in line.lower():
                        continue  # Skip the opening line
                    elif not in_code_block:
                        break  # End of code block
                    continue

                if in_code_block:
                    json_lines.append(line)

            if json_lines:
                try:
                    json_str = '\n'.join(json_lines).strip()
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass

        # Try to find JSON-like structures in the text
        import re
        # Look for JSON objects between { and }
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, response, re.DOTALL)

        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # If all else fails, create a basic structure
        return {
            "title": "Extracted Content",
            "summary": "Content extraction completed but structured data parsing failed.",
            "main_content": response,
            "key_points": [],
            "metadata": {"parsing_error": True}
        }

    def _save_jsonl_data(self, url: str, extracted_data: Union[dict, list], filepath: Path):
        """Save extracted data as JSONL."""
        try:
            jsonl_filepath = filepath.with_suffix('.jsonl')

            # Handle list of extracted objects
            if isinstance(extracted_data, list) and extracted_data:
                # Use the first item from the list for single page processing
                extracted_data = extracted_data[0]

            # Create JSONL entry
            jsonl_entry = {
                "url": url,
                "title": extracted_data.get('title', 'Untitled'),
                "summary": extracted_data.get('summary', ''),
                "main_content": extracted_data.get('main_content', ''),
                "key_points": extracted_data.get('key_points', []),
                "metadata": extracted_data.get('metadata', {})
            }

            # Write to JSONL file
            with open(jsonl_filepath, 'w', encoding='utf-8') as jsonl_f:
                jsonl_f.write(json.dumps(jsonl_entry, ensure_ascii=False) + '\n')

            print(f"Saved JSONL: {jsonl_filepath}")

            # Add to consolidated data
            self.all_extracted_data.append(jsonl_entry)

        except Exception as e:
            print(f"Error saving JSONL data for {url}: {str(e)}")
    
    def save_consolidated_jsonl(self):
        """Save all extracted data to a single JSONL file for RAG use."""
        if self.all_extracted_data:
            consolidated_filepath = self.output_dir / "consolidated_data.jsonl"
            with open(consolidated_filepath, 'w', encoding='utf-8') as f:
                for entry in self.all_extracted_data:
                    f.write(json.dumps(entry) + '\n')
            print(f"Saved consolidated JSONL: {consolidated_filepath}")
            print(f"Total entries: {len(self.all_extracted_data)}")
            
            # If consolidate_output is True, delete individual JSONL files to avoid duplication
            if self.consolidate_output:
                for file in self.output_dir.glob("*.jsonl"):
                    # Skip the consolidated file itself
                    if file.name != "consolidated_data.jsonl":
                        try:
                            file.unlink()
                            print(f"Deleted individual JSONL file: {file}")
                        except Exception as e:
                            print(f"Error deleting {file}: {e}")
    
    async def crawl_with_bfs(self):
        """Crawl the website using BFS strategy and save pages in markdown format."""
        # Configure browser for headless operation with optimized settings
        browser_config = BrowserConfig(
            headless=True,
            verbose=False,
            text_mode=True,  # Disable images for better performance
            extra_args=[
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
                "--disable-ipc-flooding-protection",
                "--disable-extensions",
                "--disable-plugins",
                "--disable-images",
                "--disable-javascript-harmony-shipping",
                "--disable-background-networking",
                "--disable-default-apps",
                "--disable-sync",
                "--disable-translate",
                "--hide-scrollbars",
                "--mute-audio"
            ],
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
                        instruction="""Extract and structure the content from this webpage according to the JSON schema provided.

                        CRITICAL: You must return a valid JSON object that exactly matches the schema with these fields:
                        - title: The main title of the page
                        - summary: A concise summary (2-3 sentences)
                        - main_content: The cleaned main content in plain text format
                        - key_points: Array of important bullet points
                        - metadata: Object with additional page information

                        IMPORTANT:
                        - Return ONLY valid JSON, no markdown formatting
                        - Do not wrap your response in ```json``` or any other markdown
                        - Ensure all text content is plain text, not markdown
                        - Include code examples but format them as plain text
                        - Focus on the most important information and remove noise like navigation, ads, and irrelevant content""",
                        chunk_token_threshold=2000,
                        overlap_rate=0.1,
                        apply_chunking=True,
                        input_format="html",  # Change from markdown to html for better extraction
                        extra_args={"temperature": 0.1, "max_tokens": 2000},
                        verbose=False  # Reduce LLM verbosity
                    )
                    print(f"Using LLM extraction with provider: {self.llm_provider}")
                except Exception as e:
                    print(f"Error configuring LLM extraction: {e}")
                    print("Falling back to standard markdown extraction.")
                    self.use_llm_extraction = False
        
        # Configure crawl with BFS strategy
        md_generator = DefaultMarkdownGenerator(
            content_source="cleaned_html",
            options={
                "skip_internal_links": True,
                "ignore_images": True,
                "ignore_links": True
            }
        )
        
        # Create BFS crawl strategy
        bfs_strategy = BFSDeepCrawlStrategy(
            max_depth=self.max_depth,
            include_external=False,
            max_pages=self.max_pages if self.max_pages is not None else 999999  # Use large number for unlimited
        )
        
        crawl_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,  # Skip caching for speed
            stream=True,  # Stream results for better memory management
            markdown_generator=md_generator,
            extraction_strategy=extraction_strategy,
            excluded_tags=["nav", "footer", "header", "aside", "script", "style", "form", "iframe", "embed", "video", "audio"],
            exclude_external_links=True,
            exclude_social_media_links=True,
            exclude_domains=["adtrackers.com", "spammynews.org"],
            word_count_threshold=10,
            deep_crawl_strategy=bfs_strategy,
            wait_until="domcontentloaded",  # Faster than "load"
            page_timeout=30000,  # 30 seconds timeout
            remove_overlay_elements=True,  # Remove popups and overlays
            delay_before_return_html=0.1,  # Small delay between requests to be respectful
            verbose=False  # Reduce verbosity for cleaner output
        )
        
        # Increased concurrency settings for better performance
        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=85.0,  # Increased threshold
            check_interval=0.5,  # More frequent checks
            max_session_permit=self.max_concurrent
        )
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            print(f"\n=== Crawling with BFS Strategy ===")
            print(f"Base URL: {self.base_url}")
            print(f"Max Depth: {self.max_depth}")
            print(f"Max Pages: {self.max_pages if self.max_pages else 'Unlimited'}")
            print(f"Max Concurrent: {self.max_concurrent}")
            
            # Counter for successfully processed pages
            processed_count = 0
            
            # Use the BFS strategy to crawl
            try:
                async for result in await crawler.arun(
                    url=self.base_url,
                    config=crawl_config
                ):
                    if result.success:
                        print(f"[OK] {result.url}")
                        
                        # Save the content
                        if self.use_llm_extraction and hasattr(result, 'extracted_content') and result.extracted_content:
                            try:
                                # Parse the extracted JSON content
                                extracted_data = self._parse_llm_response(result.extracted_content)
                                self.save_content(result.url, result.markdown, extracted_data)
                                print(f"Successfully extracted LLM data from {result.url}")
                            except Exception as e:
                                print(f"Error parsing extracted content: {e}")
                                print(f"Raw extracted content preview: {result.extracted_content[:200]}...")
                                # Fallback to standard markdown
                                self.save_content(result.url, result.markdown)
                        else:
                            # Save standard markdown
                            try:
                                self.save_content(result.url, result.markdown)
                            except Exception as e:
                                print(f"Error saving content for {result.url}: {e}")
                                
                        # Track visited URLs to avoid duplication
                        norm_url = self.normalize_url(result.url)
                        self.visited_urls.add(norm_url)
                        processed_count += 1
                        
                        # Check if we've reached the max pages limit
                        if self.max_pages and processed_count >= self.max_pages:
                            print(f"Reached maximum pages limit ({self.max_pages}). Stopping crawl.")
                            break
                    else:
                        # Handle errors gracefully without stopping the crawl
                        print(f"[ERROR] {result.url}: {result.error_message}")
                        
            except Exception as e:
                print(f"[FATAL ERROR] Unexpected error during crawling: {str(e)}")
                
        print(f"\nCrawling completed. Total pages downloaded: {len(self.visited_urls)}")
        print(f"Files saved to: {self.output_dir.absolute()}")
        
        # Save consolidated JSONL if we have extracted data and consolidation is enabled
        if self.use_llm_extraction and self.all_extracted_data and self.consolidate_output:
            self.save_consolidated_jsonl()


def print_help():
    """Print help information."""
    help_text = """
Website Downloader with LLM Extraction - BFS Version
===================================================

Downloads an entire website using BFSDeepCrawlStrategy and saves pages in markdown format.
Uses LLM-based extraction for intelligent content filtering and summarization.
Stores URLs in JSON data and outputs in JSONL format for RAG use.

USAGE:
    python website_downloader_bfs.py <URL> [OPTIONS]

REQUIRED ARGUMENTS:
    URL                    The starting URL to crawl from (e.g., https://example.com)

OPTIONAL ARGUMENTS:
    -o, --output-dir DIR   Output directory to save downloaded pages (default: downloaded_website)
    -d, --max-depth NUM    Maximum depth to crawl recursively (default: 3)
    --max-pages NUM        Maximum number of pages to crawl (default: unlimited)
    -c, --max-concurrent NUM  Maximum number of concurrent requests (default: 30)
    --llm-provider STR     LLM provider string (default: openai/qwen-turbo)
    --llm-api-key STR      API key for the LLM provider (default: OPENAI_API_KEY or DASHSCOPE_API_KEY env var)
    --llm-base-url STR     Base URL for the LLM API endpoint (default: DashScope compatible endpoint)
    --no-llm               Disable LLM extraction and use standard markdown
    --json                 Generate JSON files from LLM extraction (no markdown files)
    --consolidate-output   Consolidate all JSONL data into a single file and delete individual JSONL files
    -h, --help             Show this help message

EXAMPLES:
    # Download a website with LLM extraction (default)
    python website_downloader_bfs.py https://docs.crawl4ai.com

    # Download with custom output directory and depth
    python website_downloader_bfs.py https://example.com -o my_site -d 2

    # Download with page limit
    python website_downloader_bfs.py https://example.com --max-pages 50

    # Download with increased concurrency for faster crawling
    python website_downloader_bfs.py https://example.com -c 20

    # Download with different LLM provider
    python website_downloader_bfs.py https://example.com --llm-provider ollama/llama2

    # Download with custom LLM base URL
    python website_downloader_bfs.py https://example.com --llm-base-url https://custom-llm-endpoint.com/v1

    # Download without LLM extraction (standard markdown only)
    python website_downloader_bfs.py https://example.com --no-llm

    # Consolidate output into a single JSONL file
    python website_downloader_bfs.py https://example.com --consolidate-output

    # Generate JSON files only (no markdown) using LLM extraction
    python website_downloader_bfs.py https://example.com --json

    # Generate JSON files only and consolidate into single file
    python website_downloader_bfs.py https://example.com --json --consolidate-output

OUTPUT:
    The script creates markdown files in the output directory:
    
    With LLM extraction (default):
    - *.md files contain structured content with title, summary, key points, and main content
    - *.jsonl files contain the extracted data in JSONL format for RAG use
    - consolidated_data.jsonl contains all extracted data in a single file (when --consolidate-output is used)
    
    Without LLM extraction (--no-llm):
    - *.md files contain standard markdown
    
    Files are organized in a directory structure that mirrors the website's URL structure.
    The JSONL format includes the URL in each entry for easy reference and RAG applications.

ENVIRONMENT VARIABLES:
    DASHSCOPE_API_KEY      API key for DashScope models (if not provided via --llm-api-key)
    OPENAI_API_KEY         API key for OpenAI models (if not provided via --llm-api-key)
"""
    print(help_text)


async def main():
    """Main function to run the website downloader."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Download an entire website using BFS strategy and save pages in markdown format.",
        add_help=False  # We'll handle help manually to show our custom help
    )
    
    parser.add_argument("url", nargs="?", help="The starting URL to crawl from")
    parser.add_argument("-o", "--output-dir", default="downloaded_website",
                       help="Output directory to save downloaded pages (default: downloaded_website)")
    parser.add_argument("-d", "--max-depth", type=int, default=3,
                       help="Maximum depth to crawl recursively (default: 3)")
    parser.add_argument("--max-pages", type=int, default=None,
                       help="Maximum number of pages to crawl (default: unlimited)")
    parser.add_argument("-c", "--max-concurrent", type=int, default=30,
                       help="Maximum number of concurrent requests (default: 30)")
    parser.add_argument("--llm-provider", default="openai/qwen-turbo",
                       help="LLM provider string (default: openai/qwen-turbo)")
    parser.add_argument("--llm-api-key", default=None,
                       help="API key for the LLM provider (default: DASHSCOPE_API_KEY or OPENAI_API_KEY env var)")
    parser.add_argument("--llm-base-url", default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                       help="Base URL for the LLM API endpoint (default: DashScope compatible endpoint)")
    parser.add_argument("--no-llm", action="store_true",
                       help="Disable LLM extraction and use standard markdown")
    parser.add_argument("--json", action="store_true",
                       help="Generate JSON files from LLM extraction (no markdown files)")
    parser.add_argument("--consolidate-output", action="store_true",
                       help="Consolidate all JSONL data into a single file and delete individual JSONL files")
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
    
    if args.max_pages is not None and args.max_pages < 1:
        print("Error: Max pages must be at least 1")
        return

    # Validate JSON flag conflicts
    if args.json and args.no_llm:
        print("Error: --json flag cannot be used with --no-llm. LLM extraction is required for JSON generation.")
        return
    
    # Configuration
    config = {
        "base_url": args.url,
        "output_dir": args.output_dir,
        "max_depth": args.max_depth,
        "max_pages": args.max_pages,
        "max_concurrent": args.max_concurrent,
        "llm_provider": args.llm_provider,
        "llm_api_key": args.llm_api_key,
        "llm_base_url": args.llm_base_url,
        "use_llm_extraction": not args.no_llm,
        "json_output_only": args.json,
        "consolidate_output": args.consolidate_output
    }
    
    print("Website Downloader with LLM Extraction (BFS Strategy)")
    print("=" * 50)
    print(f"Base URL: {config['base_url']}")
    print(f"Output Directory: {config['output_dir']}")
    print(f"Max Depth: {config['max_depth']}")
    print(f"Max Pages: {config['max_pages'] if config['max_pages'] else 'Unlimited'}")
    print(f"Max Concurrent: {config['max_concurrent']}")
    print(f"LLM Provider: {config['llm_provider']}")
    print(f"LLM Base URL: {config['llm_base_url']}")
    print(f"LLM Extraction: {'Enabled' if config['use_llm_extraction'] else 'Disabled'}")
    print(f"Consolidate Output: {'Enabled' if config['consolidate_output'] else 'Disabled'}")
    if config['use_llm_extraction'] and not config['llm_api_key']:
        print("Note: Using DASHSCOPE_API_KEY or OPENAI_API_KEY environment variable for LLM authentication")
    print()
    
    downloader = WebsiteDownloaderBFS(**config)
    await downloader.crawl_with_bfs()


if __name__ == "__main__":
    asyncio.run(main())