import os
import re
import time
import json
import json5
import tiktoken
import threading
import subprocess

from pathlib import Path
from pydantic import SecretStr
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, PDFMinerLoader, PyMuPDFLoader, UnstructuredPDFLoader

load_dotenv()

class GPT:
    def __init__(self, model_name, temperature, max_tokens, provider="openrouter"):
        """
        Initialize an LLM wrapper for either OpenAI or OpenRouter

        Args:
            model_name (str): The model to use
            temperature (float): Temperature setting for generation
            max_tokens (int): Maximum tokens for completion
            provider (str): Either "openai" or "openrouter"
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = provider.lower()

        # Configure based on provider
        if provider == "openrouter":
            api_key = SecretStr(os.getenv("OPENROUTER_API_KEY", ""))
            base_url = "https://openrouter.ai/api/v1"
        else:  # default to openai
            api_key = SecretStr(os.getenv("OPENAI_API_KEY", ""))
            base_url = None

        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            max_retries=50
        )

    def __call__(self, prompt, **kwargs):
        max_retries = 50
        base_delay = 2

        for attempt in range(max_retries):
            try:
                chain = prompt | self.llm
                response = chain.invoke(kwargs)

                parsed_json = self._parse_json(response)
                return {
                    'json': parsed_json,
                    'raw_response': response
                }

            except Exception as e:
                error_str = str(e)

                retryable_errors = [
                    "Internal Server Error",
                    "500",
                    "502",
                    "503",
                    "504",
                    "timeout",
                    "connection",
                    "rate limit",
                    "too many requests",
                    "json",
                    "invalid json"
                ]

                is_retryable = any(error_type.lower() in error_str.lower() for error_type in retryable_errors)
                is_retryable = True

                if not is_retryable or attempt == max_retries - 1:
                    raise e

                delay = min(base_delay * (2 ** attempt), 60)
                print(f"Error (attempt {attempt + 1}/{max_retries}): {error_str}")
                # print(f"Error (attempt {attempt + 1}/{max_retries})")
                # print(f"Retrying in {delay} seconds...")
                time.sleep(delay)

    def _parse_json(self, response):
        """
        Parse JSON from the response, handling various formats including code blocks
        """
        content = response.content if hasattr(response, 'content') else str(response)

        try:
            # Try to parse the entire content as JSON first
            return json.loads(content)
        except json.JSONDecodeError:
            try:
                return json5.loads(content)
            except json.JSONDecodeError:
                pass

        # Try to extract JSON from code blocks (```json ... ``` or ``` ... ```)
        code_block_patterns = [
            r'```json\s*\n(.*?)\n```',  # ```json ... ```
            r'```\s*\n(.*?)\n```',      # ``` ... ```
            r'`(.*?)`'                   # `...` (single backticks)
        ]

        for pattern in code_block_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    return json5.loads(match.strip())
                except json.JSONDecodeError:
                    continue

        # Try to find JSON object patterns in the response
        try:
            # Look for JSON object patterns (handles nested objects better)
            json_pattern = r'\{(?:[^{}]|{[^{}]*})*\}'
            matches = re.findall(json_pattern, content, re.DOTALL)

            if matches:
                # Try to parse each match, starting with the largest
                matches.sort(key=len, reverse=True)
                for match in matches:
                    try:
                        return json5.loads(match)
                    except json.JSONDecodeError:
                        continue

        except (json.JSONDecodeError, ValueError):
            pass

        # If we get here, JSON parsing failed completely
        raise ValueError(f"Invalid JSON response. Content:\n{content}")


class TokenBucketRateLimiter:
    """Rate limiter using token bucket algorithm"""
    
    def __init__(self, calls_per_minute=20, burst_size=8):
        """
        Args:
            calls_per_minute: Sustained rate limit (20 calls/minute)
            burst_size: Maximum burst capacity (8 calls instantly)
        """
        self.rate = calls_per_minute / 60.0  # Convert to tokens per second
        self.burst_size = burst_size         # Maximum tokens in bucket
        self.tokens = float(burst_size)      # Start with full bucket
        self.last_update = time.time()       # Last time we added tokens
        self.lock = threading.Lock()         # Thread safety
    
    def acquire_token(self):
        """Wait if necessary to respect rate limit (iterative approach)"""
        while True:
            with self.lock:
                now = time.time()
                elapsed = now - self.last_update
                tokens_to_add = elapsed * self.rate
                self.tokens = min(self.burst_size, self.tokens + tokens_to_add)
                self.last_update = now

                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return  # We have a token, proceed

                # Calculate wait time if no token is available
                tokens_needed = 1.0 - self.tokens
                wait_time = tokens_needed / self.rate
            
            if wait_time > 0:
                time.sleep(wait_time)


class RateLimitedGPT(GPT):
    """GPT wrapper with shared token bucket rate limiting"""
    
    # Class-level rate limiter shared across all instances
    _rate_limiter = TokenBucketRateLimiter(
        calls_per_minute=500,    # Conservative API limit
        burst_size=100           # Allow bursts for chunk processing
    )
    
    def __call__(self, prompt, **kwargs):
        # Wait for token before making API call
        self._rate_limiter.acquire_token()
        
        # Make the actual API call (with existing retry logic)
        return super().__call__(prompt, **kwargs)


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def _marker_parser(pdf_file, force_reparse=False):
    """
    Parse a PDF file using the marker CLI tool and convert to markdown.
    Checks if the PDF file is from financeBench or has been parsed already.

    Args:
        pdf_file (str): Path to the PDF file to be processed
        force_reparse (bool): Whether to reparse the PDF even if markdown already exists

    Returns:
        str: Path to the generated markdown file
    """
    # Get PDF filename without extension
    pdf_path = Path(pdf_file)
    pdf_name = pdf_path.stem

    # Check if pdf is already parsed from financeBench
    markdown_path = ".." / Path("marker_financebench") / pdf_name / f"{pdf_name}.md"
    if not force_reparse and markdown_path.exists():
        # print(f"Found existing financeBench markdown for {pdf_name}: {markdown_path}")
        return str(markdown_path)

    # Create output directory if it doesn't exist
    markdown_path = Path("marker") / pdf_name
    markdown_path.mkdir(parents=True, exist_ok=True)
    if not force_reparse and markdown_path.exists():
        print(f"Found existing marker markdown for {pdf_name}: {markdown_path}")
        return str(markdown_path)

    # Run marker CLI command
    try:
        output_dir = Path("marker")
        print(f"Parsing {pdf_file} with marker...")
        cmd = ["marker_single", pdf_file, "--output_dir", str(output_dir), "--output_format", "markdown", "--format_lines"]
        subprocess.run(cmd, check=True)

        # Check if markdown was generated
        if markdown_path.exists():
            print(f"Successfully parsed {pdf_file}. Markdown saved to {markdown_path}")
            return str(markdown_path)
        else:
            print(f"Marker didn't generate markdown for {pdf_file}")
            return None
    except Exception as e:
        print(f"Error parsing {pdf_file} with marker: {e}")
        return None


def _process_documents(documents, chunk_size, chunk_overlap, use_tiktoken=False):
    """Helper function to count tokens and split documents."""
    # Calculate token count from document content
    content = str(documents) if len(documents) > 1 else documents[0].page_content
    token_count = num_tokens_from_string(content, "cl100k_base")

    # Choose appropriate text splitter
    if use_tiktoken:
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    else:
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    split_documents = text_splitter.split_documents(documents)
    # print(f'PDF Token Count: {token_count}, Document Count: {len(split_documents)}')
    return split_documents, token_count


def load_pdf_chunk(pdf_file, chunk_size, chunk_overlap, method):
    """
    The function loads a pdf file and makes it ready for querying

    Args:
        pdf_file (str): name of the pdf file to be processed
        chunk_size (int): size of each chunk
        chunk_overlap (int): overlap between chunks
        method (str): method to use for PDF parsing

    Returns:
        pages (list): list of all the pages in the pdf
    """
    documents = None

    # Handle marker method separately due to its unique workflow
    if method == "marker":

        # Try parsing with marker
        markdown_path = _marker_parser(pdf_file)
        if markdown_path:
            with open(markdown_path, 'r', encoding='utf-8') as f:
                content = f.read()
            documents = [Document(page_content=content, metadata={"source": pdf_file})]
            return _process_documents(documents, chunk_size, chunk_overlap, use_tiktoken=True)

        # Fallback to PDFMinerLoader
        print("Marker parsing failed, falling back to PDFMinerLoader")
        method = "default"

    # Handle all other loader methods
    if method == "Load page-wise PDF":
        loader = PyPDFLoader(pdf_file)
    elif method == "pymu":
        loader = PyMuPDFLoader(pdf_file)
    elif method == "unstructured":
        loader = UnstructuredPDFLoader(pdf_file, mode="elements", strategy="hi_res")
    else:  # default case (including fallback from marker)
        loader = PDFMinerLoader(pdf_file)

    documents = loader.load()
    return _process_documents(documents, chunk_size, chunk_overlap, use_tiktoken=True)