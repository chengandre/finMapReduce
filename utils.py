import os
import re
import time
import json
import json5
import tiktoken
import threading
import subprocess
import uuid
import logging
import pickle
import hashlib

from pathlib import Path
from pydantic import SecretStr
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, PDFMinerLoader, PyMuPDFLoader, UnstructuredPDFLoader
from langchain.prompts import load_prompt
import yaml

load_dotenv()

# Suppress langchain text splitter warnings
logging.getLogger("langchain_text_splitters.base").setLevel(logging.ERROR)

class GPT:
    def __init__(self, model_name, temperature, max_tokens, provider="openrouter", key=None):
        """
        Initialize an LLM wrapper for either OpenAI or OpenRouter

        Args:
            model_name (str): The model to use
            temperature (float): Temperature setting for generation
            max_tokens (int): Maximum tokens for completion
            provider (str): Either "openai" or "openrouter"
            key (str): Key selector - if "self" and provider is not "openrouter", uses SELF_OPENAI_API_KEY, else OPENAI_API_KEY
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = provider.lower()
        self.key = key

        # Configure based on provider
        if provider == "openrouter":
            api_key = SecretStr(os.getenv("OPENROUTER_API_KEY", ""))
            base_url = "https://openrouter.ai/api/v1"
        else:  # default to openai
            if key == "self":
                api_key = SecretStr(os.getenv("SELF_OPENAI_API_KEY", ""))
            elif key == "elm":
                api_key = SecretStr(os.getenv("ELM_OPENAI_API_KEY", ""))
            else:
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

        # Create directory for saving prompts if it doesn't exist
        self.prompts_dir = Path("prompts_log")
        self.prompts_dir.mkdir(exist_ok=True)

    def get_model_name(self):
        """Get the model name."""
        return self.model_name

    def get_temperature(self):
        """Get the temperature setting."""
        return self.temperature

    def get_max_tokens(self):
        """Get the max tokens setting."""
        return self.max_tokens

    def get_provider(self):
        """Get the provider setting."""
        return self.provider

    def get_key(self):
        """Get the key setting."""
        return self.key

    def __call__(self, prompt, **kwargs):
        max_retries = 50
        base_delay = 2

        # Generate unique ID for this prompt
        prompt_id = str(uuid.uuid4())
        prompt_file = self.prompts_dir / f"prompt_{prompt_id}.json"

        # Save the prompt to file
        prompt_data = {
            "kwargs": kwargs,
            "model": self.model_name,
            "timestamp": time.time()
        }

        with open(prompt_file, "w", encoding="utf-8") as f:
            json.dump(prompt_data, f, indent=2)

        for attempt in range(max_retries):
            try:
                chain = prompt | self.llm
                response = chain.invoke(kwargs)

                parsed_json = self._parse_json(response)

                # Success - delete the prompt file
                if prompt_file.exists():
                    prompt_file.unlink()

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
                    # Keep the prompt file for analysis (don't delete it)
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
        except Exception as e:
            try:
                return json5.loads(content)
            except Exception as e:
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
                except Exception as e:
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
                    except Exception as e:
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


class DualRateLimiter:
    """
    Thread-safe rate limiter managing both request and token limits using token bucket algorithm.

    Enforces both requests-per-minute and tokens-per-minute limits with configurable burst sizes.
    Designed for development and testing with multiple concurrent API calls.
    """

    def __init__(self, config):
        """
        Initialize the dual rate limiter.

        Args:
            config (dict): Configuration with keys:
                - requests_per_minute: Maximum requests per minute
                - tokens_per_minute: Maximum tokens per minute
                - request_burst_size: Maximum burst requests (optional, default: 10)
                - token_burst_size: Maximum burst tokens (optional, auto-calculated)
        """
        # Configuration
        self.requests_per_minute = config['requests_per_minute']
        self.tokens_per_minute = config['tokens_per_minute']
        self.request_burst_size = config.get('request_burst_size', 10)
        self.token_burst_size = config.get('token_burst_size',
                                          int(self.tokens_per_minute / 60 * 2))  # 2 seconds of tokens

        # Bucket state
        self.request_bucket = float(self.request_burst_size)
        self.token_bucket = float(self.token_burst_size)
        self.last_refill_time = time.time()

        # Thread safety
        self._lock = threading.Lock()

        # Metrics
        self.stats = {
            'total_requests': 0,
            'total_tokens': 0,
            'total_wait_time': 0.0,
            'request_limited_count': 0,
            'token_limited_count': 0
        }

    def _refill_buckets(self):
        """
        Refill buckets based on TOTAL elapsed time since last refill.
        Must be called within lock.
        """
        now = time.time()
        elapsed = now - self.last_refill_time  # Full elapsed time, not capped

        # Calculate tokens to add for entire elapsed period
        request_tokens_to_add = elapsed * (self.requests_per_minute / 60.0)
        token_tokens_to_add = elapsed * (self.tokens_per_minute / 60.0)

        # Add tokens but cap at burst size
        self.request_bucket = min(
            self.request_bucket + request_tokens_to_add,
            self.request_burst_size
        )
        self.token_bucket = min(
            self.token_bucket + token_tokens_to_add,
            self.token_burst_size
        )

        self.last_refill_time = now

    def wait_for_permission(self, tokens_needed):
        """
        Wait for permission to make a request with the specified token count.

        Thread-safe method that blocks until both request and token resources are available.

        Args:
            tokens_needed (int): Number of tokens required for the request

        Raises:
            ValueError: If tokens_needed exceeds token_burst_size
        """
        if tokens_needed > self.token_burst_size:
            raise ValueError(f"Request requires {tokens_needed} tokens but burst size is only {self.token_burst_size}")

        start_time = time.time()
        was_throttled = False
        limiting_factor = None

        while True:
            with self._lock:
                self._refill_buckets()

                if self.request_bucket >= 1 and self.token_bucket >= tokens_needed:
                    # Consume resources
                    self.request_bucket -= 1
                    self.token_bucket -= tokens_needed

                    # Update stats
                    self.stats['total_requests'] += 1
                    self.stats['total_tokens'] += int(tokens_needed)
                    wait_time = time.time() - start_time
                    self.stats['total_wait_time'] += wait_time

                    # Only increment throttle counters if we actually waited
                    if was_throttled:
                        if limiting_factor == 'requests':
                            self.stats['request_limited_count'] += 1
                        else:
                            self.stats['token_limited_count'] += 1

                    return

                # Calculate wait time
                request_wait = 0 if self.request_bucket >= 1 else \
                              (1 - self.request_bucket) * 60 / self.requests_per_minute
                token_wait = 0 if self.token_bucket >= tokens_needed else \
                            (tokens_needed - self.token_bucket) * 60 / self.tokens_per_minute

                wait_time = max(request_wait, token_wait)

                # Track that we're throttling and why (only on first iteration)
                if not was_throttled and wait_time > 0:
                    was_throttled = True
                    limiting_factor = 'requests' if request_wait > token_wait else 'tokens'

            # Sleep outside the lock
            if wait_time > 0:
                time.sleep(wait_time)

    def get_stats(self):
        """
        Get current rate limiting statistics in a thread-safe manner.

        Returns:
            dict: Copy of current statistics
        """
        with self._lock:
            return self.stats.copy()


def estimate_tokens(model_name, text, max_tokens=None):
    """
    Estimates total token usage for a request.

    Args:
        model_name (str): The model being used
        text (str): The prompt text
        max_tokens (int, optional): Maximum completion tokens (if specified)

    Returns:
        int: Estimated total tokens (prompt + completion + buffer)
    """
    try:
        # Use cl100k_base encoding for all models
        encoding = tiktoken.get_encoding("cl100k_base")
        prompt_tokens = len(encoding.encode(text))

        # Add completion tokens if specified
        completion_tokens = max_tokens if max_tokens else 0

        # Calculate total with 15% safety buffer
        total_tokens = int((prompt_tokens + completion_tokens) * 1.15)

        return total_tokens

    except Exception:
        # Fallback to character-based estimation with safety buffer
        prompt_tokens = len(text) // 4
        completion_tokens = max_tokens if max_tokens else 0
        return int((prompt_tokens + completion_tokens) * 1.15)


class RetryLLM:
    """
    LLM wrapper with retry mechanism but no JSON parsing.

    Provides the same retry logic as GPT class but returns raw responses
    without attempting to parse JSON. Useful when you want direct access
    to the LLM response but still need robust error handling.
    """

    def __init__(self, model_name, temperature, max_tokens, provider="openrouter", key=None):
        """
        Initialize RetryLLM wrapper.

        Args:
            model_name (str): The model to use
            temperature (float): Temperature setting for generation
            max_tokens (int): Maximum tokens for completion
            provider (str): Either "openai" or "openrouter"
            key (str): Key selector - if "self" and provider is not "openrouter", uses SELF_OPENAI_API_KEY, else OPENAI_API_KEY
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = provider.lower()
        self.key = key

        # Configure based on provider
        if provider == "openrouter":
            api_key = SecretStr(os.getenv("OPENROUTER_API_KEY", ""))
            base_url = "https://openrouter.ai/api/v1"
        else:  # default to openai
            if key == "self":
                api_key = SecretStr(os.getenv("SELF_OPENAI_API_KEY", ""))
            elif key == "elm":
                api_key = SecretStr(os.getenv("ELM_OPENAI_API_KEY", ""))
            else:
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

        # Create directory for saving prompts if it doesn't exist
        self.prompts_dir = Path("prompts_log")
        self.prompts_dir.mkdir(exist_ok=True)

    def get_model_name(self):
        """Get the model name."""
        return self.model_name

    def get_temperature(self):
        """Get the temperature setting."""
        return self.temperature

    def get_max_tokens(self):
        """Get the max tokens setting."""
        return self.max_tokens

    def get_provider(self):
        """Get the provider setting."""
        return self.provider

    def get_key(self):
        """Get the key setting."""
        return self.key

    def invoke(self, prompt_text):
        """
        Invoke the LLM with retry logic but no JSON parsing.

        Args:
            prompt_text (str): The formatted prompt text to send to the LLM

        Returns:
            ChatOpenAI response object (with .content and .usage_metadata attributes)
        """
        max_retries = 50
        base_delay = 2

        # Generate unique ID for this prompt
        prompt_id = str(uuid.uuid4())
        prompt_file = self.prompts_dir / f"prompt_{prompt_id}.json"

        # Save the prompt to file
        prompt_data = {
            "prompt_text": prompt_text,
            "model": self.model_name,
            "timestamp": time.time()
        }

        with open(prompt_file, "w", encoding="utf-8") as f:
            json.dump(prompt_data, f, indent=2)

        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(prompt_text)

                # Success - delete the prompt file
                if prompt_file.exists():
                    prompt_file.unlink()

                return response

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
                    "too many requests"
                ]

                is_retryable = any(error_type.lower() in error_str.lower() for error_type in retryable_errors)
                is_retryable = True

                if not is_retryable or attempt == max_retries - 1:
                    # Keep the prompt file for analysis (don't delete it)
                    raise e

                delay = min(base_delay * (2 ** attempt), 60)
                print(f"Error (attempt {attempt + 1}/{max_retries}): {error_str}")
                time.sleep(delay)


class RateLimitedRetryLLM(RetryLLM):
    """
    LLM wrapper with both rate limiting and retry mechanism but no JSON parsing.

    Combines the retry logic from RetryLLM with rate limiting from DualRateLimiter.
    Returns raw responses without attempting to parse JSON.
    """

    def __init__(self, model_name="gpt-4o-mini", temperature=0.0, max_tokens=8000, provider="openai", key="self", rate_limit_config=None):
        """
        Initialize RateLimitedRetryLLM with rate limiting configuration.

        Args:
            model_name (str): The model to use
            temperature (float): Temperature setting for generation
            max_tokens (int): Maximum tokens for completion
            provider (str): Either "openai" or "openrouter"
            key (str): Key selector for API key
            rate_limit_config (dict, optional): Rate limiting configuration
        """
        # Initialize parent RetryLLM class
        super().__init__(model_name, temperature, max_tokens, provider, key)

        # Initialize rate limiter with defaults if not provided
        default_config = {
            'requests_per_minute': 5000,
            'tokens_per_minute': 4000000,
            'request_burst_size': 500
        }
        config = rate_limit_config or default_config
        self.rate_limiter = DualRateLimiter(config)

    def invoke(self, prompt_text):
        """
        Make a rate-limited API call with retry logic but no JSON parsing.

        Args:
            prompt_text (str): The formatted prompt text to send to the LLM

        Returns:
            ChatOpenAI response object (with .content and .usage_metadata attributes)
        """
        # Estimate tokens needed for this request (prompt + max_tokens for completion)
        tokens_needed = estimate_tokens(self.model_name, prompt_text, self.max_tokens)

        # Wait for permission (both request and token limits)
        self.rate_limiter.wait_for_permission(tokens_needed)

        # Make the actual API call (with existing retry logic from parent)
        return super().invoke(prompt_text)

    def get_rate_limit_stats(self):
        """
        Get current rate limiting statistics.

        Returns:
            dict: Rate limiting statistics
        """
        return self.rate_limiter.get_stats()


class RateLimitedGPT(GPT):
    """
    GPT wrapper with dual rate limiting for both requests and tokens.

    Supports both requests-per-minute and tokens-per-minute limits with configurable
    burst sizes. Uses the DualRateLimiter for thread-safe concurrent access.
    """

    def __init__(self, model_name="gpt-4o-mini", temperature=None, max_tokens=8000, provider="openai", key="self", rate_limit_config=None):
        """
        Initialize RateLimitedGPT with rate limiting configuration.

        Args:
            model_name (str): The model to use
            temperature (float): Temperature setting for generation
            max_tokens (int): Maximum tokens for completion
            provider (str): Either "openai" or "openrouter"
            key (str): Key selector for API key
            rate_limit_config (dict, optional): Rate limiting configuration
        """
        # Initialize parent GPT class
        super().__init__(model_name, temperature, max_tokens, provider, key)

        # Initialize rate limiter with defaults if not provided
        default_config = {
            'requests_per_minute': 5000,
            'tokens_per_minute': 4000000,
            'request_burst_size': 500
        }
        config = rate_limit_config or default_config
        self.rate_limiter = DualRateLimiter(config)

    def __call__(self, prompt, **kwargs):
        """
        Make a rate-limited API call.

        Args:
            prompt: LangChain prompt template
            **kwargs: Variables to be formatted into the prompt (e.g., context, question)

        Returns:
            dict: Response with 'json' and 'raw_response' keys
        """
        # Use the prompt's format method to get the full formatted text
        try:
            formatted_prompt_text = prompt.format(**kwargs)
        except (AttributeError, KeyError, ValueError):
            # Fallback: convert to string and estimate conservatively
            prompt_text = str(prompt)
            kwargs_text = ' '.join(str(v) for v in kwargs.values()) if kwargs else ''
            formatted_prompt_text = prompt_text + ' ' + kwargs_text

        # Estimate tokens needed for this request (prompt + max_tokens for completion)
        tokens_needed = estimate_tokens(self.model_name, formatted_prompt_text, self.max_tokens)

        # Wait for permission (both request and token limits)
        self.rate_limiter.wait_for_permission(tokens_needed)

        # Make the actual API call (with existing retry logic)
        return super().__call__(prompt, **kwargs)

    def get_rate_limit_stats(self):
        """
        Get current rate limiting statistics.

        Returns:
            dict: Rate limiting statistics
        """
        return self.rate_limiter.get_stats()


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def _resolve_document_path(document_file):
    """
    Resolve a document file input to a valid absolute path.

    Handles multiple input formats:
    - Document name only (e.g., "APPLE_2020") - searches for .pdf and .md files
    - Full path (e.g., "/path/to/APPLE_2020.pdf") - validates existence
    - Relative path - converts to absolute and validates

    Search locations for document names:
    - Current directory
    - ../financebench/data/ (for FinanceBench PDFs)
    - ../edgartools_finqa/ (for FinQA markdown files)

    Args:
        document_file (str): Document name, relative path, or absolute path

    Returns:
        Path: Absolute path to the resolved document file

    Raises:
        FileNotFoundError: If document cannot be found in any search location
    """
    document_path = Path(document_file)

    # If it's already an absolute path, just validate it exists
    if document_path.is_absolute():
        if document_path.exists():
            return document_path
        else:
            raise FileNotFoundError(f"Document not found: {document_file}")

    # If it has an extension and exists relative to current dir, use it
    if document_path.suffix and document_path.exists():
        return document_path.resolve()

    # If it's just a name without extension, search for it
    if not document_path.suffix:
        search_locations = [
            Path.cwd(),  # Current directory
            Path("../financebench/pdfs"),  # FinanceBench PDFs
            Path("../edgartools_finqa"),   # FinQA markdown files
        ]

        # Try both .pdf and .md extensions
        extensions = ['.pdf', '.md', '.markdown']

        for location in search_locations:
            if location.exists():
                for ext in extensions:
                    candidate = location / f"{document_file}{ext}"
                    if candidate.exists():
                        return candidate.resolve()

    # If it has an extension, search in known locations
    if document_path.suffix:
        search_locations = [
            Path.cwd(),
            Path("../financebench/pdfs"),
            Path("../edgartools_finqa"),
        ]

        for location in search_locations:
            if location.exists():
                candidate = location / document_path.name
                if candidate.exists():
                    return candidate.resolve()

    # Not found anywhere
    raise FileNotFoundError(
        f"Document '{document_file}' not found. Searched in: current directory, "
        f"../financebench/data, ../edgartools_finqa. "
        f"Please provide either a valid document name (e.g., 'APPLE_2020') or full path."
    )


def _get_document_cache_path(document_file, method, chunk_size, chunk_overlap):
    """
    Generate a cache file path for document parsing results based on file content and parameters.

    Args:
        document_file (str or Path): Path to the document file (PDF or Markdown)
        method (str): Document parsing method used
        chunk_size (int): Chunk size parameter
        chunk_overlap (int): Chunk overlap parameter

    Returns:
        Path: Path to the cache file
    """
    # Create a hash of the file content + parameters for cache key
    document_path = Path(document_file)

    # Get file modification time and size for cache validity
    try:
        stat = document_path.stat()
        file_info = f"{stat.st_mtime}_{stat.st_size}"
    except (OSError, FileNotFoundError):
        file_info = "unknown"

    # Create hash of filename, method, parameters, and file info
    cache_key = f"{document_path.name}_{method}_{chunk_size}_{chunk_overlap}_{file_info}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()

    # Create cache directory structure - use document_cache instead of pdf_cache
    cache_dir = Path("document_cache") / method
    cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir / f"{document_path.stem}_{cache_hash}.pkl"




def _save_document_cache(cache_path, documents, token_count):
    """
    Save document parsing results to cache.

    Args:
        cache_path (Path): Path to save the cache file
        documents (list): List of Document objects
        token_count (int): Token count for the documents
    """
    try:
        cache_data = {
            'documents': documents,
            'token_count': token_count,
            'timestamp': time.time()
        }

        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)

        # print(f"Saved document parsing results to cache: {cache_path}")

    except Exception as e:
        print(f"Warning: Failed to save document cache: {e}")


def _load_document_cache(cache_path):
    """
    Load document parsing results from cache.

    Args:
        cache_path (Path): Path to the cache file

    Returns:
        tuple: (documents, token_count) or (None, None) if cache invalid/missing
    """
    try:
        if not cache_path.exists():
            return None, None

        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)

        documents = cache_data.get('documents', [])
        token_count = cache_data.get('token_count', 0)

        print(f"Loaded document parsing results from cache: {cache_path}")
        return documents, token_count

    except Exception as e:
        print(f"Warning: Failed to load document cache: {e}")
        return None, None




def _marker_parser(pdf_file, force_reparse=False):
    """
    Parse a PDF file using the marker CLI tool and convert to markdown.
    Checks if the PDF file is from financeBench or has been parsed already.

    Args:
        pdf_file (str or Path): Path to the PDF file to be processed (should be absolute path)
        force_reparse (bool): Whether to reparse the PDF even if markdown already exists

    Returns:
        str: Path to the generated markdown file, or None if parsing failed
    """
    # Ensure we have a Path object and get the document name
    pdf_path = Path(pdf_file)
    pdf_name = pdf_path.stem

    # Check if pdf is already parsed from financeBench
    financebench_markdown_path = Path("../marker_financebench") / pdf_name / f"{pdf_name}.md"
    if not force_reparse and financebench_markdown_path.exists():
        # print(f"Found existing financeBench markdown for {pdf_name}: {financebench_markdown_path}")
        return str(financebench_markdown_path)

    # Check local marker output directory
    local_markdown_dir = Path("marker") / pdf_name
    local_markdown_file = local_markdown_dir / f"{pdf_name}.md"

    if not force_reparse and local_markdown_file.exists():
        print(f"Found existing marker markdown for {pdf_name}: {local_markdown_file}")
        return str(local_markdown_file)

    # Create output directory for this document
    local_markdown_dir.mkdir(parents=True, exist_ok=True)

    # Run marker CLI command with absolute path
    try:
        output_dir = Path("marker")
        print(f"Parsing {pdf_path} with marker...")
        cmd = ["marker_single", str(pdf_path), "--output_dir", str(output_dir), "--output_format", "markdown", "--format_lines"]
        subprocess.run(cmd, check=True)

        # Check if markdown was generated
        if local_markdown_file.exists():
            print(f"Successfully parsed {pdf_path}. Markdown saved to {local_markdown_file}")
            return str(local_markdown_file)
        else:
            print(f"Marker didn't generate markdown for {pdf_path}")
            return None
    except Exception as e:
        print(f"Error parsing {pdf_path} with marker: {e}")
        return None


def _process_documents(documents, chunk_size, chunk_overlap, use_tiktoken=False):
    """Helper function to count tokens and split documents."""
    # Calculate token count from document content
    content = str(documents) if len(documents) > 1 else documents[0].page_content
    token_count = num_tokens_from_string(content, "cl100k_base")

    # Choose appropriate text splitter
    if use_tiktoken:
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
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


def load_document_chunk(document_file, chunk_size, chunk_overlap, method=None, force_reparse=False):
    """
    Load and chunk a document file (PDF or Markdown) for processing.

    Unified interface that auto-detects file type and handles path resolution.
    Supports both document names (e.g., "APPLE_2020") and full paths.

    Args:
        document_file (str): Document name, relative path, or absolute path
        chunk_size (int): Size of each chunk (must be > 0)
        chunk_overlap (int): Overlap between chunks (must be >= 0 and < chunk_size)
        method (str, optional): Parsing method. Auto-detected from file extension if None.
                               For PDFs: "marker", "pypdf", "pymu", "unstructured", "default"
                               For Markdown: "markdown"
        force_reparse (bool): Whether to bypass cache and reparse the document

    Returns:
        tuple: (list of Document objects, token count)

    Examples:
        # Document name - will search for APPLE_2020.pdf or APPLE_2020.md
        load_document_chunk("APPLE_2020", 1000, 200)

        # Full path
        load_document_chunk("/path/to/document.pdf", 1000, 200, method="marker")

        # Markdown file
        load_document_chunk("report.md", 1000, 200)
    """
    # Validate parameters
    if not document_file or not document_file.strip():
        raise ValueError("document_file cannot be empty")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    # Resolve document path
    try:
        resolved_path = _resolve_document_path(document_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return [], 0

    # Auto-detect method from file extension if not specified
    if method is None:
        extension = resolved_path.suffix.lower()
        if extension == '.pdf':
            method = "marker"  # Default to marker for PDFs
        elif extension in ['.md', '.markdown']:
            method = "markdown"
        else:
            raise ValueError(f"Unsupported file type: {extension}. Supported: .pdf, .md, .markdown")

    # Validate method parameter
    valid_pdf_methods = ["marker", "pypdf", "pymu", "unstructured", "default"]
    valid_markdown_methods = ["markdown"]
    extension = resolved_path.suffix.lower()

    if extension == '.pdf' and method not in valid_pdf_methods:
        raise ValueError(f"Invalid method '{method}' for PDF files. Valid methods: {valid_pdf_methods}")
    elif extension in ['.md', '.markdown'] and method not in valid_markdown_methods:
        raise ValueError(f"Invalid method '{method}' for Markdown files. Valid methods: {valid_markdown_methods}")

    # Handle markdown files
    if method == "markdown" or resolved_path.suffix.lower() in ['.md', '.markdown']:
        return _load_markdown_document(resolved_path, chunk_size, chunk_overlap, force_reparse)

    # Handle PDF files
    return _load_pdf_document(resolved_path, chunk_size, chunk_overlap, method, force_reparse)


def _load_markdown_document(markdown_path, chunk_size, chunk_overlap, force_reparse=False):
    """Load and process a markdown document."""
    # Check cache first (unless forcing reparse)
    if not force_reparse:
        cache_path = _get_document_cache_path(markdown_path, "markdown", chunk_size, chunk_overlap)
        cached_documents, cached_token_count = _load_document_cache(cache_path)
        if cached_documents is not None:
            return cached_documents, cached_token_count

    try:
        with open(markdown_path, 'r', encoding='utf-8') as f:
            content = f.read()

        documents = [Document(page_content=content, metadata={"source": str(markdown_path)})]
        split_documents, token_count = _process_documents(documents, chunk_size, chunk_overlap, use_tiktoken=True)

        # Save to cache
        cache_path = _get_document_cache_path(markdown_path, "markdown", chunk_size, chunk_overlap)
        _save_document_cache(cache_path, split_documents, token_count)

        return split_documents, token_count

    except FileNotFoundError:
        print(f"Warning: File not found: {markdown_path}")
        return [], 0
    except Exception as e:
        print(f"Error loading markdown file {markdown_path}: {e}")
        return [], 0


def _load_pdf_document(pdf_path, chunk_size, chunk_overlap, method, force_reparse=False):
    """Load and process a PDF document using the specified method."""
    documents = None

    # Handle marker method separately due to its unique workflow
    if method == "marker":
        # Try parsing with marker
        markdown_path = _marker_parser(str(pdf_path))
        if markdown_path:
            with open(markdown_path, 'r', encoding='utf-8') as f:
                content = f.read()
            documents = [Document(page_content=content, metadata={"source": str(pdf_path)})]
            return _process_documents(documents, chunk_size, chunk_overlap, use_tiktoken=True)

        # Fallback to PDFMinerLoader
        print("Marker parsing failed, falling back to PDFMinerLoader")
        method = "default"

    # Check cache for non-marker methods (only if not forcing reparse)
    if not force_reparse:
        cache_path = _get_document_cache_path(pdf_path, method, chunk_size, chunk_overlap)
        cached_documents, cached_token_count = _load_document_cache(cache_path)

        if cached_documents is not None:
            return cached_documents, cached_token_count

    # Handle all other loader methods
    pdf_str = str(pdf_path)
    if method == "pypdf":
        loader = PyPDFLoader(pdf_str)
    elif method == "pymu":
        loader = PyMuPDFLoader(pdf_str)
    elif method == "unstructured":
        loader = UnstructuredPDFLoader(pdf_str, mode="elements", strategy="hi_res")
    else:  # default case (including fallback from marker)
        loader = PDFMinerLoader(pdf_str)

    documents = loader.load()
    split_documents, token_count = _process_documents(documents, chunk_size, chunk_overlap, use_tiktoken=True)

    # Save to cache for future use (except for marker method which has its own caching)
    if method != "marker":
        cache_path = _get_document_cache_path(pdf_path, method, chunk_size, chunk_overlap)
        _save_document_cache(cache_path, split_documents, token_count)

    return split_documents, token_count




# ===== EVALUATION STATISTICS FUNCTIONS =====

def calculate_token_usage_summary(qa_data):
    """
    Calculate aggregated token usage statistics across all QA pairs.

    Args:
        qa_data (list): List of QA pair dictionaries with token_stats

    Returns:
        dict: Aggregated token statistics including totals, averages, and efficiency ratio
    """
    if not qa_data:
        return {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cache_read_tokens": 0,
            "total_tokens": 0,
            "avg_input_tokens_per_question": 0,
            "avg_output_tokens_per_question": 0,
            "avg_cache_read_tokens_per_question": 0,
            "token_efficiency_ratio": 0
        }

    total_input = sum(qa.get("token_stats", {}).get("total", {}).get("input_tokens", 0) for qa in qa_data)
    total_output = sum(qa.get("token_stats", {}).get("total", {}).get("output_tokens", 0) for qa in qa_data)
    total_cache_read = sum(qa.get("token_stats", {}).get("total", {}).get("cache_read_tokens", 0) for qa in qa_data)

    return {
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cache_read_tokens": total_cache_read,
        "total_tokens": total_input + total_output,
        "avg_input_tokens_per_question": total_input / len(qa_data),
        "avg_output_tokens_per_question": total_output / len(qa_data),
        "avg_cache_read_tokens_per_question": total_cache_read / len(qa_data),
        "token_efficiency_ratio": total_output / total_input if total_input > 0 else 0
    }


def calculate_accuracy_by_question_type(qa_data):
    """
    Calculate accuracy metrics broken down by question_type.

    Args:
        qa_data (list): List of QA pair dictionaries with question_type and judgment fields

    Returns:
        dict: Nested dictionary with accuracy stats per question type
    """
    question_type_stats = {}

    for qa in qa_data:
        q_type = qa.get("question_type", "unknown")
        judgment = qa.get("judgment", "unknown").lower()

        if q_type not in question_type_stats:
            question_type_stats[q_type] = {
                "correct": 0, "coherent": 0, "deviated": 0,
                "incorrect": 0, "no_answer": 0, "total": 0
            }

        # Normalize judgment to match dictionary keys
        if judgment == "correct":
            question_type_stats[q_type]["correct"] += 1
        elif judgment == "coherent":
            question_type_stats[q_type]["coherent"] += 1
        elif judgment == "deviated":
            question_type_stats[q_type]["deviated"] += 1
        elif judgment == "incorrect":
            question_type_stats[q_type]["incorrect"] += 1
        elif judgment == "no answer" or judgment == "no_answer":
            question_type_stats[q_type]["no_answer"] += 1

        question_type_stats[q_type]["total"] += 1

    # Calculate accuracy for each question type
    for q_type in question_type_stats:
        stats = question_type_stats[q_type]
        if stats["total"] > 0:
            stats["accuracy"] = stats["correct"] / stats["total"]
        else:
            stats["accuracy"] = 0

    return question_type_stats


def calculate_accuracy_by_question_reasoning(qa_data):
    """
    Calculate accuracy metrics broken down by question_reasoning.

    Args:
        qa_data (list): List of QA pair dictionaries with question_reasoning and judgment fields

    Returns:
        dict: Nested dictionary with accuracy stats per question reasoning category
    """
    reasoning_stats = {}

    for qa in qa_data:
        q_reasoning = qa.get("question_reasoning")
        # Handle null/None values
        if q_reasoning is None:
            q_reasoning = "null"

        judgment = qa.get("judgment", "unknown").lower()

        if q_reasoning not in reasoning_stats:
            reasoning_stats[q_reasoning] = {
                "correct": 0, "coherent": 0, "deviated": 0,
                "incorrect": 0, "no_answer": 0, "total": 0
            }

        # Normalize judgment to match dictionary keys
        if judgment == "correct":
            reasoning_stats[q_reasoning]["correct"] += 1
        elif judgment == "coherent":
            reasoning_stats[q_reasoning]["coherent"] += 1
        elif judgment == "deviated":
            reasoning_stats[q_reasoning]["deviated"] += 1
        elif judgment == "incorrect":
            reasoning_stats[q_reasoning]["incorrect"] += 1
        elif judgment == "no answer" or judgment == "no_answer":
            reasoning_stats[q_reasoning]["no_answer"] += 1

        reasoning_stats[q_reasoning]["total"] += 1

    # Calculate accuracy for each reasoning category
    for q_reasoning in reasoning_stats:
        stats = reasoning_stats[q_reasoning]
        if stats["total"] > 0:
            stats["accuracy"] = stats["correct"] / stats["total"]
        else:
            stats["accuracy"] = 0.0

    return reasoning_stats


def load_prompt_set(prompt_set_name=None):
    """
    Load all prompts for a prompt set and return loaded prompt objects

    Args:
        prompt_set_name (str): Name of the prompt set to load (default, old, last_year, etc.)

    Returns:
        dict: Dictionary containing pre-loaded prompt objects with keys:
              'map_prompt', 'reduce_prompt', 'judge_prompt'
    """
    config_path = "prompts/prompt_config.yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if prompt_set_name is None:
        prompt_set_name = config.get('default_set', 'default')

    if prompt_set_name not in config['prompt_sets']:
        available_sets = list(config['prompt_sets'].keys())
        raise ValueError(f"Unknown prompt set '{prompt_set_name}'. Available: {available_sets}")

    prompt_files = config['prompt_sets'][prompt_set_name]

    # Pre-load all prompt objects
    loaded_prompts = {
        'map_prompt': load_prompt(prompt_files['map_prompt']),
        'reduce_prompt': load_prompt(prompt_files['reduce_prompt']),
        'judge_prompt': load_prompt(prompt_files['judge_prompt']),
        'prompt_set_name': prompt_set_name
    }

    # Load question improvement prompt if available
    if 'question_improvement_prompt' in prompt_files:
        loaded_prompts['question_improvement_prompt'] = load_prompt(prompt_files['question_improvement_prompt'])

    return loaded_prompts