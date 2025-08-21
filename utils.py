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
import yaml

from pathlib import Path
from pydantic import SecretStr
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, PDFMinerLoader, PyMuPDFLoader, UnstructuredPDFLoader
from langchain.prompts import load_prompt
from typing import Optional, Dict, Any, Protocol, runtime_checkable
from dataclasses import dataclass

load_dotenv()

# Suppress langchain text splitter warnings
logging.getLogger("langchain_text_splitters.base").setLevel(logging.ERROR)

# ============================================================================
# Configuration and Data Classes
# ============================================================================

@dataclass
class LLMConfig:
    """Configuration for LLM models"""
    model_name: str
    temperature: float
    max_tokens: int
    provider: str = "openai"
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    requests_per_minute: int = 5000
    tokens_per_minute: int = 4000000
    request_burst_size: int = 500
    token_burst_size: Optional[int] = None

    def __post_init__(self):
        if self.token_burst_size is None:
            self.token_burst_size = int(self.tokens_per_minute / 60 * 2)


# ============================================================================
# Provider Factory
# ============================================================================

class LLMProviderFactory:
    """Factory for creating configured LLM providers"""

    @staticmethod
    def create_provider(config: LLMConfig) -> ChatOpenAI:
        """Create a configured LLM provider based on config"""
        api_key, base_url = LLMProviderFactory._get_credentials(config)

        return ChatOpenAI(
            model=config.model_name,
            api_key=api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            base_url=base_url,
            max_retries=50
        )

    @staticmethod
    def _get_credentials(config: LLMConfig) -> tuple[SecretStr, Optional[str]]:
        """Get API credentials based on provider and key configuration"""
        if config.provider.lower() == "openrouter":
            api_key = SecretStr(os.getenv("OPENROUTER_API_KEY", ""))
            base_url = config.base_url or "https://openrouter.ai/api/v1"
        else:  # OpenAI
            key_mapping = {
                "self": "SELF_OPENAI_API_KEY",
                "elm": "ELM_OPENAI_API_KEY",
                None: "OPENAI_API_KEY"
            }
            env_var = key_mapping.get(config.api_key_env, "SELF_OPENAI_API_KEY")
            api_key = SecretStr(os.getenv(env_var, ""))
            base_url = config.base_url

        return api_key, base_url


# ============================================================================
# Token Estimation Strategy
# ============================================================================

class TokenEstimator:
    """Strategy for estimating token usage"""

    @staticmethod
    def estimate(text: str, max_tokens: Optional[int] = None) -> int:
        """
        Estimates total token usage for a request.

        Args:
            text: The prompt text
            max_tokens: Maximum completion tokens (if specified)

        Returns:
            Estimated total tokens with 15% safety buffer
        """
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            prompt_tokens = len(encoding.encode(text))
        except Exception:
            # Fallback to character-based estimation
            prompt_tokens = len(text) // 4

        completion_tokens = max_tokens if max_tokens else 0
        total_tokens = int((prompt_tokens + completion_tokens) * 1.15)

        return total_tokens


# ============================================================================
# Rate Limiting
# ============================================================================

class TokenBucketRateLimiter:
    """Simple token bucket rate limiter for single metric"""

    def __init__(self, calls_per_minute: int = 20, burst_size: int = 8):
        self.rate = calls_per_minute / 60.0
        self.burst_size = burst_size
        self.tokens = float(burst_size)
        self.last_update = time.time()
        self.lock = threading.Lock()

    def acquire_token(self, tokens_needed: float = 1.0) -> None:
        """Wait if necessary to acquire tokens"""
        while True:
            with self.lock:
                now = time.time()
                elapsed = now - self.last_update
                tokens_to_add = elapsed * self.rate
                self.tokens = min(self.burst_size, self.tokens + tokens_to_add)
                self.last_update = now

                if self.tokens >= tokens_needed:
                    self.tokens -= tokens_needed
                    return

                wait_time = (tokens_needed - self.tokens) / self.rate

            if wait_time > 0:
                time.sleep(wait_time)


class DualRateLimiter:
    """Rate limiter managing both request and token limits"""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.request_limiter = TokenBucketRateLimiter(
            config.requests_per_minute,
            config.request_burst_size
        )
        self.token_limiter = TokenBucketRateLimiter(
            config.tokens_per_minute,
            config.token_burst_size or int(config.tokens_per_minute / 60 * 2)
        )
        self.stats = {
            'total_requests': 0,
            'total_tokens': 0,
            'total_wait_time': 0.0,
            'request_limited_count': 0,
            'token_limited_count': 0
        }
        self._lock = threading.Lock()

    def wait_for_permission(self, tokens_needed: int) -> None:
        """Wait for permission for both request and token resources"""
        token_burst_size = self.config.token_burst_size or int(self.config.tokens_per_minute / 60 * 2)
        if tokens_needed > token_burst_size:
            raise ValueError(
                f"Request requires {tokens_needed} tokens but burst size is only {self.config.token_burst_size}"
            )

        start_time = time.time()

        # Acquire request token
        self.request_limiter.acquire_token(1.0)

        # Acquire token tokens
        self.token_limiter.acquire_token(tokens_needed)

        # Update stats
        with self._lock:
            self.stats['total_requests'] += 1
            self.stats['total_tokens'] += tokens_needed
            self.stats['total_wait_time'] += time.time() - start_time

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        with self._lock:
            return self.stats.copy()


# ============================================================================
# Response Processors
# ============================================================================

@runtime_checkable
class ResponseProcessor(Protocol):
    """Protocol for response processors"""
    def process(self, response: Any) -> Any:
        """Process the LLM response"""
        ...


class RawResponseProcessor:
    """Returns response as-is"""
    def process(self, response: Any) -> Any:
        return response


class JSONResponseProcessor:
    """Extracts and parses JSON from response"""

    def process(self, response: Any) -> Dict[str, Any]:
        """Parse JSON from response with multiple fallback strategies"""
        content = response.content if hasattr(response, 'content') else str(response)

        parsed_json = self._parse_json(content)
        return {
            'json': parsed_json,
            'raw_response': response
        }

    def _parse_json(self, content: str) -> Any:
        """Parse JSON from content with multiple strategies"""
        # Try direct JSON parsing
        try:
            return json.loads(content)
        except Exception:
            try:
                return json5.loads(content)
            except Exception:
                pass

        # Try code blocks
        code_block_patterns = [
            r'```json\s*\n(.*?)\n```',
            r'```\s*\n(.*?)\n```',
            r'`(.*?)`'
        ]

        for pattern in code_block_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    return json5.loads(match.strip())
                except Exception:
                    continue

        # Try JSON object patterns
        json_pattern = r'\{(?:[^{}]|{[^{}]*})*\}'
        matches = re.findall(json_pattern, content, re.DOTALL)

        if matches:
            matches.sort(key=len, reverse=True)
            for match in matches:
                try:
                    return json5.loads(match)
                except Exception:
                    continue

        raise ValueError(f"Invalid JSON response. Content:\n{content}")


# ============================================================================
# Retry Strategy
# ============================================================================

class RetryStrategy:
    """Configurable retry strategy for API calls"""

    def __init__(self, max_retries: int = 50, base_delay: float = 2.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retryable_errors = [
            "Internal Server Error", "500", "502", "503", "504",
            "timeout", "connection", "rate limit", "too many requests"
        ]

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if we should retry based on error and attempt number"""
        if attempt >= self.max_retries:
            return False

        # Check for specific timeout exceptions by type
        import asyncio
        if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
            return True

        # Check for JSON parsing errors (these should be retryable)
        if isinstance(error, ValueError) and "Invalid JSON response" in str(error):
            return True

        error_str = str(error).lower()
        return any(err.lower() in error_str for err in self.retryable_errors)

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt"""
        return min(self.base_delay * (2 ** attempt), self.max_delay)


# ============================================================================
# Prompt Logger
# ============================================================================

class PromptLogger:
    """Handles logging of prompts for debugging"""

    def __init__(self, log_dir: str = "prompts_log"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

    def log_prompt(self, prompt_data: Dict[str, Any]) -> Path:
        """Log a prompt and return the log file path"""
        prompt_id = str(uuid.uuid4())
        prompt_file = self.log_dir / f"prompt_{prompt_id}.json"

        prompt_data['timestamp'] = time.time()

        with open(prompt_file, "w", encoding="utf-8") as f:
            json.dump(prompt_data, f, indent=2)

        return prompt_file

    def remove_log(self, log_file: Path) -> None:
        """Remove a log file if it exists"""
        if log_file.exists():
            log_file.unlink()


# ============================================================================
# Main LLM Client
# ============================================================================

class LLMClient:
    """
    Flexible LLM client with pluggable components.
    Uses composition to combine different capabilities.
    """

    def __init__(
        self,
        config: LLMConfig,
        rate_limiter: Optional[DualRateLimiter] = None,
        response_processor: Optional[ResponseProcessor] = None,
        retry_strategy: Optional[RetryStrategy] = None,
        prompt_logger: Optional[PromptLogger] = None,
        token_estimator: Optional[TokenEstimator] = None
    ):
        self.config = config
        self.provider = LLMProviderFactory.create_provider(config)
        self.rate_limiter = rate_limiter
        self.response_processor = response_processor or RawResponseProcessor()
        self.retry_strategy = retry_strategy or RetryStrategy()
        self.prompt_logger = prompt_logger
        self.token_estimator = token_estimator or TokenEstimator()

    def invoke(self, prompt, **kwargs) -> Any:
        """
        Invoke the LLM with the given prompt.

        Args:
            prompt: Either a string or a LangChain prompt template
            **kwargs: Variables to format into the prompt (if using template)

        Returns:
            Processed response based on the configured processor
        """
        # Format prompt if it's a template
        prompt_text = self._format_prompt(prompt, kwargs)

        # Apply rate limiting if configured
        if self.rate_limiter:
            tokens_needed = self.token_estimator.estimate(
                prompt_text,
                self.config.max_tokens
            )
            self.rate_limiter.wait_for_permission(tokens_needed)

        # Log prompt if configured
        log_file = None
        if self.prompt_logger:
            log_data = {
                'prompt': prompt_text,
                'kwargs': kwargs,
                'model': self.config.model_name
            }
            log_file = self.prompt_logger.log_prompt(log_data)

        # Execute with retries (including response processing)
        processed_response = self._execute_with_retry(prompt, kwargs, prompt_text)

        # Clean up log on success
        if self.prompt_logger and log_file:
            self.prompt_logger.remove_log(log_file)

        return processed_response

    def _format_prompt(self, prompt, kwargs: Dict[str, Any]) -> str:
        """Format prompt into text"""
        if isinstance(prompt, str):
            return prompt

        # Handle LangChain prompt templates
        try:
            if hasattr(prompt, 'format'):
                return prompt.format(**kwargs)
            elif hasattr(prompt, '__or__'):
                # It's a chain, we need to estimate from kwargs
                return ' '.join(str(v) for v in kwargs.values())
        except (AttributeError, KeyError, ValueError):
            pass

        # Fallback
        return str(prompt) + ' ' + ' '.join(str(v) for v in kwargs.values())

    def _execute_with_retry(self, prompt, kwargs: Dict[str, Any], prompt_text: str) -> Any:
        """Execute the LLM call with retry logic including response processing"""
        for attempt in range(self.retry_strategy.max_retries):
            try:
                # Determine how to invoke based on prompt type
                if isinstance(prompt, str):
                    # Direct string prompt
                    raw_response = self.provider.invoke(prompt_text)
                elif hasattr(prompt, 'format'):
                    # LangChain prompt template - format then invoke
                    formatted_prompt = prompt.format(**kwargs)
                    raw_response = self.provider.invoke(formatted_prompt)
                else:
                    # Fallback: try to use prompt directly
                    raw_response = self.provider.invoke(str(prompt))

                # Process response (this can raise JSON parsing errors)
                return self.response_processor.process(raw_response)

            except Exception as e:
                if not self.retry_strategy.should_retry(e, attempt + 1):
                    raise e

                if attempt < self.retry_strategy.max_retries - 1:
                    delay = self.retry_strategy.get_delay(attempt)
                    print(f"Error (attempt {attempt + 1}/{self.retry_strategy.max_retries}): {str(e)}")
                    time.sleep(delay)
                else:
                    raise e

    def get_stats(self) -> Optional[Dict[str, Any]]:
        """Get rate limiter stats if available"""
        if self.rate_limiter:
            return self.rate_limiter.get_stats()
        return None

    def get_model_name(self) -> str:
        """Get the model name"""
        return self.config.model_name

    def get_temperature(self) -> float:
        """Get the temperature setting"""
        return self.config.temperature

    def get_max_tokens(self) -> int:
        """Get the max tokens setting"""
        return self.config.max_tokens

    def get_provider(self) -> str:
        """Get the provider name"""
        return self.config.provider

    def get_key(self) -> Optional[str]:
        """Get the API key environment variable name"""
        return self.config.api_key_env


# ============================================================================
# Convenience Factory Functions
# ============================================================================

def create_simple_llm(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 8000,
    provider: str = "openai",
    api_key_env: Optional[str] = None
) -> LLMClient:
    """Create a simple LLM client without rate limiting or JSON parsing"""
    config = LLMConfig(model_name, temperature, max_tokens, provider, api_key_env)
    return LLMClient(config)


def create_json_llm(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 8000,
    provider: str = "openai",
    api_key_env: Optional[str] = None,
    enable_logging: bool = True
) -> LLMClient:
    """Create an LLM client with JSON response parsing"""
    config = LLMConfig(model_name, temperature, max_tokens, provider, api_key_env)
    prompt_logger = PromptLogger() if enable_logging else None

    return LLMClient(
        config,
        response_processor=JSONResponseProcessor(),
        prompt_logger=prompt_logger
    )


def create_rate_limited_llm(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 8192,
    provider: str = "openai",
    api_key_env: Optional[str] = None,
    rate_limit_config: Optional[RateLimitConfig] = None,
    parse_json: bool = False,
    enable_logging: bool = True
) -> LLMClient:
    """Create an LLM client with rate limiting"""
    config = LLMConfig(model_name, temperature, max_tokens, provider, api_key_env)
    rate_config = rate_limit_config or RateLimitConfig()
    rate_limiter = DualRateLimiter(rate_config)

    response_processor = JSONResponseProcessor() if parse_json else RawResponseProcessor()
    prompt_logger = PromptLogger() if enable_logging else None

    return LLMClient(
        config,
        rate_limiter=rate_limiter,
        response_processor=response_processor,
        prompt_logger=prompt_logger
    )



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

        # print(f"Loaded document parsing results from cache: {cache_path}")
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