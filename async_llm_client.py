import asyncio
import time
import os
import re
import json
import json5
import tiktoken
import uuid
from typing import Any, Dict, Optional, Protocol, runtime_checkable
from dataclasses import dataclass
from pathlib import Path
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


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
            api_key_value = os.getenv(env_var, "")
            if not api_key_value:
                available_keys = [k for k in os.environ.keys() if 'OPENAI' in k]
                raise ValueError(f"API key not found in environment variable {env_var}. "
                               f"Available OPENAI env vars: {available_keys}. "
                               f"Please set {env_var} or check your environment setup.")
            api_key = SecretStr(api_key_value)
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
# Async Rate Limiting
# ============================================================================

class AsyncTokenBucketRateLimiter:
    """Simple async token bucket rate limiter using loop time"""

    def __init__(self, calls_per_minute: int = 20, burst_size: int = 8):
        self.rate = calls_per_minute / 60.0
        self.burst_size = burst_size
        self.tokens = float(burst_size)
        self.last_update = None  # Will use loop.time()
        self.lock = asyncio.Lock()

    async def acquire_token(self, tokens_needed: float = 1.0) -> None:
        """Acquire tokens with async waiting"""
        loop = asyncio.get_running_loop()

        while True:
            wait_time = 0

            async with self.lock:
                now = loop.time()

                # Initialize on first call
                if self.last_update is None:
                    self.last_update = now

                # Refill tokens based on elapsed time
                elapsed = now - self.last_update
                tokens_to_add = elapsed * self.rate
                self.tokens = min(self.burst_size, self.tokens + tokens_to_add)
                self.last_update = now

                if self.tokens >= tokens_needed:
                    self.tokens -= tokens_needed
                    return

                # Calculate wait time
                wait_time = (tokens_needed - self.tokens) / self.rate

            # Wait outside the lock
            if wait_time > 0:
                await asyncio.sleep(wait_time)


class AsyncDualRateLimiter:
    """Async version managing both request and token limits"""

    def __init__(self, config):
        self.config = config
        self.request_limiter = AsyncTokenBucketRateLimiter(
            config.requests_per_minute,
            config.request_burst_size
        )
        self.token_limiter = AsyncTokenBucketRateLimiter(
            config.tokens_per_minute,
            config.token_burst_size or int(config.tokens_per_minute / 60 * 2)
        )
        self.stats = {
            'total_requests': 0,
            'total_tokens': 0,
            'total_wait_time': 0.0
        }
        self._lock = asyncio.Lock()

    async def wait_for_permission(self, tokens_needed: int) -> None:
        """Wait for both request and token resources"""
        loop = asyncio.get_running_loop()
        start_time = loop.time()

        # Acquire request token
        await self.request_limiter.acquire_token(1.0)

        # Acquire token tokens
        await self.token_limiter.acquire_token(float(tokens_needed))

        # Update stats
        async with self._lock:
            self.stats['total_requests'] += 1
            self.stats['total_tokens'] += tokens_needed
            self.stats['total_wait_time'] += loop.time() - start_time

    async def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        async with self._lock:
            return self.stats.copy()


# ============================================================================
# Async LLM Client
# ============================================================================

class AsyncLLMClient:
    """
    Async wrapper for LLM calls, prioritizing LangChain's ainvoke
    Falls back to httpx only if necessary
    """

    def __init__(
        self,
        config: LLMConfig,
        rate_limiter: Optional[AsyncDualRateLimiter] = None,
        response_processor: Optional[ResponseProcessor] = None,
        retry_strategy: Optional[RetryStrategy] = None,
        token_estimator: Optional[TokenEstimator] = None,
        request_timeout: float = 600.0
    ):
        self.config = config
        self.provider = LLMProviderFactory.create_provider(config)
        self.rate_limiter = rate_limiter
        self.response_processor = response_processor or RawResponseProcessor()
        self.retry_strategy = retry_strategy or RetryStrategy()
        self.token_estimator = token_estimator or TokenEstimator()
        self.request_timeout = request_timeout

    async def invoke(self, prompt, **kwargs) -> Any:
        """
        Async invoke that maintains compatibility with existing response structure
        """
        # Format prompt
        prompt_text = self._format_prompt(prompt, kwargs)

        # Apply rate limiting if configured
        if self.rate_limiter:
            tokens_needed = self.token_estimator.estimate(
                prompt_text,
                self.config.max_tokens
            )
            await self.rate_limiter.wait_for_permission(tokens_needed)

        # Execute with retries
        response = await self._execute_with_retry(prompt, kwargs, prompt_text)

        # Process and return response (keeping sync processor for simplicity)
        return self.response_processor.process(response)

    async def _execute_with_retry(self, prompt, kwargs: Dict[str, Any], prompt_text: str) -> Any:
        """Execute with async retry logic"""
        for attempt in range(self.retry_strategy.max_retries):
            try:
                # Use LangChain's async support
                if isinstance(prompt, str):
                    response = await asyncio.wait_for(
                        self.provider.ainvoke(prompt_text),
                        timeout=self.request_timeout
                    )
                elif hasattr(prompt, 'format'):
                    formatted_prompt = prompt.format(**kwargs)
                    response = await asyncio.wait_for(
                        self.provider.ainvoke(formatted_prompt),
                        timeout=self.request_timeout
                    )
                else:
                    response = await asyncio.wait_for(
                        self.provider.ainvoke(str(prompt)),
                        timeout=self.request_timeout
                    )

                return response

            except Exception as e:
                if not self.retry_strategy.should_retry(e, attempt + 1):
                    raise e

                if attempt < self.retry_strategy.max_retries - 1:
                    delay = self.retry_strategy.get_delay(attempt)
                    print(f"Error (attempt {attempt + 1}/{self.retry_strategy.max_retries}): {str(e)}")
                    await asyncio.sleep(delay)  # Async sleep
                else:
                    raise e


    def _format_prompt(self, prompt, kwargs: Dict[str, Any]) -> str:
        """Format prompt into text - reuse logic from sync client"""
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

    # Maintain compatibility methods
    def get_model_name(self) -> str:
        return self.config.model_name

    def get_temperature(self) -> float:
        return self.config.temperature

    def get_max_tokens(self) -> int:
        return self.config.max_tokens

    def get_provider(self) -> str:
        return self.config.provider

    def get_key(self) -> Optional[str]:
        return self.config.api_key_env

    async def get_stats(self) -> Optional[Dict[str, Any]]:
        """Get rate limiter stats if available"""
        if self.rate_limiter:
            return await self.rate_limiter.get_stats()
        return None


# ============================================================================
# Convenience Factory Functions
# ============================================================================

def create_async_simple_llm(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 8000,
    provider: str = "openai",
    api_key_env: Optional[str] = None
) -> AsyncLLMClient:
    """Create a simple async LLM client without rate limiting or JSON parsing"""
    config = LLMConfig(model_name, temperature, max_tokens, provider, api_key_env)
    return AsyncLLMClient(config)


def create_async_json_llm(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 8000,
    provider: str = "openai",
    api_key_env: Optional[str] = None
) -> AsyncLLMClient:
    """Create an async LLM client with JSON response parsing"""
    config = LLMConfig(model_name, temperature, max_tokens, provider, api_key_env)

    return AsyncLLMClient(
        config,
        response_processor=JSONResponseProcessor()
    )


def create_async_rate_limited_llm(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 8000,
    provider: str = "openai",
    api_key_env: Optional[str] = None,
    rate_limit_config = None,
    parse_json: bool = False
) -> AsyncLLMClient:
    """Create an async LLM client with rate limiting"""
    config = LLMConfig(model_name, temperature, max_tokens, provider, api_key_env)
    rate_config = rate_limit_config or RateLimitConfig()
    rate_limiter = AsyncDualRateLimiter(rate_config)

    response_processor = JSONResponseProcessor() if parse_json else RawResponseProcessor()

    return AsyncLLMClient(
        config,
        rate_limiter=rate_limiter,
        response_processor=response_processor
    )