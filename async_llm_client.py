import asyncio
import time
from functools import partial
from typing import Any, Dict, Optional, Protocol, runtime_checkable
from types import SimpleNamespace
from dataclasses import dataclass

from utils import (
    LLMConfig, LLMProviderFactory, TokenEstimator, 
    ResponseProcessor, RawResponseProcessor, JSONResponseProcessor,
    RetryStrategy, PromptLogger
)


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
        prompt_logger: Optional[PromptLogger] = None,
        token_estimator: Optional[TokenEstimator] = None,
        request_timeout: float = 60.0
    ):
        self.config = config
        self.provider = LLMProviderFactory.create_provider(config)
        self.rate_limiter = rate_limiter
        self.response_processor = response_processor or RawResponseProcessor()
        self.retry_strategy = retry_strategy or RetryStrategy()
        self.prompt_logger = prompt_logger
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

        # Log prompt if configured
        log_file = None
        if self.prompt_logger:
            log_data = {
                'prompt': prompt_text,
                'kwargs': kwargs,
                'model': self.config.model_name
            }
            # Run log creation in executor to avoid blocking
            loop = asyncio.get_running_loop()
            log_file = await loop.run_in_executor(
                None, self.prompt_logger.log_prompt, log_data
            )

        # Execute with retries
        response = await self._execute_with_retry(prompt, kwargs, prompt_text)

        # Clean up log on success
        if self.prompt_logger and log_file:
            # Run cleanup in executor to avoid blocking
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.prompt_logger.remove_log, log_file)

        # Process and return response (keeping sync processor for simplicity)
        return self.response_processor.process(response)

    async def _execute_with_retry(self, prompt, kwargs: Dict[str, Any], prompt_text: str) -> Any:
        """Execute with async retry logic"""
        for attempt in range(self.retry_strategy.max_retries):
            try:
                # Check if provider supports ainvoke (LangChain)
                if hasattr(self.provider, 'ainvoke'):
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
                else:
                    # Fallback to sync in executor (temporary bridge)
                    loop = asyncio.get_running_loop()
                    response = await loop.run_in_executor(
                        None,
                        partial(self._sync_invoke, prompt, kwargs, prompt_text)
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

    def _sync_invoke(self, prompt, kwargs, prompt_text):
        """Helper for sync fallback"""
        if isinstance(prompt, str):
            return self.provider.invoke(prompt_text)
        elif hasattr(prompt, 'format'):
            return self.provider.invoke(prompt.format(**kwargs))
        else:
            return self.provider.invoke(str(prompt))

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
    api_key_env: Optional[str] = None,
    enable_logging: bool = True
) -> AsyncLLMClient:
    """Create an async LLM client with JSON response parsing"""
    config = LLMConfig(model_name, temperature, max_tokens, provider, api_key_env)
    prompt_logger = PromptLogger() if enable_logging else None

    return AsyncLLMClient(
        config,
        response_processor=JSONResponseProcessor(),
        prompt_logger=prompt_logger
    )


def create_async_rate_limited_llm(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 8000,
    provider: str = "openai",
    api_key_env: Optional[str] = None,
    rate_limit_config = None,
    parse_json: bool = False,
    enable_logging: bool = True
) -> AsyncLLMClient:
    """Create an async LLM client with rate limiting"""
    from utils import RateLimitConfig  # Import here to avoid circular imports
    
    config = LLMConfig(model_name, temperature, max_tokens, provider, api_key_env)
    rate_config = rate_limit_config or RateLimitConfig()
    rate_limiter = AsyncDualRateLimiter(rate_config)

    response_processor = JSONResponseProcessor() if parse_json else RawResponseProcessor()
    prompt_logger = PromptLogger() if enable_logging else None

    return AsyncLLMClient(
        config,
        rate_limiter=rate_limiter,
        response_processor=response_processor,
        prompt_logger=prompt_logger
    )