import time
import threading
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from langchain.prompts import PromptTemplate

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from async_llm_client import create_async_rate_limited_llm, RateLimitConfig, AsyncLLMClient


class TestLLMClientIntegration:
    """Integration tests for LLMClient class with rate limiting"""

    def test_initialization_with_default_config(self):
        """Test that LLMClient initializes with default rate limiting config"""
        with patch('utils.ChatOpenAI'):
            gpt = create_rate_limited_llm(
                model_name="gpt-4o-mini",
                temperature=0.7,
                max_tokens=1000,
                parse_json=True
            )

            # Check that rate limiter was initialized with defaults
            assert gpt.rate_limiter.config.requests_per_minute == 5000
            assert gpt.rate_limiter.config.tokens_per_minute == 4000000
            assert gpt.rate_limiter.config.request_burst_size == 500

    def test_initialization_with_custom_config(self):
        """Test that LLMClient initializes with custom rate limiting config"""
        custom_config = RateLimitConfig(
            requests_per_minute=120,
            tokens_per_minute=150000,
            request_burst_size=20
        )

        with patch('utils.ChatOpenAI'):
            gpt = create_rate_limited_llm(
                model_name="gpt-4",
                temperature=0.5,
                max_tokens=2000,
                rate_limit_config=custom_config,
                parse_json=True
            )

            # Check that rate limiter was initialized with custom config
            assert gpt.rate_limiter.config.requests_per_minute == 120
            assert gpt.rate_limiter.config.tokens_per_minute == 150000
            assert gpt.rate_limiter.config.request_burst_size == 20

    def test_rate_limiting_during_api_calls(self):
        """Test that rate limiting is enforced during API calls - simplified version"""

        # Create LLMClient with very restrictive limits
        config = RateLimitConfig(
            requests_per_minute=60,  # 1 per second
            tokens_per_minute=60000,  # High enough to not be limiting
            request_burst_size=2
        )

        # Mock the ChatOpenAI provider
        with patch('utils.ChatOpenAI') as mock_openai:
            mock_openai.return_value.invoke.return_value.content = '{"summary": "test response", "analysis": "completed"}'

            gpt = create_rate_limited_llm(
                model_name="gpt-4",
                temperature=0,
                max_tokens=100,
                rate_limit_config=config,
                parse_json=True
            )

            # Create a simple prompt
            prompt = PromptTemplate.from_template("Test prompt: {text}")

            # First two calls should be fast (burst)
            start_time = time.time()
            gpt.invoke(prompt, text="first")
            gpt.invoke(prompt, text="second")
            burst_time = time.time() - start_time

            assert burst_time < 0.5  # Should be very fast

            # Third call should be rate limited
            start_time = time.time()
            gpt.invoke(prompt, text="third")
            limited_time = time.time() - start_time

            assert limited_time > 0.8  # Should wait approximately 1 second

    def test_token_estimation_integration(self):
        """Test that token estimation works properly with actual prompts"""
        # Mock the ChatOpenAI provider
        with patch('utils.ChatOpenAI') as mock_openai:
            mock_openai.return_value.invoke.return_value.content = '{"result": "test"}'

            gpt = create_rate_limited_llm(model_name="gpt-4", temperature=0, max_tokens=100, parse_json=True)

            # Create prompts of different lengths
            short_prompt = PromptTemplate.from_template("Hi {name}")
            long_prompt = PromptTemplate.from_template(
                "This is a very long prompt with lots of text. " * 20 + "Question: {question}"
            )

            # Mock the wait_for_permission method to capture token estimates
            token_estimates = []
            original_wait = gpt.rate_limiter.wait_for_permission

            def capture_tokens(tokens_needed):
                token_estimates.append(tokens_needed)
                # Don't actually wait for this test

            gpt.rate_limiter.wait_for_permission = capture_tokens

            # Make calls with different prompt lengths
            gpt.invoke(short_prompt, name="Alice")
            gpt.invoke(long_prompt, question="What is the meaning of life?")

            # Verify that longer prompt resulted in higher token estimate
            assert len(token_estimates) == 2
            assert token_estimates[1] > token_estimates[0]

    def test_concurrent_calls_with_rate_limiting(self):
        """Test that concurrent calls are properly rate limited"""
        # Mock the ChatOpenAI provider
        with patch('utils.ChatOpenAI') as mock_openai:
            mock_openai.return_value.invoke.return_value.content = '{"result": "test"}'

            # Create LLMClient with restrictive limits
            config = RateLimitConfig(
                requests_per_minute=120,  # 2 per second
                tokens_per_minute=120000,  # High enough
                request_burst_size=3
            )

            gpt = create_rate_limited_llm(
                model_name="gpt-4",
                temperature=0,
                max_tokens=100,
                rate_limit_config=config,
                parse_json=True
            )

            prompt = PromptTemplate.from_template("Test: {text}")
            results = []

            def make_call(text):
                start = time.time()
                gpt.invoke(prompt, text=text)
                end = time.time()
                results.append(end - start)

            # Launch 6 concurrent threads
            threads = []
            for i in range(6):
                t = threading.Thread(target=make_call, args=(f"call_{i}",))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            # Sort results to analyze timing
            results.sort()

            # First 3 should be fast (burst), others should wait
            assert results[0] < 0.2
            assert results[1] < 0.2
            assert results[2] < 0.2
            assert results[3] > 0.4  # Should wait

    def test_stats_tracking_integration(self):
        """Test that rate limiting stats are tracked correctly"""
        # Mock the ChatOpenAI provider
        with patch('utils.ChatOpenAI') as mock_openai:
            mock_openai.return_value.invoke.return_value.content = '{"result": "test"}'

            # Use high limits to avoid waiting
            config = RateLimitConfig(
                requests_per_minute=60000,
                tokens_per_minute=6000000,
                request_burst_size=1000
            )

            gpt = create_rate_limited_llm(
                model_name="gpt-4",
                temperature=0,
                max_tokens=100,
                rate_limit_config=config,
                parse_json=True
            )
            prompt = PromptTemplate.from_template("Test: {text}")

            # Make several calls
            gpt.invoke(prompt, text="first")
            gpt.invoke(prompt, text="second")
            gpt.invoke(prompt, text="third")

            # Check stats
            stats = gpt.get_stats()

            assert stats['total_requests'] == 3
            assert stats['total_tokens'] > 0
            assert stats['total_wait_time'] >= 0

    def test_prompt_formatting_error_handling(self):
        """Test that prompt formatting errors are handled gracefully"""
        # Mock the ChatOpenAI provider
        with patch('utils.ChatOpenAI') as mock_openai:
            mock_openai.return_value.invoke.return_value.content = '{"result": "test"}'

            # Use high limits to avoid waiting
            config = RateLimitConfig(
                requests_per_minute=60000,
                tokens_per_minute=6000000,
                request_burst_size=1000
            )

            gpt = create_rate_limited_llm(
                model_name="gpt-4",
                temperature=0,
                max_tokens=100,
                rate_limit_config=config,
                parse_json=True
            )

            # Create a prompt that expects a parameter
            prompt = PromptTemplate.from_template("Hello {name}, how are you?")

            # Patch the format method to raise an exception using patch.object on the class
            with patch.object(PromptTemplate, 'format', side_effect=KeyError("Missing parameter")):
                # Call should still work (using fallback estimation)
                result = gpt.invoke(prompt, name="Alice")  # This will fail format but should still work

                assert result is not None
                assert 'json' in result

    def test_token_limited_scenario(self):
        """Test scenario where token limits are hit before request limits"""
        # Mock the ChatOpenAI provider
        with patch('utils.ChatOpenAI') as mock_openai:
            mock_openai.return_value.invoke.return_value.content = '{"result": "test"}'

            # Create config where tokens are more restrictive than requests
            config = RateLimitConfig(
                requests_per_minute=6000,  # Very high
                tokens_per_minute=60,  # 1 per second
                request_burst_size=100,
                token_burst_size=2
            )

            gpt = create_rate_limited_llm(
                model_name="gpt-4",
                temperature=0,
                max_tokens=1,  # Small to make token estimation predictable
                rate_limit_config=config,
                parse_json=True
            )

            # Create a prompt that will use up the token budget
            prompt = PromptTemplate.from_template("Test")

            # First call should be fast
            start_time = time.time()
            gpt.invoke(prompt)
            first_call_time = time.time() - start_time

            assert first_call_time < 0.1

            # Second call should be rate limited by tokens
            start_time = time.time()
            gpt.invoke(prompt)
            second_call_time = time.time() - start_time

            # Should wait because of token limit (not request limit)
            assert second_call_time > 0.5

    def test_llm_client_configuration_access(self):
        """Test that LLMClient provides access to configuration settings"""
        # Mock the ChatOpenAI provider
        with patch('utils.ChatOpenAI'):
            gpt = create_rate_limited_llm(
                model_name="gpt-4",
                temperature=0,
                max_tokens=500,
                parse_json=True
            )

            # Check that configuration is accessible
            assert gpt.get_model_name() == "gpt-4"
            assert gpt.get_temperature() == 0
            assert gpt.get_max_tokens() == 500

            # Check that client methods work
            prompt = PromptTemplate.from_template("Test")

            # Mock rate limiter to avoid waiting
            gpt.rate_limiter.wait_for_permission = Mock()

            # Mock the provider invoke method
            gpt.provider.invoke = Mock(return_value=Mock(content='{"result": "test"}'))

            result = gpt.invoke(prompt)

            # Should return processed JSON response
            assert 'json' in result
            assert 'raw_response' in result