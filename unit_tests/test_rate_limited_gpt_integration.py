import time
import threading
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from langchain.prompts import PromptTemplate

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import RateLimitedGPT


class TestRateLimitedGPTIntegration:
    """Integration tests for RateLimitedGPT class"""

    def test_initialization_with_default_config(self):
        """Test that RateLimitedGPT initializes with default rate limiting config"""
        with patch('utils.ChatOpenAI'):
            gpt = RateLimitedGPT(
                model_name="gpt-4o-mini",
                temperature=0.7,
                max_tokens=1000
            )

            # Check that rate limiter was initialized with defaults
            assert gpt.rate_limiter.requests_per_minute == 5000
            assert gpt.rate_limiter.tokens_per_minute == 4000000
            assert gpt.rate_limiter.request_burst_size == 500

    def test_initialization_with_custom_config(self):
        """Test that RateLimitedGPT initializes with custom rate limiting config"""
        custom_config = {
            'requests_per_minute': 120,
            'tokens_per_minute': 150000,
            'request_burst_size': 20
        }

        with patch('utils.ChatOpenAI'):
            gpt = RateLimitedGPT(
                model_name="gpt-4",
                temperature=0.5,
                max_tokens=2000,
                rate_limit_config=custom_config
            )

            # Check that rate limiter was initialized with custom config
            assert gpt.rate_limiter.requests_per_minute == 120
            assert gpt.rate_limiter.tokens_per_minute == 150000
            assert gpt.rate_limiter.request_burst_size == 20

    def test_rate_limiting_during_api_calls(self):
        """Test that rate limiting is enforced during API calls - simplified version"""

        # Create RateLimitedGPT with very restrictive limits
        config = {
            'requests_per_minute': 60,  # 1 per second
            'tokens_per_minute': 60000,  # High enough to not be limiting
            'request_burst_size': 2
        }

        # Mock the parent GPT class __call__ method to avoid LLM calls
        with patch.object(RateLimitedGPT.__bases__[0], '__call__') as mock_parent_call:
            mock_parent_call.return_value = {
                'json': {"summary": "test response", "analysis": "completed"},
                'raw_response': 'mock_response'
            }

            gpt = RateLimitedGPT(
                model_name="gpt-4",
                temperature=0,
                max_tokens=100,
                rate_limit_config=config
            )

            # Create a simple prompt
            prompt = PromptTemplate.from_template("Test prompt: {text}")

            # First two calls should be fast (burst)
            start_time = time.time()
            gpt(prompt, text="first")
            gpt(prompt, text="second")
            burst_time = time.time() - start_time

            assert burst_time < 0.5  # Should be very fast

            # Third call should be rate limited
            start_time = time.time()
            gpt(prompt, text="third")
            limited_time = time.time() - start_time

            assert limited_time > 0.8  # Should wait approximately 1 second

    def test_token_estimation_integration(self):
        """Test that token estimation works properly with actual prompts"""
        # Mock the parent GPT class __call__ method
        with patch.object(RateLimitedGPT.__bases__[0], '__call__') as mock_parent_call:
            mock_parent_call.return_value = {
                'json': {"result": "test"},
                'raw_response': 'mock_response'
            }

            gpt = RateLimitedGPT(model_name="gpt-4", temperature=0, max_tokens=100)

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
            gpt(short_prompt, name="Alice")
            gpt(long_prompt, question="What is the meaning of life?")

            # Verify that longer prompt resulted in higher token estimate
            assert len(token_estimates) == 2
            assert token_estimates[1] > token_estimates[0]

    def test_concurrent_calls_with_rate_limiting(self):
        """Test that concurrent calls are properly rate limited"""
        # Mock the parent GPT class __call__ method
        with patch.object(RateLimitedGPT.__bases__[0], '__call__') as mock_parent_call:
            mock_parent_call.return_value = {
                'json': {"result": "test"},
                'raw_response': 'mock_response'
            }

            # Create RateLimitedGPT with restrictive limits
            config = {
                'requests_per_minute': 120,  # 2 per second
                'tokens_per_minute': 120000,  # High enough
                'request_burst_size': 3
            }

            gpt = RateLimitedGPT(
                model_name="gpt-4",
                temperature=0,
                max_tokens=100,
                rate_limit_config=config
            )

            prompt = PromptTemplate.from_template("Test: {text}")
            results = []

            def make_call(text):
                start = time.time()
                gpt(prompt, text=text)
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
        # Mock the parent GPT class __call__ method
        with patch.object(RateLimitedGPT.__bases__[0], '__call__') as mock_parent_call:
            mock_parent_call.return_value = {
                'json': {"result": "test"},
                'raw_response': 'mock_response'
            }

            # Use high limits to avoid waiting
            config = {
                'requests_per_minute': 60000,
                'tokens_per_minute': 6000000,
                'request_burst_size': 1000
            }

            gpt = RateLimitedGPT(
                model_name="gpt-4",
                temperature=0,
                max_tokens=100,
                rate_limit_config=config
            )
            prompt = PromptTemplate.from_template("Test: {text}")

            # Make several calls
            gpt(prompt, text="first")
            gpt(prompt, text="second")
            gpt(prompt, text="third")

            # Check stats
            stats = gpt.get_rate_limit_stats()

            assert stats['total_requests'] == 3
            assert stats['total_tokens'] > 0
            assert stats['total_wait_time'] >= 0

    def test_prompt_formatting_error_handling(self):
        """Test that prompt formatting errors are handled gracefully"""
        # Mock the parent GPT class __call__ method
        with patch.object(RateLimitedGPT.__bases__[0], '__call__') as mock_parent_call:
            mock_parent_call.return_value = {
                'json': {"result": "test"},
                'raw_response': 'mock_response'
            }

            # Use high limits to avoid waiting
            config = {
                'requests_per_minute': 60000,
                'tokens_per_minute': 6000000,
                'request_burst_size': 1000
            }

            gpt = RateLimitedGPT(
                model_name="gpt-4",
                temperature=0,
                max_tokens=100,
                rate_limit_config=config
            )

            # Create a prompt that expects a parameter
            prompt = PromptTemplate.from_template("Hello {name}, how are you?")

            # Patch the format method to raise an exception using patch.object on the class
            with patch.object(PromptTemplate, 'format', side_effect=KeyError("Missing parameter")):
                # Call should still work (using fallback estimation)
                result = gpt(prompt, name="Alice")  # This will fail format but should still work

                assert result is not None
                assert 'json' in result

    def test_token_limited_scenario(self):
        """Test scenario where token limits are hit before request limits"""
        # Mock the parent GPT class __call__ method
        with patch.object(RateLimitedGPT.__bases__[0], '__call__') as mock_parent_call:
            mock_parent_call.return_value = {
                'json': {"result": "test"},
                'raw_response': 'mock_response'
            }

            # Create config where tokens are more restrictive than requests
            config = {
                'requests_per_minute': 6000,  # Very high
                'tokens_per_minute': 60,  # 1 per second
                'request_burst_size': 100,
                'token_burst_size': 2
            }

            gpt = RateLimitedGPT(
                model_name="gpt-4",
                temperature=0,
                max_tokens=1,  # Small to make token estimation predictable
                rate_limit_config=config
            )

            # Create a prompt that will use up the token budget
            prompt = PromptTemplate.from_template("Test")

            # First call should be fast
            start_time = time.time()
            gpt(prompt)
            first_call_time = time.time() - start_time

            assert first_call_time < 0.1

            # Second call should be rate limited by tokens
            start_time = time.time()
            gpt(prompt)
            second_call_time = time.time() - start_time

            # Should wait because of token limit (not request limit)
            assert second_call_time > 0.5

    def test_inheritance_from_gpt_class(self):
        """Test that RateLimitedGPT properly inherits from GPT class"""
        # Mock the parent GPT class __call__ method
        with patch.object(RateLimitedGPT.__bases__[0], '__call__') as mock_parent_call:
            mock_parent_call.return_value = {
                'json': {"result": "test"},
                'raw_response': 'mock_response'
            }

            gpt = RateLimitedGPT(model_name="gpt-4", temperature=0, max_tokens=500)

            # Check that parent class attributes are set
            assert gpt.model_name == "gpt-4"
            assert gpt.temperature == 0
            assert gpt.max_tokens == 500

            # Check that parent class methods work
            prompt = PromptTemplate.from_template("Test")

            # Mock rate limiter to avoid waiting
            gpt.rate_limiter.wait_for_permission = Mock()

            result = gpt(prompt)

            # Should call parent's JSON parsing logic
            assert 'json' in result
            assert 'raw_response' in result