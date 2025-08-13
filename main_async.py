import asyncio
import argparse
from pathlib import Path
from typing import Optional

from utils import LLMConfig, RateLimitConfig, load_prompt_set
from async_llm_client import (
    create_async_simple_llm, 
    create_async_json_llm, 
    create_async_rate_limited_llm
)
from factory import MapReducePipelineFactory


async def main_async():
    """Async main entry point"""
    parser = argparse.ArgumentParser(description='Async MapReduce QA Pipeline')
    
    # Dataset and data args
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['financebench', 'finqa', 'hybrid', 'last_year', 'last_year_json'],
                       help='Dataset/pipeline type to run')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to the dataset file')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples to process (None for all)')
    
    # Model configuration
    parser.add_argument('--model-name', type=str, default='gpt-4o-mini',
                       help='Model name to use')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Temperature setting')
    parser.add_argument('--max-tokens', type=int, default=8000,
                       help='Maximum tokens per request')
    parser.add_argument('--provider', type=str, default='openai',
                       choices=['openai', 'openrouter'],
                       help='LLM provider')
    parser.add_argument('--key', type=str, default=None,
                       help='API key selector (self, elm, or None for default)')
    
    # Concurrency settings
    parser.add_argument('--max-concurrent-qa', type=int, default=40,
                       help='Maximum concurrent QA pairs')
    parser.add_argument('--max-concurrent-chunks', type=int, default=40,
                       help='Maximum concurrent chunks in map phase')
    
    # Processing settings
    parser.add_argument('--chunk-size', type=int, default=36000,
                       help='Document chunk size')
    parser.add_argument('--chunk-overlap', type=int, default=1000,
                       help='Chunk overlap size')
    parser.add_argument('--prompt', type=str, default='default',
                       help='Prompt set to use')
    
    # Rate limiting
    parser.add_argument('--requests-per-minute', type=int, default=5000,
                       help='Requests per minute rate limit')
    parser.add_argument('--tokens-per-minute', type=int, default=4000000,
                       help='Tokens per minute rate limit')
    
    # Mode selection
    parser.add_argument('--use-async', action='store_true',
                       help='Use async processing (default: sync)')
    parser.add_argument('--parse-json', action='store_true',
                       help='Parse JSON responses')
    parser.add_argument('--enable-rate-limiting', action='store_true',
                       help='Enable rate limiting')
    parser.add_argument('--enable-logging', action='store_true',
                       help='Enable prompt logging')

    args = parser.parse_args()

    print(f"Starting async MapReduce pipeline for {args.dataset}")
    print(f"Data path: {args.data_path}")
    print(f"Model: {args.model_name} (Provider: {args.provider})")
    print(f"Concurrency: {args.max_concurrent_qa} QA pairs, {args.max_concurrent_chunks} chunks")
    
    # Load prompts
    try:
        prompts_dict = load_prompt_set(args.prompt)
        print(f"Loaded prompt set: {args.prompt}")
    except Exception as e:
        print(f"Error loading prompts: {e}")
        return

    # Initialize configuration
    config = LLMConfig(
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        provider=args.provider,
        api_key_env=args.key
    )

    # Create LLM client
    if args.use_async:
        if args.enable_rate_limiting:
            # Create rate limiter configuration
            rate_config = RateLimitConfig(
                requests_per_minute=args.requests_per_minute,
                tokens_per_minute=args.tokens_per_minute
            )
            
            llm_client = create_async_rate_limited_llm(
                model_name=args.model_name,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                provider=args.provider,
                api_key_env=args.key,
                rate_limit_config=rate_config,
                parse_json=args.parse_json,
                enable_logging=args.enable_logging
            )
            print(f"Created async rate-limited LLM client (RPM: {args.requests_per_minute}, TPM: {args.tokens_per_minute})")
        elif args.parse_json:
            llm_client = create_async_json_llm(
                model_name=args.model_name,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                provider=args.provider,
                api_key_env=args.key,
                enable_logging=args.enable_logging
            )
            print("Created async JSON-parsing LLM client")
        else:
            llm_client = create_async_simple_llm(
                model_name=args.model_name,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                provider=args.provider,
                api_key_env=args.key
            )
            print("Created simple async LLM client")
    else:
        # Create sync client for comparison
        from utils import create_simple_llm, create_json_llm, create_rate_limited_llm
        
        if args.enable_rate_limiting:
            rate_config = RateLimitConfig(
                requests_per_minute=args.requests_per_minute,
                tokens_per_minute=args.tokens_per_minute
            )
            
            llm_client = create_rate_limited_llm(
                model_name=args.model_name,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                provider=args.provider,
                api_key_env=args.key,
                rate_limit_config=rate_config,
                parse_json=args.parse_json,
                enable_logging=args.enable_logging
            )
            print(f"Created sync rate-limited LLM client (RPM: {args.requests_per_minute}, TPM: {args.tokens_per_minute})")
        elif args.parse_json:
            llm_client = create_json_llm(
                model_name=args.model_name,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                provider=args.provider,
                api_key_env=args.key,
                enable_logging=args.enable_logging
            )
            print("Created sync JSON-parsing LLM client")
        else:
            llm_client = create_simple_llm(
                model_name=args.model_name,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                provider=args.provider,
                api_key_env=args.key
            )
            print("Created simple sync LLM client")

    # Create pipeline
    try:
        pipeline = MapReducePipelineFactory.create_pipeline(
            pipeline_type=args.dataset,
            llm=llm_client,
            prompts_dict=prompts_dict,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_concurrent_qa=args.max_concurrent_qa,
            max_concurrent_chunks=args.max_concurrent_chunks,
            # Add judge_llm as the same as main LLM for simplicity
            judge_llm=llm_client
        )
        print(f"Created {args.dataset} pipeline")
    except Exception as e:
        print(f"Error creating pipeline: {e}")
        return

    # Run processing
    try:
        if args.use_async:
            # Check if pipeline supports async processing
            if hasattr(pipeline, 'process_dataset_async'):
                print("Running async processing...")
                results = await pipeline.process_dataset_async(
                    data_path=args.data_path,
                    model_name=args.model_name,
                    num_samples=args.num_samples,
                    judge_llm=llm_client
                )
            else:
                print("Pipeline does not support async processing, falling back to sync...")
                results = pipeline.process_dataset(
                    data_path=args.data_path,
                    model_name=args.model_name,
                    num_samples=args.num_samples,
                    judge_llm=llm_client
                )
        else:
            print("Running sync processing...")
            results = pipeline.process_dataset(
                data_path=args.data_path,
                model_name=args.model_name,
                num_samples=args.num_samples,
                judge_llm=llm_client
            )

        print(f"\nProcessing complete!")
        print(f"Samples processed: {results['num_samples']}")
        print(f"Total time: {results['time_taken']:.1f}s")
        
        # Print accuracy if available
        evaluations = results.get('evaluations', {})
        for judge_name, eval_data in evaluations.items():
            accuracy = eval_data.get('accuracy', 0)
            print(f"Accuracy ({judge_name}): {accuracy:.2%}")
        
        # Print rate limiting stats if available
        if hasattr(llm_client, 'get_stats'):
            if args.use_async:
                stats = await llm_client.get_stats()
            else:
                stats = llm_client.get_stats()
            
            if stats:
                print(f"\nRate limiting stats:")
                print(f"  Total requests: {stats['total_requests']}")
                print(f"  Total tokens: {stats['total_tokens']}")
                print(f"  Total wait time: {stats['total_wait_time']:.2f}s")

        return results

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Entry point that handles both sync and async modes"""
    import sys
    
    if '--use-async' in sys.argv:
        return asyncio.run(main_async())
    else:
        return asyncio.run(main_async())  # Run everything through async for simplicity


if __name__ == "__main__":
    main()