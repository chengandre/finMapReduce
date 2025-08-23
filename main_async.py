import asyncio
import argparse

from utils import load_prompt_set
from async_llm_client import RateLimitConfig
from async_llm_client import create_async_rate_limited_llm
from factory import PipelineFactory


async def main_async():
    """Async main entry point"""
    parser = argparse.ArgumentParser(description='Async MapReduce QA Pipeline')

    # Dataset and data args
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['financebench', 'finqa'],
                       help='Dataset/pipeline type to run')
    parser.add_argument('--approach', type=str, default='mapreduce',
                       choices=['mapreduce', 'truncation'],
                       help='Pipeline approach to use')
    parser.add_argument('--format_type', type=str, default='hybrid',
                        choices=['json', 'hybrid', 'plain_text'])
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to the dataset file (auto-detected if not provided)')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to process (None for all)')
    parser.add_argument('--doc_dir', type=str, default='../edgartools_finqa',
                       help='Directory containing FinQA document markdown files (for FinQA only)')

    # Model configuration
    parser.add_argument('--model_name', type=str, default='gpt-4o-mini',
                       help='Model name to use')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Temperature setting')
    parser.add_argument('--max-tokens', type=int, default=8192,
                       help='Maximum tokens per request')
    parser.add_argument('--provider', type=str, default='openai',
                       choices=['openai', 'openrouter'],
                       help='LLM provider')
    parser.add_argument('--key', type=str, default='self',
                       help='API key selector (self, elm, or None for default)')

    # Processing settings
    parser.add_argument('--chunk-size', type=int, default=32768,
                       help='Document chunk size')
    parser.add_argument('--chunk-overlap', type=int, default=4096,
                       help='Chunk overlap size')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Prompt set to use (auto-detected from format_type if not provided)')
    parser.add_argument('--score_threshold', type=int, default=5,
                       help='Score threshold for filtering (plain_text and hybrid formats)')
    parser.add_argument('--pdf_parser', type=str, default='marker',
                       help='PDF parsing method to use (default: marker)')

    # Truncation-specific settings (only used when approach='truncation')
    parser.add_argument('--strategy', type=str, default='start',
                       choices=['start', 'end', 'smart'],
                       help='Truncation strategy for truncation approach')
    parser.add_argument('--context_window', type=int, default=128000,
                       help='Maximum context window size for truncation approach')
    parser.add_argument('--buffer', type=int, default=2000,
                       help='Safety buffer for response tokens in truncation approach')
    parser.add_argument('--max-document-tokens', type=int, default=None,
                       help='Override for max document tokens in truncation approach')

    # Rate limiting
    parser.add_argument('--requests-per-minute', type=int, default=30000,
                       help='Requests per minute rate limit')
    parser.add_argument('--tokens-per-minute', type=int, default=150000000,
                       help='Tokens per minute rate limit')
    parser.add_argument('--request_burst_size', type=int, default=3000,
                       help='Maximum burst size for requests')
    parser.add_argument('--max_total_requests', type=int, default=1000,
                       help='Maximum total concurrent requests across entire pipeline')

    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed results for each QA pair')
    parser.add_argument('--comment', type=str, default=None,
                       help='Comment to save alongside the configuration')

    args = parser.parse_args()

    # Auto-detect data path if not provided
    if args.data_path is None:
        if args.dataset == 'financebench':
            data_path = 'financebench/financebench_open_source.jsonl'
        else:
            data_path = 'finqa/finqa_test.json'
    else:
        data_path = args.data_path

    # Auto-detect prompt set from format_type if not provided
    if args.prompt is None:
        if args.format_type == 'hybrid':
            prompt_set = 'hybrid'
        elif args.format_type == 'plain_text':
            prompt_set = 'baseline'
        else:
            prompt_set = 'default'
    else:
        prompt_set = args.prompt

    print(f"Starting async {args.approach} pipeline for {args.dataset}")
    print(f"Data path: {data_path}")
    print(f"Approach: {args.approach}")
    print(f"Format type: {args.format_type}")
    print(f"Model: {args.model_name} (Provider: {args.provider})")
    print(f"Prompt set: {prompt_set}")
    print(f"API key: {args.key}")
    if args.approach == 'truncation':
        print(f"Truncation strategy: {args.strategy}")
        print(f"Context window: {args.context_window}")
        print(f"Buffer: {args.buffer}")

    # Load prompts
    try:
        prompts_dict = load_prompt_set(prompt_set)
        print(f"Loaded prompt set: {prompt_set}")
    except Exception as e:
        print(f"Error loading prompts: {e}")
        return

    # Create rate limiter configuration
    rate_config = RateLimitConfig(
        requests_per_minute=args.requests_per_minute,
        tokens_per_minute=args.tokens_per_minute,
        request_burst_size=args.request_burst_size
    )
    judge_rate_config = rate_config

    # Configure LLMs based on approach and format_type
    map_llm = None
    reduce_llm = None
    judge_llm = None
    llm_client = None

    if args.approach == 'mapreduce':
        if args.format_type == 'hybrid':
            # Hybrid: separate map (text) and reduce (JSON) LLMs
            map_llm = create_async_rate_limited_llm(
                model_name=args.model_name,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                provider=args.provider,
                api_key_env=args.key,
                rate_limit_config=rate_config,
                parse_json=False
            )
            reduce_llm = create_async_rate_limited_llm(
                model_name=args.model_name,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                provider=args.provider,
                api_key_env=args.key,
                rate_limit_config=rate_config,
                parse_json=True
            )
            llm_client = reduce_llm  # Primary LLM for base class

        else:
            # Standard MapReduce (json): JSON parsing
            llm_client = create_async_rate_limited_llm(
                model_name=args.model_name,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                provider=args.provider,
                api_key_env=args.key,
                rate_limit_config=rate_config,
                parse_json=args.format_type == 'json'
            )

    else:  # truncation approach
        # Truncation: single LLM with JSON parsing (format_type is ignored for truncation)
        llm_client = create_async_rate_limited_llm(
            model_name=args.model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            provider=args.provider,
            api_key_env=args.key,
            rate_limit_config=rate_config,
            parse_json=False
        )

    # Judge LLM for hybrid: gpt-5-nano
    judge_llm = create_async_rate_limited_llm(
        model_name="gpt-5-nano",
        temperature=1.0,
        max_tokens=8192,
        provider="openai",
        api_key_env=args.key,
        rate_limit_config=judge_rate_config,
        parse_json=True
    )

    print(f"Created LLM configuration for {args.dataset} + {args.approach} + {args.format_type}")
    if args.approach == 'mapreduce':
        if map_llm and reduce_llm:
            print(f"  Map LLM: {args.model_name} (text output)")
            print(f"  Reduce LLM: {args.model_name} (JSON output)")
        else:
            json_mode = 'unknown'
            if llm_client and hasattr(llm_client, 'response_processor'):
                json_mode = llm_client.response_processor.__class__.__name__ == 'JSONResponseProcessor'
            print(f"  Main LLM: {args.model_name} (JSON: {json_mode})")
    else:  # truncation
        print(f"  Truncation LLM: {args.model_name} (JSON: False)")

    judge_model_name = 'unknown'
    if judge_llm and hasattr(judge_llm, 'get_model_name'):
        judge_model_name = judge_llm.get_model_name()
    print(f"  Judge LLM: {judge_model_name}")

    # Create pipeline
    try:
        if args.approach == 'mapreduce':
            # MapReduce pipeline
            pipeline_kwargs = {
                'dataset': args.dataset,
                'format_type': args.format_type,
                'llm': llm_client,
                'prompts_dict': prompts_dict,
                'chunk_size': args.chunk_size,
                'chunk_overlap': args.chunk_overlap,
                'pdf_parser': args.pdf_parser,
                'score_threshold': args.score_threshold,
                'max_total_requests': args.max_total_requests
            }

            # Add dataset-specific arguments
            if args.dataset == 'finqa':
                pipeline_kwargs['doc_dir'] = args.doc_dir

            # Add LLM configuration based on format type
            if args.format_type == 'hybrid' and map_llm and reduce_llm:
                pipeline_kwargs['map_llm'] = map_llm
                pipeline_kwargs['reduce_llm'] = reduce_llm

            pipeline = PipelineFactory.create_pipeline(**pipeline_kwargs)
            print(f"Created {args.dataset} + {args.approach} + {args.format_type} pipeline")

        else:  # truncation approach
            # Truncation pipeline
            pipeline_kwargs = {
                'dataset': args.dataset,
                'strategy': args.strategy,
                'context_window': args.context_window,
                'buffer': args.buffer,
                'llm': llm_client,
                'prompts_dict': prompts_dict,
                'pdf_parser': args.pdf_parser,
                'max_total_requests': args.max_total_requests
            }

            # Add optional max_document_tokens if provided
            if args.max_document_tokens is not None:
                pipeline_kwargs['max_document_tokens'] = args.max_document_tokens

            # Add dataset-specific arguments
            if args.dataset == 'finqa':
                pipeline_kwargs['doc_dir'] = args.doc_dir

            pipeline = PipelineFactory.create_truncation_pipeline(**pipeline_kwargs)
            print(f"Created {args.dataset} + {args.approach} + {args.strategy} pipeline")
    except Exception as e:
        print(f"Error creating pipeline: {e}")
        import traceback
        traceback.print_exc()
        return

    # Run async processing
    try:
        print("Running async processing...")
        results = await pipeline.process_dataset_async(
            data_path=data_path,
            model_name=args.model_name,
            num_samples=args.num_samples,
            judge_llm=judge_llm,
            comment=args.comment
        )

        print(f"\nProcessing complete!")
        print(f"Samples processed: {results['num_samples']}")
        print(f"Total time: {results['time_taken']:.1f}s")

        # Print accuracy if available
        evaluations = results.get('evaluations', {})
        for judge_name, eval_data in evaluations.items():
            accuracy = eval_data.get('accuracy', 0)
            print(f"Accuracy ({judge_name}): {accuracy:.2%}")

        # Print detailed results if verbose
        if args.verbose:
            print("\n===== Detailed Results =====")
            for i, qa_item in enumerate(results["qa_data"]):
                print(f"\nQuestion {i+1}: {qa_item['question']}")
                print(f"Question type: {qa_item.get('question_type', 'N/A')}")
                print(f"Document: {qa_item.get('doc_name', 'N/A')}")
                print(f"LLM Answer: {qa_item['llm_answer']}")
                print(f"Golden Answer: {qa_item['answer']}")
                print(f"Judgment: {qa_item.get('judgment', 'N/A')}")

        # Print rate limiting stats if available
        if llm_client and hasattr(llm_client, 'get_stats'):
            try:
                stats = await llm_client.get_stats()
                if stats:
                    print(f"\nRate limiting stats:")
                    print(f"  Total requests: {stats['total_requests']}")
                    print(f"  Total tokens: {stats['total_tokens']}")
                    print(f"  Total wait time: {stats['total_wait_time']:.2f}s")
            except Exception as e:
                print(f"\nCould not retrieve stats: {e}")

        return results

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Entry point for async processing"""
    return asyncio.run(main_async())


if __name__ == "__main__":
    main()