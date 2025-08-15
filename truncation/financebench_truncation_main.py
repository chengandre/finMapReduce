"""
Main script for running FinanceBench QA using truncation baseline.

This script provides a command-line interface for running the truncation
baseline on FinanceBench data, allowing comparison with MapReduce results.
"""

import sys
import os
import argparse
import shutil

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truncation_factory import TruncationPipelineFactory
from utils import create_rate_limited_llm, load_prompt_set, RateLimitConfig


def main():
    """Main entry point for FinanceBench truncation baseline."""
    
    # Clean up prompts log directory
    prompts_log_dir = "prompts_log"
    if os.path.exists(prompts_log_dir):
        shutil.rmtree(prompts_log_dir)
    os.makedirs(prompts_log_dir, exist_ok=True)

    parser = argparse.ArgumentParser(description="Run Truncation QA on FinanceBench data")
    
    # Data and model arguments
    parser.add_argument('--jsonl_path', type=str, default="../financebench/data/financebench_open_source.jsonl",
                      help='Path to the FinanceBench JSONL file')
    parser.add_argument('--model_name', type=str, default="gpt-4o-mini",
                      help='Name of the LLM model to use')
    parser.add_argument('--num_samples', type=int, default=None,
                      help='Number of samples to process from the dataset')
    
    # LLM configuration
    parser.add_argument('--temperature', type=float, default=0.00,
                      help='Temperature parameter for the LLM')
    parser.add_argument('--max_tokens', type=int, default=8192,
                      help='Maximum number of tokens for LLM response')
    parser.add_argument('--provider', type=str, default="openai",
                      help='Provider of the LLM (openai, openrouter, etc.)')
    parser.add_argument('--key', type=str, default="elm",
                      help='API key selector: "self" uses SELF_OPENAI_API_KEY, otherwise uses OPENAI_API_KEY')
    
    # Truncation-specific arguments
    parser.add_argument('--truncation_strategy', type=str, default="start",
                      choices=TruncationPipelineFactory.get_available_strategies(),
                      help='Truncation strategy to use')
    parser.add_argument('--context_window', type=int, default=120000,
                      help='Model context window size')
    parser.add_argument('--truncation_buffer', type=int, default=2000,
                      help='Safety buffer for response tokens')
    parser.add_argument('--max_document_tokens', type=int, default=None,
                      help='Maximum tokens from document (None = auto-calculate)')
    
    # Document processing
    parser.add_argument('--pdf_parser', type=str, default="marker",
                      help='PDF parsing method to use')
    
    # Execution configuration
    parser.add_argument('--max_concurrent_qa', type=int, default=150,
                      help='Maximum number of QA pairs to process concurrently')
    
    # Rate limiting
    parser.add_argument('--requests_per_minute', type=int, default=30000,
                      help='Maximum requests per minute for rate limiting')
    parser.add_argument('--tokens_per_minute', type=int, default=150000000,
                      help='Maximum tokens per minute for rate limiting')
    parser.add_argument('--request_burst_size', type=int, default=3000,
                      help='Maximum burst size for requests')
    
    # Prompt configuration
    parser.add_argument('--prompt', type=str, default='direct',
                      help='Prompt set to use (default, old, last_year, etc.)')
    
    # Output and debugging
    parser.add_argument('--verbose', action='store_true',
                      help='Print detailed results for each QA pair')
    parser.add_argument('--comment', type=str, default=None,
                      help='Comment to save alongside the configuration')

    args = parser.parse_args()

    # Create rate limiting configurations
    rate_config = RateLimitConfig(
        requests_per_minute=args.requests_per_minute,
        tokens_per_minute=args.tokens_per_minute,
        request_burst_size=args.request_burst_size
    )

    judge_rate_config = RateLimitConfig(
        requests_per_minute=20,
        tokens_per_minute=4000000,
        request_burst_size=4
    )

    # Load prompts
    prompts_dict = load_prompt_set(args.prompt)

    # Create LLM instances
    llm = create_rate_limited_llm(
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        provider=args.provider,
        api_key_env=args.key,
        rate_limit_config=rate_config,
        parse_json=False
    )

    judge = create_rate_limited_llm(
        model_name="gpt-5-nano",
        temperature=1.0,
        max_tokens=8192,
        provider="openai",
        api_key_env="elm",
        rate_limit_config=judge_rate_config,
        parse_json=True
    )

    # Print configuration
    print(f"\nCONFIGURATION:")
    print(f"  Dataset: financebench")
    print(f"  Approach: Truncation")
    print(f"  Truncation strategy: {args.truncation_strategy}")
    print(f"  Context window: {args.context_window:,} tokens")
    print(f"  Truncation buffer: {args.truncation_buffer:,} tokens")
    print(f"  Max document tokens: {args.max_document_tokens if args.max_document_tokens else 'auto'}")
    print(f"  Model name: {args.model_name}")
    print(f"  Number of samples: {args.num_samples if args.num_samples else 'all'}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Provider: {args.provider}")
    print(f"  Key: {args.key if args.key else 'default'}")
    print(f"  Path: {args.jsonl_path}")
    print(f"  Prompt set: {args.prompt if args.prompt else 'default'}")
    print(f"  PDF parser: {args.pdf_parser}")
    print(f"  Max concurrent QA: {args.max_concurrent_qa}")
    print(f"  Comment: {args.comment if args.comment else 'None'}")

    # Validate configuration
    validation = TruncationPipelineFactory.validate_configuration(
        dataset='financebench',
        truncation_strategy=args.truncation_strategy,
        context_window=args.context_window,
        llm=llm,
        prompts_dict=prompts_dict,
        pdf_parser=args.pdf_parser,
        max_document_tokens=args.max_document_tokens
    )
    
    if not validation['valid']:
        print("\nConfiguration Errors:")
        for error in validation['errors']:
            print(f"  - {error}")
        return 1
    
    if validation['warnings']:
        print("\nConfiguration Warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")

    # Create FinanceBench truncation pipeline
    pipeline = TruncationPipelineFactory.create_pipeline(
        dataset='financebench',
        truncation_strategy=args.truncation_strategy,
        context_window=args.context_window,
        truncation_buffer=args.truncation_buffer,
        max_document_tokens=args.max_document_tokens,
        max_concurrent_qa=args.max_concurrent_qa,
        llm=llm,
        prompts_dict=prompts_dict,
        pdf_parser=args.pdf_parser
    )

    # Process dataset
    try:
        results = pipeline.process_dataset(
            data_path=args.jsonl_path,
            model_name=args.model_name,
            num_samples=args.num_samples,
            judge_llm=judge,
            comment=args.comment
        )
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return 1

    # Print summary results
    print("\n" + "="*60)
    print("TRUNCATION BASELINE EVALUATION RESULTS")
    print("="*60)

    judge_model_name = list(results["evaluations"].keys())[0]
    eval_summary = results["evaluations"][judge_model_name]
    token_summary = results["token_usage_summary"]
    truncation_summary = results["truncation_summary"]

    print(f"Dataset: {results['configuration']['dataset']}")
    print(f"Approach: {results['configuration']['approach']}")
    print(f"Truncation strategy: {results['configuration']['truncation_strategy']}")
    print(f"Context window: {results['configuration']['context_window']:,} tokens")
    print(f"Judge model: {judge_model_name}")
    print(f"Total samples: {eval_summary['total']}")
    print(f"Overall accuracy: {eval_summary['accuracy']:.2%}")
    print(f"Time taken: {results['time_taken']:.2f} seconds")

    print(f"\nTRUNCATION STATISTICS:")
    print(f"  Truncations applied: {truncation_summary['truncations_applied']}/{results['num_samples']} ({truncation_summary['truncation_rate']:.1%})")
    print(f"  Average original tokens: {truncation_summary['original_tokens']['avg']:.0f}")
    print(f"  Average truncated tokens: {truncation_summary['truncated_tokens']['avg']:.0f}")
    print(f"  Average retention rate: {truncation_summary['retention_rates']['avg']:.1%}")

    print(f"\nTOKEN USAGE SUMMARY:")
    print(f"  Total input tokens: {token_summary['total_input_tokens']:,}")
    print(f"  Total output tokens: {token_summary['total_output_tokens']:,}")
    print(f"  Total tokens: {token_summary['total_tokens']:,}")
    print(f"  Avg input per question: {token_summary['avg_input_tokens_per_question']:.0f}")
    print(f"  Avg output per question: {token_summary['avg_output_tokens_per_question']:.0f}")

    print(f"\nJUDGMENT DISTRIBUTION:")
    judgments = eval_summary['judgment_distribution']
    percentages = eval_summary['judgment_percentages']
    print(f"  Correct: {judgments['correct']} ({percentages['correct']:.1%})")
    print(f"  Coherent: {judgments['coherent']} ({percentages['coherent']:.1%})")
    print(f"  Deviated: {judgments['deviated']} ({percentages['deviated']:.1%})")
    print(f"  Incorrect: {judgments['incorrect']} ({percentages['incorrect']:.1%})")
    print(f"  No answer: {judgments['no_answer']} ({percentages['no_answer']:.1%})")

    print(f"\nACCURACY BY QUESTION TYPE:")
    accuracy_by_type = eval_summary['accuracy_by_question_type']
    for q_type, stats in accuracy_by_type.items():
        print(f"  {q_type}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")

    if args.verbose:
        print("\n===== Detailed Results =====")
        for i, qa_item in enumerate(results["qa_data"]):
            print(f"\nQuestion {i+1}: {qa_item['question']}")
            print(f"Question type: {qa_item.get('question_type', 'N/A')}")
            print(f"Document: {qa_item.get('doc_name', 'N/A')}")
            print(f"LLM Answer: {qa_item['llm_answer']}")
            print(f"Golden Answer: {qa_item.get('answer', 'N/A')}")
            print(f"Judgment: {qa_item.get('judgment', 'N/A')}")
            
            # Show truncation stats for this question
            token_stats = qa_item.get('token_stats', {})
            truncation_stats = token_stats.get('truncation_stats', {})
            if truncation_stats:
                print(f"Truncation: {truncation_stats.get('truncation_applied', False)} "
                      f"({truncation_stats.get('original_tokens', 0)} → "
                      f"{truncation_stats.get('truncated_tokens', 0)} tokens, "
                      f"{truncation_stats.get('retention_rate', 0.0):.1%} retained)")

    print(f"\nResults saved to file")
    return 0


if __name__ == "__main__":
    exit(main())