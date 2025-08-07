from factory import MapReducePipelineFactory
from utils import RateLimitedGPT, load_prompt_set
import argparse
import os
import shutil


def main():
    # Clear prompts_log directory at the beginning
    prompts_log_dir = "prompts_log"
    if os.path.exists(prompts_log_dir):
        shutil.rmtree(prompts_log_dir)
    os.makedirs(prompts_log_dir, exist_ok=True)

    parser = argparse.ArgumentParser(description="Run MapReduce QA on FinanceBench data with LLM evaluation")
    parser.add_argument('--jsonl_path', type=str, default="../financebench/data/financebench_open_source.jsonl",
                      help='Path to the financebench jsonl file')
    parser.add_argument('--model_name', type=str, default="gpt-4o-mini",
                      help='Name of the OpenAI model to use')
    parser.add_argument('--num_samples', type=int, default=None,
                      help='Number of samples to process from the dataset')
    parser.add_argument('--temperature', type=float, default=0.00,
                      help='Temperature parameter for the LLM')
    parser.add_argument('--chunk_size', type=int, default=36000,
                      help='Size of each document chunk')
    parser.add_argument('--chunk_overlap', type=int, default=1000,
                      help='Overlap between chunks')
    parser.add_argument('--verbose', action='store_true',
                      help='Print detailed results for each QA pair')
    parser.add_argument('--max_tokens', type=int, default=8192,
                      help='Maximum number of tokens for the LLM')
    parser.add_argument('--provider', type=str, default="openai",
                      help='Provider of the LLM')
    parser.add_argument('--max_concurrent_qa', type=int, default=40,
                      help='Maximum number of QA pairs to process concurrently')
    parser.add_argument('--key', type=str, default=None,
                      help='API key selector: "self" uses SELF_OPENAI_API_KEY, otherwise uses OPENAI_API_KEY')
    parser.add_argument('--prompt', type=str, default=None,
                      help='Prompt set to use (default, old, last_year, standard, wo_icl)')
    parser.add_argument('--requests_per_minute', type=int, default=30000,
                      help='Maximum requests per minute for rate limiting')
    parser.add_argument('--tokens_per_minute', type=int, default=150000000,
                      help='Maximum tokens per minute for rate limiting')
    parser.add_argument('--request_burst_size', type=int, default=3000,
                      help='Maximum burst size for requests')
    parser.add_argument('--pdf_parser', type=str, default="marker",
                      help='PDF parsing method to use (default: marker)')
    parser.add_argument('--comment', type=str, default=None,
                      help='Comment to save alongside the configuration')

    args = parser.parse_args()

    config = {
            'requests_per_minute': args.requests_per_minute,
            'tokens_per_minute': args.tokens_per_minute,
            'request_burst_size': args.request_burst_size
        }

    judge_config = {
            'requests_per_minute': 20,
            'tokens_per_minute': 4000000,
            'request_burst_size': 4
    }

    # Load prompts once at the beginning
    prompts_dict = load_prompt_set(args.prompt)

    llm = RateLimitedGPT(model_name=args.model_name, temperature=args.temperature, max_tokens=args.max_tokens, provider=args.provider, key=args.key, rate_limit_config=config)
    judge = RateLimitedGPT(model_name="deepseek/deepseek-r1-0528:free",
                           temperature=0.0,
                           max_tokens=8192,
                           provider="openrouter",
                           rate_limit_config=judge_config)

    print(f"\nCONFIGURATION:")
    print(f"  Model name: {args.model_name}")
    print(f"  Number of samples: {args.num_samples if args.num_samples else 'all'}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Chunk size: {args.chunk_size}")
    print(f"  Overlap: {args.chunk_overlap}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Key: {args.key if args.key else 'default'}")
    print(f"  Path: {args.jsonl_path}")
    print(f"  Prompt set: {args.prompt if args.prompt else 'default'}")
    print(f"  Requests per minute: {args.requests_per_minute}")
    print(f"  Tokens per minute: {args.tokens_per_minute}")
    print(f"  Request burst size: {args.request_burst_size}")
    print(f"  PDF parser: {args.pdf_parser}")
    print(f"  Comment: {args.comment if args.comment else 'None'}")

    # Create FinanceBench pipeline using factory
    pipeline = MapReducePipelineFactory.create_financebench_pipeline(
        llm=llm,
        prompts_dict=prompts_dict,
        pdf_parser=args.pdf_parser,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_concurrent_qa=args.max_concurrent_qa
    )

    # Process the dataset
    results = pipeline.process_dataset(
        data_path=args.jsonl_path,
        model_name=args.model_name,
        num_samples=args.num_samples,
        judge_llm=judge,
        comment=args.comment
    )

    # Print summary results
    print("\n" + "="*60)
    print("MAPREDUCE EVALUATION RESULTS")
    print("="*60)

    # Get judge model name and evaluation results
    judge_model_name = list(results["evaluations"].keys())[0]
    eval_summary = results["evaluations"][judge_model_name]
    token_summary = results["token_usage_summary"]

    # Basic results
    print(f"Judge model: {judge_model_name}")
    print(f"Total samples: {eval_summary['total']}")
    print(f"Overall accuracy: {eval_summary['accuracy']:.2%}")
    print(f"Time taken: {results['time_taken']:.2f} seconds")

    # Token usage summary
    print(f"\nTOKEN USAGE SUMMARY:")
    print(f"  Total input tokens: {token_summary['total_input_tokens']:,}")
    print(f"  Total output tokens: {token_summary['total_output_tokens']:,}")
    print(f"  Total tokens: {token_summary['total_tokens']:,}")
    print(f"  Avg input per question: {token_summary['avg_input_tokens_per_question']:.0f}")
    print(f"  Avg output per question: {token_summary['avg_output_tokens_per_question']:.0f}")
    print(f"  Token efficiency ratio: {token_summary['token_efficiency_ratio']:.3f}")

    # Judgment distribution
    print(f"\nJUDGMENT DISTRIBUTION:")
    judgments = eval_summary['judgment_distribution']
    percentages = eval_summary['judgment_percentages']
    print(f"  Correct: {judgments['correct']} ({percentages['correct']:.1%})")
    print(f"  Coherent: {judgments['coherent']} ({percentages['coherent']:.1%})")
    print(f"  Deviated: {judgments['deviated']} ({percentages['deviated']:.1%})")
    print(f"  Incorrect: {judgments['incorrect']} ({percentages['incorrect']:.1%})")
    print(f"  No answer: {judgments['no_answer']} ({percentages['no_answer']:.1%})")

    # Accuracy by question type
    print(f"\nACCURACY BY QUESTION TYPE:")
    accuracy_by_type = eval_summary['accuracy_by_question_type']
    for q_type, stats in accuracy_by_type.items():
        print(f"  {q_type}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")

    # MapReduce-specific metrics
    print(f"\nMAPREDUCE CONFIG:")
    config = results["configuration"]
    print(f"  Chunk size: {config['chunk_size']}")
    print(f"  Chunk overlap: {config['chunk_overlap']}")
    print(f"  Max concurrent QA: {config['max_concurrent_qa']}")
    print(f"  PDF parser: {config.get('pdf_parser', 'unknown')}")
    print(f"  Approach: {config.get('approach', 'MapReduce')}")

    # Print detailed results for each QA pair if verbose flag is set
    if args.verbose:
        print("\n===== Detailed Results =====")
        for i, qa_item in enumerate(results["qa_data"]):
            print(f"\nQuestion {i+1}: {qa_item['question']}")
            print(f"Question type: {qa_item.get('question_type', 'N/A')}")
            print(f"Document: {qa_item['doc_name']}")
            print(f"LLM Answer: {qa_item['llm_answer']}")
            print(f"Golden Answer: {qa_item['answer']}")
            print(f"Judgment: {qa_item.get('judgment', 'N/A')}")
            print(f"Reasoning: {qa_item.get('reasoning', 'N/A')}")

    print(f"\nResults saved to file")


if __name__ == "__main__":
    main()