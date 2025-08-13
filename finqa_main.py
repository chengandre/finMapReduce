from factory import MapReducePipelineFactory
from utils import create_rate_limited_llm, load_prompt_set, RateLimitConfig
import argparse
import os
import shutil


def main():
    prompts_log_dir = "prompts_log"
    if os.path.exists(prompts_log_dir):
        shutil.rmtree(prompts_log_dir)
    os.makedirs(prompts_log_dir, exist_ok=True)

    parser = argparse.ArgumentParser(description="Run MapReduce QA on FinQA data with new architecture")
    parser.add_argument('--json_path', type=str, default="../finqa_balanced_subset.json",
                      help='Path to the FinQA json file')
    parser.add_argument('--doc_dir', type=str, default="../edgartools_finqa",
                      help='Directory containing FinQA document markdown files')
    parser.add_argument('--model_name', type=str, default="gpt-4o-mini",
                      help='Name of the OpenAI model to use')
    parser.add_argument('--format_type', type=str, default="json", choices=["json", "plain_text", "hybrid"],
                      help='Output format type')
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
    parser.add_argument('--max_tokens', type=int, default=8000,
                      help='Maximum number of tokens for the LLM')
    parser.add_argument('--provider', type=str, default="openai",
                      help='Provider of the LLM')
    parser.add_argument('--max_concurrent_qa', type=int, default=20,
                      help='Maximum number of QA pairs to process concurrently')
    parser.add_argument('--key', type=str, default=None,
                      help='API key selector: "self" uses SELF_OPENAI_API_KEY, otherwise uses OPENAI_API_KEY')
    parser.add_argument('--prompt', type=str, default=None,
                      help='Prompt set to use (default, old, last_year, hybrid)')
    parser.add_argument('--requests_per_minute', type=int, default=5000,
                      help='Maximum requests per minute for rate limiting')
    parser.add_argument('--tokens_per_minute', type=int, default=4000000,
                      help='Maximum tokens per minute for rate limiting')
    parser.add_argument('--request_burst_size', type=int, default=500,
                      help='Maximum burst size for requests')
    parser.add_argument('--score_threshold', type=int, default=50,
                      help='Score threshold for filtering (plain_text and hybrid formats)')
    parser.add_argument('--comment', type=str, default=None,
                      help='Comment to save alongside the configuration')

    args = parser.parse_args()

    rate_config = RateLimitConfig(
        requests_per_minute=args.requests_per_minute,
        tokens_per_minute=args.tokens_per_minute,
        request_burst_size=args.request_burst_size
    )

    prompts_dict = load_prompt_set(args.prompt)

    llm = create_rate_limited_llm(
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        provider=args.provider,
        api_key_env=args.key,
        rate_limit_config=rate_config,
        parse_json=True
    )

    print(f"\nCONFIGURATION:")
    print(f"  Dataset: finqa")
    print(f"  Format type: {args.format_type}")
    print(f"  Model name: {args.model_name}")
    print(f"  Number of samples: {args.num_samples if args.num_samples else 'all'}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Chunk size: {args.chunk_size}")
    print(f"  Overlap: {args.chunk_overlap}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Key: {args.key if args.key else 'default'}")
    print(f"  JSON path: {args.json_path}")
    print(f"  Doc dir: {args.doc_dir}")
    print(f"  Prompt set: {args.prompt if args.prompt else 'default'}")
    print(f"  Score threshold: {args.score_threshold}")
    print(f"  Comment: {args.comment if args.comment else 'None'}")

    # Create FinQA pipeline using new factory
    pipeline = MapReducePipelineFactory.create_pipeline(
        dataset='finqa',
        format_type=args.format_type,
        llm=llm,
        prompts_dict=prompts_dict,
        doc_dir=args.doc_dir,
        score_threshold=args.score_threshold,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_concurrent_qa=args.max_concurrent_qa
    )

    results = pipeline.process_dataset(
        data_path=args.json_path,
        model_name=args.model_name,
        num_samples=args.num_samples,
        judge_llm=llm,
        comment=args.comment
    )

    # Print summary results
    print("\n" + "="*60)
    print("FINQA MAPREDUCE EVALUATION RESULTS")
    print("="*60)

    judge_model_name = list(results["evaluations"].keys())[0]
    eval_summary = results["evaluations"][judge_model_name]
    token_summary = results["token_usage_summary"]

    print(f"Dataset: {results['configuration']['dataset']}")
    print(f"Format: {args.format_type}")
    print(f"Approach: {results['configuration']['approach']}")
    print(f"Judge model: {judge_model_name}")
    print(f"Total samples: {eval_summary['total']}")
    print(f"Overall accuracy: {eval_summary['accuracy']:.2%}")
    print(f"Time taken: {results['time_taken']:.2f} seconds")

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

    if args.verbose:
        print("\n===== Detailed Results =====")
        for i, qa_item in enumerate(results["qa_data"]):
            print(f"\nQuestion {i+1}: {qa_item['question']}")
            print(f"Document: {qa_item['doc_name']}")
            print(f"LLM Answer: {qa_item['llm_answer']}")
            print(f"Golden Answer: {qa_item['answer']}")
            print(f"Judgment: {qa_item.get('judgment', 'N/A')}")

    print(f"\nResults saved to file")


if __name__ == "__main__":
    main()