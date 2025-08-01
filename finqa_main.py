from mapreduce_qa import process_finqa_qa
from utils import GPT, RateLimitedGPT
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run MapReduce QA on FinQA data with LLM evaluation")
    parser.add_argument('--json_path', type=str, default="../finqa_balanced_subset.json",
                      help='Path to the FinQA json file')
    parser.add_argument('--doc_dir', type=str, default="../edgartools_finqa",
                      help='Directory containing FinQA document markdown files')
    parser.add_argument('--model_name', type=str, default="deepseek/deepseek-r1-0528:free",
                      help='Name of the OpenAI model to use')
    parser.add_argument('--num_samples', type=int, default=None,
                      help='Number of samples to process from the dataset')
    parser.add_argument('--temperature', type=float, default=0.01,
                      help='Temperature parameter for the LLM')
    parser.add_argument('--chunk_size', type=int, default=36000,
                      help='Size of each document chunk')
    parser.add_argument('--chunk_overlap', type=int, default=1000,
                      help='Overlap between chunks')
    parser.add_argument('--verbose', action='store_true',
                      help='Print detailed results for each QA pair')
    parser.add_argument('--max_tokens', type=int, default=8000,
                      help='Maximum number of tokens for the LLM')
    parser.add_argument('--provider', type=str, default="openrouter",
                      help='Provider of the LLM')
    parser.add_argument('--max_concurrent_qa', type=int, default=20,
                      help='Maximum number of QA pairs to process concurrently')
    parser.add_argument('--key', type=str, default=None,
                      help='API key selector: "self" uses SELF_OPENAI_API_KEY, otherwise uses OPENAI_API_KEY')
    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    llm = RateLimitedGPT(model_name=args.model_name, temperature=args.temperature, max_tokens=args.max_tokens, provider=args.provider, key=args.key)

    print(f"Processing {args.num_samples if args.num_samples else 'all'} samples from {args.json_path}")
    results = process_finqa_qa(
        json_path=args.json_path,
        doc_dir=args.doc_dir,
        model_name=args.model_name,
        llm=llm,
        num_samples=args.num_samples,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_concurrent_qa=args.max_concurrent_qa
    )

    # Print summary results
    print("\n" + "="*60)
    print("FINQA MAPREDUCE EVALUATION RESULTS")
    print("="*60)
    
    eval_summary = results["evaluation_summary"]
    token_summary = results["token_usage_summary"]
    
    # Basic results
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
    
    # Accuracy by question type (if available)
    if 'accuracy_by_question_type' in eval_summary:
        print(f"\nACCURACY BY QUESTION TYPE:")
        accuracy_by_type = eval_summary['accuracy_by_question_type']
        for q_type, stats in accuracy_by_type.items():
            print(f"  {q_type}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")
    
    # MapReduce-specific metrics
    print(f"\nMAPREDUCE CONFIG:")
    mapreduce_config = results["mapreduce_config"]
    print(f"  Chunk size: {mapreduce_config['chunk_size']}")
    print(f"  Chunk overlap: {mapreduce_config['chunk_overlap']}")
    print(f"  Max concurrent QA: {mapreduce_config['max_concurrent_qa']}")

    # Print detailed results for each QA pair if verbose flag is set
    if args.verbose:
        print("\n===== Detailed Results =====")
        for i, qa_item in enumerate(results["qa_data"]):
            print(f"\nQuestion {i+1}: {qa_item['question']}")
            print(f"Document: {qa_item['doc_name']}")
            print(f"LLM Answer: {qa_item['llm_answer']}")
            print(f"Golden Answer: {qa_item['answer']}")
            print(f"Judgment: {qa_item.get('judgment', 'N/A')}")
            print(f"Reasoning: {qa_item.get('reasoning', 'N/A')}")

    print(f"\nResults saved to file")


if __name__ == "__main__":
    main()