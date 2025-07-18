from mapreduce_qa import process_financebench_qa
from utils import GPT, RateLimitedGPT
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run MapReduce QA on FinanceBench data with LLM evaluation")
    parser.add_argument('--jsonl_path', type=str, default="../financebench/data/financebench_open_source.jsonl",
                      help='Path to the financebench jsonl file')
    parser.add_argument('--model_name', type=str, default="deepseek/deepseek-r1-0528:free",
                      help='Name of the OpenAI model to use')
    parser.add_argument('--num_samples', type=int, default=10,
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
    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    llm = RateLimitedGPT(model_name=args.model_name, temperature=args.temperature, max_tokens=args.max_tokens, provider=args.provider)

    print(f"Processing {args.num_samples if args.num_samples else 'all'} samples from {args.jsonl_path}")
    results = process_financebench_qa(
        jsonl_path=args.jsonl_path,
        model_name=args.model_name,
        llm=llm,
        num_samples=args.num_samples,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_concurrent_qa=args.max_concurrent_qa
    )

    # Print summary results
    print("\n===== Evaluation Summary =====")
    eval_summary = results["evaluation_summary"]
    print(f"Total samples: {eval_summary['total']}")
    print(f"Correct answers: {eval_summary['correct']}")
    print(f"Incorrect answers: {eval_summary['incorrect']}")
    print(f"No answers: {eval_summary['no_answer']}")
    print(f"Accuracy: {eval_summary['accuracy']:.2%}")
    print(f"Time taken: {results['time_taken']:.2f} seconds")

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