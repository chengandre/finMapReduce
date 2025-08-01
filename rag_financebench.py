"""
RAG FinanceBench Evaluation Pipeline

Evaluates the SimpleRAG system on FinanceBench dataset using the same evaluation
framework as the MapReduce pipeline for fair comparison.
"""

from simple_rag import SimpleRAG
from mapreduce_qa import load_financebench_data, evaluate_with_llm_judge
from utils import RateLimitedGPT, calculate_token_usage_summary, calculate_accuracy_by_question_type
from datetime import datetime
from tqdm import tqdm

import concurrent.futures
import time
import json
import os
import argparse
import logging
import warnings


def process_single_rag_qa(qa_pair, rag_system, llm, top_k=5):
    """
    Process a single QA pair using the RAG system.

    Args:
        qa_pair (dict): QA pair dictionary containing question and doc_name
        rag_system: SimpleRAG instance
        llm: LLM instance for token tracking
        top_k (int): Number of chunks to retrieve

    Returns:
        dict: Updated QA pair with RAG answer and token stats
    """
    # Get document name and question from the qa_pair
    doc_name = qa_pair["doc_name"]
    question = qa_pair["question"]

    # Create document ID from doc_name (remove extension and special chars)
    document_id = os.path.splitext(os.path.basename(doc_name))[0]
    document_id = document_id.replace(" ", "_").replace("-", "_")

    try:
        # Ingest document if not already present
        if not rag_system.check_document_exists(document_id):
            try:
                success = rag_system.ingest_document(
                    document_name=doc_name,
                    document_id=document_id,
                    on_duplicate="skip"
                )
                if not success:
                    print(f"Failed to ingest document: {doc_name}")
                    qa_pair["llm_answer"] = "Error: Could not ingest document"
                    qa_pair["llm_reasoning"] = "Document ingestion failed"
                    qa_pair["llm_evidence"] = []
                    qa_pair["error"] = "Document ingestion failed"
                    return qa_pair
            except Exception as e:
                print(f"Error ingesting document {doc_name}: {e}")
                qa_pair["llm_answer"] = "Error: Could not ingest document"
                qa_pair["llm_reasoning"] = f"Document ingestion error: {str(e)}"
                qa_pair["llm_evidence"] = []
                qa_pair["error"] = str(e)
                return qa_pair

        # Query the RAG system
        response = rag_system.query(
            question=question,
            document_id=document_id,
            top_k=top_k
        )

        # Extract structured components from JSON response
        json_response = response.json_response
        clean_answer = json_response.get("answer", "No answer provided")
        clean_reasoning = json_response.get("reasoning", "RAG-based retrieval and generation")
        clean_evidence = json_response.get("evidence", [])

        # If no evidence in JSON response, fall back to extracting from source chunks
        if not clean_evidence:
            for chunk in response.source_chunks:  # Limit to first 3 chunks
                if hasattr(chunk, 'page_content') and chunk.page_content:
                    evidence_snippet = chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content
                    clean_evidence.append(evidence_snippet)

        # Store the RAG answer and reasoning in qa_pair dictionary
        qa_pair["llm_answer"] = clean_answer
        qa_pair["llm_reasoning"] = clean_reasoning
        qa_pair["llm_evidence"] = clean_evidence

        # Since RAG doesn't use map-reduce, we'll estimate token usage based on query complexity
        # This is a rough approximation for comparison purposes
        estimated_input_tokens = len(question.split()) * 1.3 + sum(len(chunk.page_content.split()) * 1.3 for chunk in response.source_chunks)
        estimated_output_tokens = len(clean_answer.split()) * 1.3

        qa_pair["token_stats"] = {
            "retrieval_phase": {
                "input_tokens": int(estimated_input_tokens * 0.3),  # Embedding and search
                "output_tokens": 0
            },
            "generation_phase": {
                "input_tokens": int(estimated_input_tokens * 0.7),  # Context + question
                "output_tokens": int(estimated_output_tokens)
            },
            "total": {
                "input_tokens": int(estimated_input_tokens),
                "output_tokens": int(estimated_output_tokens)
            }
        }

        # Store additional RAG-specific metadata
        qa_pair["rag_metadata"] = {
            "chunks_retrieved": len(response.source_chunks),
            "query_time": response.query_time,
            "document_ids_used": response.document_ids,

            # Enhanced: Per-chunk details showing document filtering worked
            # "chunk_details": [
            #     {
            #         "chunk_index": i,
            #         "document_id": chunk.metadata.get("document_id"),
            #         "similarity_score": response.confidence_scores[i] if i < len(response.confidence_scores) else None,
            #         "chunk_size": len(chunk.page_content),
            #         "source_name": chunk.metadata.get("source_name", "unknown")
            #     }
            #     for i, chunk in enumerate(response.source_chunks)
            # ],

            # Enhanced: Confirmation that document filtering was applied
            "document_filtering_applied": document_id is not None,
            "target_document_id": document_id,

            # Keep original confidence_scores for backward compatibility
            "confidence_scores": response.confidence_scores
        }

    except Exception as e:
        print(f"Error processing QA pair for {doc_name}: {e}")
        qa_pair["llm_answer"] = "Error during RAG processing"
        qa_pair["llm_reasoning"] = f"RAG processing error: {str(e)}"
        qa_pair["llm_evidence"] = []
        qa_pair["error"] = str(e)
        qa_pair["token_stats"] = {
            "retrieval_phase": {"input_tokens": 0, "output_tokens": 0},
            "generation_phase": {"input_tokens": 0, "output_tokens": 0},
            "total": {"input_tokens": 0, "output_tokens": 0}
        }

    return qa_pair


def process_financebench_rag(jsonl_path, model_name, llm, num_samples=None,
                           max_concurrent_qa=3, top_k=5, embedding_model="all-MiniLM-L6-v2",
                           chunk_size=1000, chunk_overlap=200):
    """
    Process QA from financebench using RAG system with parallel processing.

    Args:
        jsonl_path (str): Path to financebench jsonl file
        model_name (str): Name of the LLM model
        llm: LLM instance to use
        num_samples (int): Number of samples to process
        max_concurrent_qa (int): Maximum number of QA pairs to process concurrently
        top_k (int): Number of chunks to retrieve for each query
        embedding_model (str): Embedding model to use
        chunk_size (int): Size of document chunks
        chunk_overlap (int): Overlap between chunks

    Returns:
        dict: Results containing model answers, golden answers, and evaluation results
    """
    print(f"Loading {num_samples if num_samples else 'all'} samples from financebench data...")
    qa_data = load_financebench_data(jsonl_path, num_samples)

    # Initialize RAG system
    print(f"Initializing RAG system with embedding model: {embedding_model}")

    # Suppress verbose logging and progress bars from embedding libraries
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("simple_rag").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    # Disable tokenizer parallelism warnings and tqdm for embeddings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Monkey patch tqdm to be silent for sentence-transformers
    def disable_tqdm():
        """Temporarily disable tqdm progress bars"""
        try:
            import sentence_transformers.util
            import transformers.utils.logging
            transformers.utils.logging.set_verbosity_error()

            # Replace tqdm with a silent version
            def silent_tqdm(iterable=None, *args, **kwargs):
                if iterable is not None:
                    return iterable
                else:
                    return tqdm(iterable, *args, disable=True, **kwargs)

            sentence_transformers.util.tqdm = silent_tqdm
        except ImportError:
            pass

    disable_tqdm()

    rag_system = SimpleRAG(
        llm=llm,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        use_local_embeddings=True
    )

    t1 = time.time()
    print(f"Processing {len(qa_data)} QA pairs with {max_concurrent_qa} concurrent workers...")

    # Process QA pairs in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_qa) as executor:
        # Submit all QA processing tasks
        future_to_qa = {
            executor.submit(process_single_rag_qa, qa_pair, rag_system, llm, top_k): i
            for i, qa_pair in enumerate(qa_data)
        }

        # Create progress bar
        with tqdm(total=len(qa_data), desc="Processing RAG QA pairs", unit="pair") as pbar:
            for future in concurrent.futures.as_completed(future_to_qa):
                qa_idx = future_to_qa[future]
                try:
                    future.result()  # Process the result but don't need to store it since qa_pair is modified in place
                    doc_name = qa_data[qa_idx]['doc_name']
                    pbar.update(1)
                    pbar.set_postfix({"file": os.path.basename(doc_name)[:20]})
                except Exception as e:
                    pbar.write(f"Error processing QA pair {qa_idx+1}: {e}")
                    qa_data[qa_idx]["llm_answer"] = "Error during processing"
                    qa_data[qa_idx]["error"] = str(e)
                    pbar.update(1)

    process_time = time.time() - t1
    print(f"RAG processing completed in {process_time:.1f} seconds ({process_time/len(qa_data):.1f}s per question)")

    # Get RAG system status for metadata
    rag_status = rag_system.get_status()
    print(f"RAG system processed {rag_status['total_documents']} unique documents")

    # Evaluate using LLM judge (reuse from mapreduce_qa.py)
    print("Evaluating answers using LLM judge...")
    evaluation_results = evaluate_with_llm_judge(llm, qa_data, batch_size=5)

    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "financebench_results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"rag_results_{timestamp}.json")

    # Calculate enhanced statistics
    token_summary = calculate_token_usage_summary(qa_data)
    accuracy_by_type = calculate_accuracy_by_question_type(qa_data)

    results = {
        "approach": "RAG",
        "model_name": model_name,
        "execution_time": datetime.now().isoformat(),
        "time_taken": time.time() - t1,
        "num_samples": len(qa_data),
        "qa_data": qa_data,

        # Enhanced: Token usage summary
        "token_usage_summary": token_summary,

        "rag_config": {
            "embedding_model": embedding_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "top_k": top_k,
            "max_concurrent_qa": max_concurrent_qa
        },
        "rag_system_status": rag_status,

        # Enhanced: Detailed evaluation summary
        "evaluation_summary": {
            "judgment_distribution": {
                "correct": evaluation_results["correct"],
                "coherent": evaluation_results["coherent"],
                "deviated": evaluation_results["deviated"],
                "incorrect": evaluation_results["incorrect"],
                "no_answer": evaluation_results["no_answer"]
            },
            "total": evaluation_results["total"],
            "accuracy": evaluation_results["accuracy"],

            # Enhanced: Accuracy by question type
            "accuracy_by_question_type": accuracy_by_type,

            # Enhanced: Judgment percentages
            "judgment_percentages": {
                "correct": evaluation_results["correct"] / evaluation_results["total"] if evaluation_results["total"] > 0 else 0,
                "coherent": evaluation_results["coherent"] / evaluation_results["total"] if evaluation_results["total"] > 0 else 0,
                "deviated": evaluation_results["deviated"] / evaluation_results["total"] if evaluation_results["total"] > 0 else 0,
                "incorrect": evaluation_results["incorrect"] / evaluation_results["total"] if evaluation_results["total"] > 0 else 0,
                "no_answer": evaluation_results["no_answer"] / evaluation_results["total"] if evaluation_results["total"] > 0 else 0
            },

            # Keep original detailed_judgments for backward compatibility
            "detailed_judgments": evaluation_results.get("detailed_judgments", [])
        }
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")
    print(f"Accuracy: {evaluation_results['accuracy']:.2%}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation on FinanceBench data with LLM evaluation")
    parser.add_argument('--jsonl_path', type=str, default="../financebench/data/financebench_open_source.jsonl",
                      help='Path to the financebench jsonl file')
    parser.add_argument('--model_name', type=str, default="gpt-4o-mini",
                      help='Name of the OpenAI model to use')
    parser.add_argument('--num_samples', type=int, default=None,
                      help='Number of samples to process from the dataset')
    parser.add_argument('--temperature', type=float, default=0.01,
                      help='Temperature parameter for the LLM')
    parser.add_argument('--chunk_size', type=int, default=1000,
                      help='Size of each document chunk for RAG')
    parser.add_argument('--chunk_overlap', type=int, default=200,
                      help='Overlap between chunks for RAG')
    parser.add_argument('--top_k', type=int, default=5,
                      help='Number of chunks to retrieve for each query')
    parser.add_argument('--embedding_model', type=str, default="all-MiniLM-L6-v2",
                      help='Embedding model to use for RAG')
    parser.add_argument('--verbose', action='store_true',
                      help='Print detailed results for each QA pair')
    parser.add_argument('--max_tokens', type=int, default=8000,
                      help='Maximum number of tokens for the LLM')
    parser.add_argument('--provider', type=str, default="openrouter",
                      help='Provider of the LLM')
    parser.add_argument('--max_concurrent_qa', type=int, default=3,
                      help='Maximum number of QA pairs to process concurrently')
    parser.add_argument('--quiet', action='store_true',
                      help='Reduce progress bar output')
    parser.add_argument('--key', type=str, default=None,
                      help='API key selector: "self" uses SELF_OPENAI_API_KEY, otherwise uses OPENAI_API_KEY')
    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    print(f"Provider: {args.provider}")
    llm = RateLimitedGPT(
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        provider=args.provider,
        key=args.key
    )

    print(f"Processing {args.num_samples if args.num_samples else 'all'} samples from {args.jsonl_path}")
    results = process_financebench_rag(
        jsonl_path=args.jsonl_path,
        model_name=args.model_name,
        llm=llm,
        num_samples=args.num_samples,
        max_concurrent_qa=args.max_concurrent_qa,
        top_k=args.top_k,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    # Print summary results
    eval_summary = results["evaluation_summary"]
    token_summary = results["token_usage_summary"]

    print("\n" + "="*60)
    print("RAG EVALUATION RESULTS")
    print("="*60)

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

    # Accuracy by question type
    print(f"\nACCURACY BY QUESTION TYPE:")
    accuracy_by_type = eval_summary['accuracy_by_question_type']
    for q_type, stats in accuracy_by_type.items():
        print(f"  {q_type}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")

    # RAG-specific metrics
    print(f"\nRAG SYSTEM STATUS:")
    rag_status = results["rag_system_status"]
    print(f"  Documents processed: {rag_status['total_documents']}")
    print(f"  Total chunks in vector store: {rag_status['vector_store_documents']}")
    print(f"  Embedding model: {rag_status['embedding_model']}")
    print(f"  Chunk size: {rag_status['chunk_size']}, Overlap: {rag_status['chunk_overlap']}")

    # Print detailed results for each QA pair if verbose flag is set
    if args.verbose:
        print("\n===== Detailed Results =====")
        for i, qa_item in enumerate(results["qa_data"]):
            print(f"\nQuestion {i+1}: {qa_item['question']}")
            print(f"Question type: {qa_item.get('question_type', 'N/A')}")
            print(f"Document: {qa_item['doc_name']}")
            print(f"RAG Answer: {qa_item['llm_answer']}")
            print(f"Golden Answer: {qa_item['answer']}")
            print(f"Judgment: {qa_item.get('judgment', 'N/A')}")
            print(f"Reasoning: {qa_item.get('reasoning', 'N/A')}")

            # Print RAG-specific metadata
            if 'rag_metadata' in qa_item:
                metadata = qa_item['rag_metadata']
                print(f"Chunks retrieved: {metadata.get('chunks_retrieved', 'N/A')}")
                print(f"Query time: {metadata.get('query_time', 'N/A'):.3f}s")

    print(f"\nResults saved to file")


if __name__ == "__main__":
    main()