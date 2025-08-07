from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional, Union
import concurrent.futures
from tqdm import tqdm
import time
import json
import os
from datetime import datetime
from pathlib import Path


class BaseMapReduceQA(ABC):
    """
    Abstract base class for MapReduce-based Question Answering pipelines.

    This class provides the core MapReduce workflow and parallel processing
    infrastructure, while allowing subclasses to customize specific behaviors
    through abstract methods.
    """

    def __init__(self,
                 llm: Any,
                 prompts_dict: Dict[str, Any],
                 chunk_size: int = 36000,
                 chunk_overlap: int = 1000,
                 max_concurrent_qa: int = 40,
                 max_concurrent_chunks: int = 40):
        """
        Initialize the MapReduce QA pipeline.

        Args:
            llm: Language model instance (can be None if subclass handles it)
            prompts_dict: Dictionary containing prompt templates
            chunk_size: Size of document chunks for processing
            chunk_overlap: Overlap between consecutive chunks
            max_concurrent_qa: Maximum concurrent QA pairs to process
            max_concurrent_chunks: Maximum concurrent chunks in map phase
        """
        self.llm = llm
        self.prompts_dict = prompts_dict
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_concurrent_qa = max_concurrent_qa
        self.max_concurrent_chunks = max_concurrent_chunks

    # ===== Abstract Methods - Must be implemented by subclasses =====

    @abstractmethod
    def load_data(self, data_path: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load QA data from dataset-specific format."""
        pass

    @abstractmethod
    def load_document_chunks(self, qa_pair: Dict[str, Any]) -> Tuple[List[Any], int]:
        """Load and chunk a document for the given QA pair."""
        pass

    @abstractmethod
    def preprocess_map_results(self, results: List[Dict[str, Any]]) -> List[Any]:
        """Filter/preprocess map phase results before reduce phase."""
        pass

    @abstractmethod
    def format_map_results_for_reduce(self, results: List[Any], question: str) -> Union[str, Dict]:
        """Format map results for the reduce phase."""
        pass

    @abstractmethod
    def parse_final_result(self, result: Any) -> Dict[str, Any]:
        """Parse the final result from reduce phase into standardized format."""
        pass

    def parse_final_result_with_map_data(self, reduce_result: Any, map_results: List[Any]) -> Dict[str, Any]:
        """
        Parse final result with access to both map and reduce results.

        Default implementation just calls parse_final_result for backward compatibility.
        Subclasses can override to use map_results for llm_evidence.

        Args:
            reduce_result: Result from reduce phase
            map_results: Filtered results from map phase (after preprocessing)

        Returns:
            Dictionary with llm_answer, llm_reasoning, llm_evidence
        """
        return self.parse_final_result(reduce_result)

    @abstractmethod
    def invoke_llm_map(self, chunk: Any, question: str) -> Dict[str, Any]:
        """Invoke LLM for map phase - handles different LLM interfaces."""
        pass

    @abstractmethod
    def invoke_llm_reduce(self, formatted_results: Any, question: str) -> Any:
        """Invoke LLM for reduce phase - handles different LLM interfaces."""
        pass

    # ===== Template Methods - Can be overridden if needed =====

    def get_document_identifier(self, qa_pair: Dict[str, Any]) -> str:
        """Get a display name for the document being processed."""
        return qa_pair.get("doc_name", "unknown")

    def get_results_directory(self) -> str:
        """Get the directory name for saving results."""
        return "results"

    def get_dataset_name(self) -> str:
        """Get the name of the dataset."""
        return "unknown"

    def get_judge_prompt_key(self) -> str:
        """Get the key for judge prompt in prompts_dict."""
        return 'judge_prompt'

    def get_evaluation_formatter_type(self) -> Optional[str]:
        """
        Get the evaluation formatter type for this pipeline.

        Override in subclasses to specify a particular formatter.
        Returns None to use auto-detection.
        """
        return None  # Auto-detect by default

    def get_map_question(self, qa_pair) -> str:
        """Get the map question"""
        return qa_pair["question"]


    # ===== Core Workflow Methods =====

    def process_single_qa(self, qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single QA pair through the MapReduce pipeline.

        This is the main template method that defines the MapReduce workflow.
        Subclasses customize behavior through the abstract methods.

        Args:
            qa_pair: Dictionary containing question and document information

        Returns:
            Updated qa_pair dictionary with results and statistics
        """

        question = qa_pair["question"]
        map_question = self.get_map_question(qa_pair=qa_pair)

        # Step 1: Load document chunks
        try:
            docs, token_count = self.load_document_chunks(qa_pair)
        except Exception as e:
            print(f"Error loading document for {self.get_document_identifier(qa_pair)}: {e}")
            return self._handle_document_error(qa_pair, str(e))

        if not docs:
            return self._handle_document_error(qa_pair, "No documents loaded")

        # Step 2: Map phase - process chunks in parallel
        map_results, map_tokens = self._map_phase(docs, map_question)

        if not map_results:
            return self._handle_processing_error(qa_pair, "No results from map phase")

        # Step 3: Preprocess/filter results
        filtered_results = self.preprocess_map_results(map_results)

        if not filtered_results:
            return self._handle_processing_error(qa_pair, "No results after preprocessing")

        # Step 4: Reduce phase - combine results
        final_result, reduce_tokens = self._reduce_phase(filtered_results, question)

        # Step 5: Parse and store results
        parsed_results = self.parse_final_result_with_map_data(final_result, filtered_results)
        qa_pair.update(parsed_results)

        # Step 6: Store token statistics
        qa_pair["token_stats"] = self._compile_token_stats(
            len(docs), map_tokens, reduce_tokens
        )

        return qa_pair

    def process_dataset(self,
                       data_path: str,
                       model_name: str,
                       num_samples: Optional[int] = None,
                       judge_llm: Optional[Any] = None,
                       **kwargs) -> Dict[str, Any]:
        """
        Process entire dataset through MapReduce pipeline.

        Args:
            data_path: Path to the dataset file
            model_name: Name of the model being used
            num_samples: Number of samples to process (None for all)
            judge_llm: Optional separate LLM for evaluation
            **kwargs: Additional arguments passed to subclasses

        Returns:
            Dictionary containing results and evaluation metrics
        """
        # Load data
        print(f"Loading {num_samples if num_samples else 'all'} samples from {self.get_dataset_name()} dataset...")
        qa_data = self.load_data(data_path, num_samples)

        if not qa_data:
            raise ValueError(f"No data loaded from {data_path}")

        # Print document information
        self._print_document_info(qa_data)

        t1 = time.time()
        print(f"Processing {len(qa_data)} QA pairs with {self.max_concurrent_qa} concurrent workers...")

        # Process QA pairs in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(qa_data),self.max_concurrent_qa)) as executor:
            future_to_qa = {
                executor.submit(self.process_single_qa, qa_pair): i
                for i, qa_pair in enumerate(qa_data)
            }

            with tqdm(total=len(qa_data), desc="Processing QA pairs", unit="pair") as pbar:
                for future in concurrent.futures.as_completed(future_to_qa):
                    qa_idx = future_to_qa[future]
                    try:
                        updated_qa_pair = future.result()
                        doc_name = self.get_document_identifier(qa_data[qa_idx])
                        pbar.update(1)
                        pbar.set_postfix({"file": os.path.basename(doc_name)[:30]})
                    except Exception as e:
                        pbar.write(f"Error processing QA pair {qa_idx+1}: {e}")
                        qa_data[qa_idx]["llm_answer"] = "Error during processing"
                        qa_data[qa_idx]["error"] = str(e)
                        qa_data[qa_idx]["token_stats"] = self._empty_token_stats()
                        pbar.update(1)

        process_time = time.time() - t1
        print(f"QA processing completed in {process_time:.1f} seconds ({process_time/len(qa_data):.1f}s per question)")

        # Evaluate with judge
        print("Evaluating answers using LLM judge...")
        judge = judge_llm if judge_llm is not None else self.llm
        evaluation_results = self._evaluate_with_judge(judge, qa_data)

        # Get judge model name
        judge_model_name = judge.get_model_name() if hasattr(judge, 'get_model_name') else model_name

        # Compile and save results
        results = self._compile_results(qa_data, evaluation_results, model_name, judge_model_name, process_time, **kwargs)
        results_file = self._save_results(results)

        # Print summary
        self._print_summary(evaluation_results, judge_model_name)

        return results

    # ===== Internal Methods =====

    def _map_phase(self, docs: List[Any], question: str) -> Tuple[List[Dict], Dict[str, int]]:
        """Execute map phase on document chunks."""
        results = []

        def process_chunk(chunk):
            try:
                return self.invoke_llm_map(chunk, question)
            except Exception as e:
                print(f"Error in map phase for chunk: {e}")
                return {"error": str(e), "content": ""}

        # Process chunks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(docs)) as executor:
            futures = {
                executor.submit(process_chunk, chunk): i
                for i, chunk in enumerate(docs)
            }

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing chunk: {e}")
                    results.append({"error": str(e)})

        # Calculate token usage
        tokens = self._calculate_map_tokens(results)
        return results, tokens

    def _reduce_phase(self, map_results: List[Any], question: str) -> Tuple[Any, Dict[str, int]]:
        """Execute reduce phase on map results."""
        # Format results for reduce
        formatted_results = self.format_map_results_for_reduce(map_results, question)

        # Invoke reduce
        result_final = self.invoke_llm_reduce(formatted_results, question)

        # Get token usage
        tokens = self._calculate_reduce_tokens(result_final)

        return result_final, tokens

    def _calculate_map_tokens(self, results: List[Dict]) -> Dict[str, int]:
        """Calculate token usage from map phase results."""
        input_tokens = 0
        output_tokens = 0
        cache_read_tokens = 0

        for result in results:
            # Handle different response formats
            if isinstance(result, dict):
                # GPT wrapper format
                raw_response = result.get('raw_response')
                if raw_response and hasattr(raw_response, 'usage_metadata') and raw_response.usage_metadata:
                    input_tokens += raw_response.usage_metadata.get("input_tokens", 0)
                    output_tokens += raw_response.usage_metadata.get("output_tokens", 0)
                    # Add cache read tokens if available
                    input_token_details = raw_response.usage_metadata.get("input_token_details", {})
                    cache_read_tokens += input_token_details.get("cache_read", 0)
                # Last year format
                elif result.get('usage'):
                    input_tokens += result['usage'].get("input_tokens", 0)
                    output_tokens += result['usage'].get("output_tokens", 0)

        return {"input_tokens": input_tokens, "output_tokens": output_tokens, "cache_read_tokens": cache_read_tokens}

    def _calculate_reduce_tokens(self, result: Any) -> Dict[str, int]:
        """Calculate token usage from reduce phase result."""
        input_tokens = 0
        output_tokens = 0
        cache_read_tokens = 0

        if isinstance(result, dict):
            raw_response = result.get('raw_response')
            if raw_response and hasattr(raw_response, 'usage_metadata') and raw_response.usage_metadata:
                input_tokens = raw_response.usage_metadata.get("input_tokens", 0)
                output_tokens = raw_response.usage_metadata.get("output_tokens", 0)
                # Add cache read tokens if available
                input_token_details = raw_response.usage_metadata.get("input_token_details", {})
                cache_read_tokens = input_token_details.get("cache_read", 0)
        elif hasattr(result, 'usage_metadata') and result.usage_metadata:
            input_tokens = result.usage_metadata.get("input_tokens", 0)
            output_tokens = result.usage_metadata.get("output_tokens", 0)
            # Add cache read tokens if available
            input_token_details = result.usage_metadata.get("input_token_details", {})
            cache_read_tokens = input_token_details.get("cache_read", 0)

        return {"input_tokens": input_tokens, "output_tokens": output_tokens, "cache_read_tokens": cache_read_tokens}

    def _compile_token_stats(self, num_docs: int, map_tokens: Dict, reduce_tokens: Dict) -> Dict:
        """Compile token statistics."""
        return {
            "len_docs": num_docs,
            "map_phase": map_tokens,
            "reduce_phase": reduce_tokens,
            "total": {
                "input_tokens": map_tokens["input_tokens"] + reduce_tokens["input_tokens"],
                "output_tokens": map_tokens["output_tokens"] + reduce_tokens["output_tokens"],
                "cache_read_tokens": map_tokens.get("cache_read_tokens", 0) + reduce_tokens.get("cache_read_tokens", 0)
            }
        }

    def _handle_document_error(self, qa_pair: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        """Handle document loading errors."""
        qa_pair["llm_answer"] = "Error: Could not load document"
        qa_pair["llm_reasoning"] = f"Document loading failed: {error_msg}"
        qa_pair["llm_evidence"] = []
        qa_pair["error"] = error_msg
        qa_pair["token_stats"] = self._empty_token_stats()
        return qa_pair

    def _handle_processing_error(self, qa_pair: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        """Handle processing errors."""
        qa_pair["llm_answer"] = "Error: Processing failed"
        qa_pair["llm_reasoning"] = error_msg
        qa_pair["llm_evidence"] = []
        qa_pair["error"] = error_msg
        qa_pair["token_stats"] = self._empty_token_stats()
        return qa_pair

    def _empty_token_stats(self) -> Dict[str, Any]:
        """Return empty token statistics structure."""
        return {
            "len_docs": 0,
            "map_phase": {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0},
            "reduce_phase": {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0},
            "total": {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0}
        }

    def _print_document_info(self, qa_data: List[Dict[str, Any]]):
        """Print information about documents to be processed."""
        print("\n=== Documents to be processed ===")
        doc_names = [self.get_document_identifier(qa_pair) for qa_pair in qa_data]
        for i, doc_name in enumerate(doc_names[:10]):  # Show first 10
            print(f"{i+1}/{len(doc_names)}: {os.path.basename(doc_name)}")
        if len(doc_names) > 10:
            print(f"... and {len(doc_names) - 10} more documents")
        print("===============================\n")

    def _evaluate_with_judge(self, judge_llm: Any, qa_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate results using LLM judge with automatic formatter detection.

        Subclasses can override get_evaluation_formatter_type() to specify
        a particular formatter.
        """
        from evaluation import evaluate_with_llm_judge

        # Get formatter type from subclass if specified
        formatter_type = self.get_evaluation_formatter_type()

        return evaluate_with_llm_judge(
            judge_llm,
            qa_data,
            self.prompts_dict,
            judge_prompt_key=self.get_judge_prompt_key(),
            formatter_type=formatter_type
        )

    def _compile_results(self, qa_data: List[Dict], evaluation_results: Dict,
                        model_name: str, judge_model_name: str, process_time: float, **kwargs) -> Dict:
        """Compile final results dictionary."""
        from utils import calculate_token_usage_summary, calculate_accuracy_by_question_type

        token_summary = calculate_token_usage_summary(qa_data)

        # Check if question_type exists in data
        has_question_type = any('question_type' in qa for qa in qa_data)
        accuracy_by_type = calculate_accuracy_by_question_type(qa_data) if has_question_type else {}

        # Get LLM configuration if available
        llm_config = {}
        if self.llm and hasattr(self.llm, 'get_model_name'):
            llm_config = {
                "model_name": self.llm.get_model_name() if hasattr(self.llm, 'get_model_name') else "unknown",
                "temperature": self.llm.get_temperature() if hasattr(self.llm, 'get_temperature') else None,
                "max_tokens": self.llm.get_max_tokens() if hasattr(self.llm, 'get_max_tokens') else None,
                "provider": self.llm.get_provider() if hasattr(self.llm, 'get_provider') else "unknown",
                "key": self.llm.get_key() if hasattr(self.llm, 'get_key') else None
            }

        results = {
            "configuration": {
                "dataset": self.get_dataset_name(),
                "model_name": model_name,
                "prompt_set": self.prompts_dict.get('prompt_set_name', 'unknown'),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "max_concurrent_qa": self.max_concurrent_qa,
                "approach": "MapReduce",
                "llm_configuration": llm_config
            },
            "execution_time": datetime.now().isoformat(),
            "time_taken": process_time,
            "num_samples": len(qa_data),
            "token_usage_summary": token_summary,
            "qa_data": qa_data,
            "evaluations": {
                judge_model_name: {
                    "judgment_distribution": {
                        "correct": evaluation_results["correct"],
                        "coherent": evaluation_results["coherent"],
                        "deviated": evaluation_results["deviated"],
                        "incorrect": evaluation_results["incorrect"],
                        "no_answer": evaluation_results["no_answer"]
                    },
                    "total": evaluation_results["total"],
                    "accuracy": evaluation_results["accuracy"],
                    "accuracy_by_question_type": accuracy_by_type,
                    "judgment_percentages": {
                        "correct": evaluation_results["correct"] / evaluation_results["total"] if evaluation_results["total"] > 0 else 0,
                        "coherent": evaluation_results["coherent"] / evaluation_results["total"] if evaluation_results["total"] > 0 else 0,
                        "deviated": evaluation_results["deviated"] / evaluation_results["total"] if evaluation_results["total"] > 0 else 0,
                        "incorrect": evaluation_results["incorrect"] / evaluation_results["total"] if evaluation_results["total"] > 0 else 0,
                        "no_answer": evaluation_results["no_answer"] / evaluation_results["total"] if evaluation_results["total"] > 0 else 0
                    },
                    "detailed_judgments": evaluation_results.get("detailed_judgments", [])
                }
            },
            "prompts_dict": {k: str(v) for k, v in self.prompts_dict.items()}
        }

        # Add any additional configuration from kwargs
        for key, value in kwargs.items():
            if key not in results["configuration"]:
                results["configuration"][key] = value

        return results

    def _save_results(self, results: Dict) -> str:
        """Save results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = self.get_results_directory()
        os.makedirs(results_dir, exist_ok=True)

        # Create filename
        prompt_name = self.prompts_dict.get('prompt_set_name', 'unknown')
        dataset_name = self.get_dataset_name()
        file_prefix = f"{prompt_name}_chunk{self.chunk_size}_overlap{self.chunk_overlap}_{results['num_samples']}_{dataset_name}"

        # Add any additional identifiers from configuration
        if 'pdf_parser' in results['configuration']:
            file_prefix += f"_{results['configuration']['pdf_parser']}"

        results_file = os.path.join(results_dir, f"{timestamp}_{file_prefix}.json")

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {results_file}")
        return results_file

    def _print_summary(self, evaluation_results: Dict, judge_model_name: str = "unknown"):
        """Print evaluation summary."""
        print(f"\n=== Evaluation Summary ({judge_model_name}) ===")
        print(f"Total: {evaluation_results['total']}")
        print(f"Correct: {evaluation_results['correct']} ({evaluation_results['correct']/evaluation_results['total']*100:.1f}%)")
        print(f"Coherent: {evaluation_results['coherent']} ({evaluation_results['coherent']/evaluation_results['total']*100:.1f}%)")
        print(f"Deviated: {evaluation_results['deviated']} ({evaluation_results['deviated']/evaluation_results['total']*100:.1f}%)")
        print(f"Incorrect: {evaluation_results['incorrect']} ({evaluation_results['incorrect']/evaluation_results['total']*100:.1f}%)")
        print(f"No answer: {evaluation_results['no_answer']} ({evaluation_results['no_answer']/evaluation_results['total']*100:.1f}%)")
        print(f"Overall Accuracy: {evaluation_results['accuracy']:.2%}")
        print("========================\n")