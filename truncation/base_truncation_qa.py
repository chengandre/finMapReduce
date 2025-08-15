from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional
import concurrent.futures
from tqdm import tqdm
import time
import json
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BaseTruncationQA(ABC):
    """
    Abstract base class for Truncation-based Question Answering pipelines.

    This class provides the core truncation workflow and infrastructure,
    while allowing subclasses to customize specific behaviors through
    abstract methods.

    Unlike MapReduce which processes document chunks in parallel,
    truncation approaches load the full document, truncate it to fit
    the model's context window, and make a single LLM call.
    """

    def __init__(self,
                 llm: Any,
                 prompts_dict: Dict[str, Any],
                 truncation_strategy: str = "start",
                 context_window: int = 128000,
                 truncation_buffer: int = 2000,
                 max_document_tokens: Optional[int] = None,
                 max_concurrent_qa: int = 20,
                 judge_llm: Optional[Any] = None):
        """
        Initialize the Truncation QA pipeline.

        Args:
            llm: Primary language model instance
            prompts_dict: Dictionary containing prompt templates
            truncation_strategy: Strategy for truncation ("start", "end", "smart")
            context_window: Maximum context window size for the model
            truncation_buffer: Safety buffer for prompt tokens
            max_document_tokens: Maximum tokens to use from document (None = auto-calculate)
            max_concurrent_qa: Maximum concurrent QA pairs to process
            judge_llm: Optional separate LLM for evaluation (defaults to llm)
        """
        self.llm = llm
        self.judge_llm = judge_llm if judge_llm is not None else llm
        self.prompts_dict = prompts_dict
        self.truncation_strategy = truncation_strategy
        self.context_window = context_window
        self.truncation_buffer = truncation_buffer
        self.max_document_tokens = max_document_tokens
        self.max_concurrent_qa = max_concurrent_qa

    # ===== Abstract Methods - Must be implemented by subclasses =====

    @abstractmethod
    def load_data(self, data_path: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load QA data from dataset-specific format."""
        pass

    @abstractmethod
    def load_full_document(self, qa_pair: Dict[str, Any]) -> Tuple[str, int]:
        """
        Load the full document content for a QA pair.
        
        Args:
            qa_pair: Dictionary containing document information
            
        Returns:
            Tuple of (document_text, token_count)
        """
        pass

    @abstractmethod
    def invoke_llm_direct(self, document_text: str, question: str) -> Any:
        """
        Invoke LLM directly with truncated document and question.
        
        Args:
            document_text: Truncated document content
            question: Question to answer
            
        Returns:
            LLM response
        """
        pass

    @abstractmethod
    def parse_result(self, llm_result: Any) -> Dict[str, Any]:
        """
        Parse the LLM result into standardized format.
        
        Args:
            llm_result: Raw LLM response
            
        Returns:
            Dictionary with llm_answer, llm_reasoning, llm_evidence
        """
        pass

    # ===== Template Methods - Can be overridden if needed =====

    def get_document_identifier(self, qa_pair: Dict[str, Any]) -> str:
        """Get a display name for the document being processed."""
        return qa_pair.get("doc_name", "unknown")

    def get_results_directory(self) -> str:
        """Get the directory name for saving results."""
        return "truncation_results"

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
        return None

    def get_prompt_key(self) -> str:
        """Get the key for the main prompt in prompts_dict."""
        return 'map_prompt'

    # ===== Core Workflow Methods =====

    def process_single_qa(self, qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single QA pair through the truncation pipeline.

        This is the main template method that defines the truncation workflow.
        Subclasses customize behavior through the abstract methods.

        Args:
            qa_pair: Dictionary containing question and document information

        Returns:
            Updated qa_pair dictionary with results and statistics
        """
        question = qa_pair["question"]

        # Step 1: Load full document
        try:
            doc_text, original_token_count = self.load_full_document(qa_pair)
        except Exception as e:
            print(f"Error loading document for {self.get_document_identifier(qa_pair)}: {e}")
            return self._handle_document_error(qa_pair, str(e))

        if not doc_text:
            return self._handle_document_error(qa_pair, "No document content loaded")

        # Step 2: Calculate truncation parameters and truncate document
        try:
            truncated_text, truncation_stats = self._truncate_document(doc_text, question)
        except Exception as e:
            print(f"Error truncating document for {self.get_document_identifier(qa_pair)}: {e}")
            return self._handle_processing_error(qa_pair, f"Document truncation failed: {str(e)}")

        # Step 3: Single LLM call with truncated document
        try:
            start_time = time.time()
            llm_result = self.invoke_llm_direct(truncated_text, question)
            llm_time = time.time() - start_time
        except Exception as e:
            print(f"Error in LLM call for {self.get_document_identifier(qa_pair)}: {e}")
            return self._handle_processing_error(qa_pair, f"LLM call failed: {str(e)}")

        # Step 4: Parse result
        try:
            parsed_results = self.parse_result(llm_result)
            qa_pair.update(parsed_results)
        except Exception as e:
            print(f"Error parsing result for {self.get_document_identifier(qa_pair)}: {e}")
            return self._handle_processing_error(qa_pair, f"Result parsing failed: {str(e)}")

        # Step 5: Store token statistics and timing
        llm_tokens = self._extract_token_usage_from_response(llm_result)
        qa_pair["token_stats"] = self._compile_token_stats(
            original_token_count, truncation_stats, llm_tokens, llm_time
        )

        return qa_pair

    def process_dataset(self,
                       data_path: str,
                       model_name: str,
                       num_samples: Optional[int] = None,
                       judge_llm: Optional[Any] = None,
                       **kwargs) -> Dict[str, Any]:
        """
        Process entire dataset through truncation pipeline.

        Args:
            data_path: Path to the dataset file
            model_name: Name of the model being used
            num_samples: Number of samples to process (None for all)
            judge_llm: Optional separate LLM for evaluation
            **kwargs: Additional arguments

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
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(qa_data), self.max_concurrent_qa)) as executor:
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
        judge = judge_llm if judge_llm is not None else self.judge_llm
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

    def _truncate_document(self, doc_text: str, question: str) -> Tuple[str, Dict[str, Any]]:
        """
        Truncate document based on strategy and token limits.
        
        Args:
            doc_text: Full document text
            question: Question being asked
            
        Returns:
            Tuple of (truncated_text, truncation_statistics)
        """
        from truncation_utils import TruncationManager
        
        # Calculate available tokens for document
        if self.max_document_tokens is not None:
            max_doc_tokens = self.max_document_tokens
        else:
            # Estimate prompt tokens (question + template overhead)
            from utils import num_tokens_from_string
            prompt_tokens = num_tokens_from_string(question, "cl100k_base") + self.truncation_buffer
            max_doc_tokens = self.context_window - prompt_tokens

        # Create truncation manager and truncate
        manager = TruncationManager(
            strategy=self.truncation_strategy,
            max_tokens=max_doc_tokens
        )
        
        return manager.truncate_document(doc_text)

    def _extract_token_usage_from_response(self, response: Any) -> Dict[str, int]:
        """Extract token usage from LLM response."""
        tokens = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0}

        if isinstance(response, dict):
            raw_response = response.get('raw_response')
            if raw_response and hasattr(raw_response, 'usage_metadata') and raw_response.usage_metadata:
                tokens["input_tokens"] = raw_response.usage_metadata.get("input_tokens", 0)
                tokens["output_tokens"] = raw_response.usage_metadata.get("output_tokens", 0)
                input_token_details = raw_response.usage_metadata.get("input_token_details", {})
                tokens["cache_read_tokens"] = input_token_details.get("cache_read", 0)
        elif hasattr(response, 'usage_metadata') and response.usage_metadata:
            tokens["input_tokens"] = response.usage_metadata.get("input_tokens", 0)
            tokens["output_tokens"] = response.usage_metadata.get("output_tokens", 0)
            input_token_details = response.usage_metadata.get("input_token_details", {})
            tokens["cache_read_tokens"] = input_token_details.get("cache_read", 0)

        return tokens

    def _compile_token_stats(self, original_tokens: int, truncation_stats: Dict, 
                            llm_tokens: Dict, llm_time: float) -> Dict:
        """Compile token statistics and timing."""
        return {
            "original_document_tokens": original_tokens,
            "truncation_stats": truncation_stats,
            "llm_call": llm_tokens,
            "timing": {
                "llm_call_time": llm_time,
                "total_time": llm_time
            },
            "total": llm_tokens.copy()
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
            "original_document_tokens": 0,
            "truncation_stats": {
                "strategy": self.truncation_strategy,
                "truncated_tokens": 0,
                "retention_rate": 0.0,
                "truncation_applied": False
            },
            "llm_call": {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0},
            "timing": {
                "llm_call_time": 0.0,
                "total_time": 0.0
            },
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
        from utils import calculate_token_usage_summary, calculate_accuracy_by_question_type, calculate_accuracy_by_question_reasoning

        token_summary = calculate_token_usage_summary(qa_data)

        # Calculate truncation statistics
        truncation_summary = self._calculate_truncation_summary(qa_data)

        # Calculate timing averages
        timing_averages = self._calculate_timing_averages(qa_data)

        # Check if question_type exists in data
        has_question_type = any('question_type' in qa for qa in qa_data)
        accuracy_by_type = calculate_accuracy_by_question_type(qa_data) if has_question_type else {}

        # Check if question_reasoning exists in data
        has_question_reasoning = any('question_reasoning' in qa for qa in qa_data)
        accuracy_by_reasoning = calculate_accuracy_by_question_reasoning(qa_data) if has_question_reasoning else {}

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
                "truncation_strategy": self.truncation_strategy,
                "context_window": self.context_window,
                "truncation_buffer": self.truncation_buffer,
                "max_document_tokens": self.max_document_tokens,
                "max_concurrent_qa": self.max_concurrent_qa,
                "approach": "Truncation",
                "llm_configuration": llm_config
            },
            "execution_time": datetime.now().isoformat(),
            "time_taken": process_time,
            "num_samples": len(qa_data),
            "token_usage_summary": token_summary,
            "truncation_summary": truncation_summary,
            "timing_summary": timing_averages,
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
                    "accuracy_by_question_reasoning": accuracy_by_reasoning,
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
            "prompts_dict": self._serialize_prompts(self.prompts_dict)
        }

        # Add any additional configuration from kwargs
        for key, value in kwargs.items():
            if key not in results["configuration"]:
                results["configuration"][key] = value

        return results

    def _calculate_truncation_summary(self, qa_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate dataset-level truncation statistics."""
        import statistics
        
        original_tokens = []
        truncated_tokens = []
        retention_rates = []
        truncations_applied = 0

        for qa_pair in qa_data:
            token_stats = qa_pair.get('token_stats', {})
            original_tokens.append(token_stats.get('original_document_tokens', 0))
            
            truncation_stats = token_stats.get('truncation_stats', {})
            if truncation_stats:
                truncated_tokens.append(truncation_stats.get('truncated_tokens', 0))
                retention_rates.append(truncation_stats.get('retention_rate', 0.0))
                if truncation_stats.get('truncation_applied', False):
                    truncations_applied += 1

        return {
            "strategy": self.truncation_strategy,
            "context_window": self.context_window,
            "truncation_buffer": self.truncation_buffer,
            "truncations_applied": truncations_applied,
            "truncation_rate": truncations_applied / len(qa_data) if qa_data else 0.0,
            "original_tokens": {
                "avg": sum(original_tokens) / len(original_tokens) if original_tokens else 0.0,
                "median": statistics.median(original_tokens) if original_tokens else 0.0,
                "max": max(original_tokens) if original_tokens else 0,
                "min": min(original_tokens) if original_tokens else 0
            },
            "truncated_tokens": {
                "avg": sum(truncated_tokens) / len(truncated_tokens) if truncated_tokens else 0.0,
                "median": statistics.median(truncated_tokens) if truncated_tokens else 0.0,
                "max": max(truncated_tokens) if truncated_tokens else 0,
                "min": min(truncated_tokens) if truncated_tokens else 0
            },
            "retention_rates": {
                "avg": sum(retention_rates) / len(retention_rates) if retention_rates else 0.0,
                "median": statistics.median(retention_rates) if retention_rates else 0.0,
                "max": max(retention_rates) if retention_rates else 0.0,
                "min": min(retention_rates) if retention_rates else 0.0
            }
        }

    def _calculate_timing_averages(self, qa_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate average and median timing statistics from all QA pairs."""
        import statistics

        llm_times = []
        total_times = []

        for qa_pair in qa_data:
            token_stats = qa_pair.get('token_stats', {})
            timing = token_stats.get('timing', {})

            if timing:
                llm_time = timing.get('llm_call_time', 0.0)
                total_time = timing.get('total_time', 0.0)

                if llm_time > 0:
                    llm_times.append(llm_time)
                if total_time > 0:
                    total_times.append(total_time)

        return {
            "average_llm_call_time": sum(llm_times) / len(llm_times) if llm_times else 0.0,
            "average_total_time": sum(total_times) / len(total_times) if total_times else 0.0,
            "median_llm_call_time": statistics.median(llm_times) if llm_times else 0.0,
            "median_total_time": statistics.median(total_times) if total_times else 0.0,
            "samples_with_timing": len(total_times)
        }

    def _save_results(self, results: Dict) -> str:
        """Save results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = self.get_results_directory()
        os.makedirs(results_dir, exist_ok=True)

        # Create filename
        prompt_name = self.prompts_dict.get('prompt_set_name', 'unknown')
        dataset_name = self.get_dataset_name()
        strategy = self.truncation_strategy
        context_window = self.context_window
        file_prefix = f"{prompt_name}_truncation_{strategy}_context{context_window}_{results['num_samples']}_{dataset_name}"

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

    def _serialize_prompts(self, prompts_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize prompts dictionary to preserve LangChain structure for easy loading.
        """
        serialized = {}

        for key, prompt in prompts_dict.items():
            if hasattr(prompt, 'template'):
                # LangChain PromptTemplate
                serialized[key] = {
                    'type': 'langchain_prompt_template',
                    'template': prompt.template,
                    'input_variables': getattr(prompt, 'input_variables', [])
                }
            elif isinstance(prompt, str):
                # Simple string prompt
                serialized[key] = {
                    'type': 'string',
                    'template': prompt
                }
            else:
                # Fallback to string representation
                serialized[key] = {
                    'type': 'string',
                    'template': str(prompt)
                }

        return serialized