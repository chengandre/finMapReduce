from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional, Union
from async_evaluation import AsyncLLMJudgeEvaluator
import concurrent.futures
from tqdm import tqdm
import json
import os
import asyncio
from datetime import datetime


class BasePipeline(ABC):
    """
    Abstract base class for Question Answering pipelines.

    This class provides the core pipeline workflow and parallel processing
    infrastructure, while allowing subclasses to customize specific behaviors
    through abstract methods. Can support various pipeline architectures beyond MapReduce.
    
    Subclasses must set self.dataset_loader and self.formatter (or equivalent) components
    for delegation methods to work properly.
    """

    def __init__(self,
                 llm: Any,
                 prompts_dict: Dict[str, Any],
                 max_total_requests: int = 300,
                 judge_llm: Optional[Any] = None,
                 **kwargs):
        """
        Initialize the QA pipeline.

        Args:
            llm: Primary language model instance
            prompts_dict: Dictionary containing prompt templates
            max_total_requests: Maximum total concurrent requests across entire pipeline
            judge_llm: Optional separate LLM for evaluation (defaults to llm)
            **kwargs: Additional pipeline-specific arguments (chunk_size, pdf_parser, etc.)
        """
        self.llm = llm
        self.judge_llm = judge_llm if judge_llm is not None else llm
        self.prompts_dict = prompts_dict
        self.max_total_requests = max_total_requests
        self.global_semaphore = asyncio.Semaphore(max_total_requests)
        
        # Store additional config for subclass use
        self.config = kwargs

        self.judge_evaluator = None

    # ===== COMPONENT PROPERTIES =====
    
    @property
    def dataset_loader(self):
        """Access to dataset loader component - must be set by subclasses."""
        if not hasattr(self, '_dataset_loader'):
            raise NotImplementedError("Subclasses must set self.dataset_loader or override delegation methods")
        return self._dataset_loader
    
    @dataset_loader.setter 
    def dataset_loader(self, value):
        self._dataset_loader = value

    @property
    def formatter(self):
        """Access to formatter component - must be set by subclasses."""
        if not hasattr(self, '_formatter'):
            raise NotImplementedError("Subclasses must set self.formatter or override delegation methods")
        return self._formatter
        
    @formatter.setter
    def formatter(self, value):
        self._formatter = value

    # ===== INITIALIZATION & SETUP =====

    def _initialize_evaluator(self):
        """Initialize the judge evaluator after subclass components are ready."""
        if self.judge_evaluator is None:
            self.judge_evaluator = AsyncLLMJudgeEvaluator(
                llm=self.judge_llm,
                prompts_dict=self.prompts_dict,
                formatter_type=self.get_evaluation_formatter_type()
            )

    # ===== ABSTRACT CORE METHODS - Pipeline-specific implementations =====

    @abstractmethod
    async def process_single_qa_async(self, qa_pair: Dict[str, Any], document_cache: Optional[Dict[str, Tuple[List[Any], int]]] = None) -> Dict[str, Any]:
        """
        Process a single QA pair using pipeline-specific approach.

        Args:
            qa_pair: QA pair to process
            document_cache: Optional cache of preloaded documents {doc_identifier: (docs, token_count)}

        Returns:
            Updated qa_pair with results and token statistics
        """
        pass

    @abstractmethod
    def compile_statistics(self, qa_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compile pipeline-specific statistics from processed QA data.

        Args:
            qa_data: List of processed QA pairs with token_stats

        Returns:
            Dictionary with pipeline-specific statistics (e.g., phase_token_totals, filtering_effectiveness)
        """
        pass

    @abstractmethod
    def _empty_token_stats(self) -> Dict[str, Any]:
        """Return empty token statistics structure - pipeline specific."""
        pass

    # ===== CONCRETE DELEGATION METHODS =====
    # These delegate to composed components (dataset_loader, formatter)

    def get_document_identifier(self, qa_pair: Dict[str, Any]) -> str:
        """Get a display name for the document being processed."""
        return self.dataset_loader.get_document_identifier(qa_pair)

    def get_results_directory(self) -> str:
        """Get the directory name for saving results."""
        return self.dataset_loader.get_results_directory()

    def get_dataset_name(self) -> str:
        """Get the name of the dataset."""
        return self.dataset_loader.get_dataset_name()

    def get_judge_prompt_key(self) -> str:
        """Get the key for judge prompt in prompts_dict."""
        return self.formatter.get_judge_prompt_key()

    def get_evaluation_formatter_type(self) -> Optional[str]:
        """
        Get the evaluation formatter type for this pipeline.
        Returns None to use auto-detection.
        """
        return self.formatter.get_evaluation_formatter_type()

    def load_document_chunks(self, qa_pair: Dict[str, Any]) -> Tuple[List[Any], int]:
        """Load document chunks for a QA pair - delegates to composed components."""
        chunk_size = self.config.get('chunk_size', 36000)
        chunk_overlap = self.config.get('chunk_overlap', 1000)
        return self.dataset_loader.load_document_chunks(qa_pair, chunk_size, chunk_overlap)

    def load_data(self, data_path: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load QA data from the given path - delegates to composed components."""
        return self.dataset_loader.load_data(data_path, num_samples)


    # ===== DOCUMENT LOADING METHODS =====

    async def _batch_load_documents_async(self, qa_data: List[Dict[str, Any]]) -> Dict[str, Tuple[List[Any], int]]:
        """
        Batch load all unique documents in parallel using ThreadPool.

        Args:
            qa_data: List of QA pairs containing document identifiers

        Returns:
            Dictionary mapping document identifiers to (docs, token_count) tuples
        """
        # Extract unique document identifiers
        unique_docs = {}
        for qa_pair in qa_data:
            doc_id = self.get_document_identifier(qa_pair)
            if doc_id not in unique_docs:
                unique_docs[doc_id] = qa_pair

        print(f"Loading {len(unique_docs)} unique documents in parallel...")

        # Load documents in parallel using ThreadPool
        document_cache = {}

        def load_single_document(doc_item):
            doc_id, qa_pair = doc_item
            try:
                docs, token_count = self.load_document_chunks(qa_pair)
                return doc_id, (docs, token_count), None
            except Exception as e:
                return doc_id, None, str(e)

        # Use ThreadPoolExecutor for I/O bound document loading
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(unique_docs)) as executor:
            # Submit all loading tasks
            futures = {
                executor.submit(load_single_document, doc_item): doc_item[0]
                for doc_item in unique_docs.items()
            }

            # Collect results with progress bar
            try:
                from tqdm import tqdm
                pbar = tqdm(total=len(futures), desc="Loading documents")

                for future in concurrent.futures.as_completed(futures):
                    doc_id, result, error = future.result()
                    if error:
                        print(f"Error loading document {doc_id}: {error}")
                        # Store empty result for failed documents
                        document_cache[doc_id] = ([], 0)
                    else:
                        document_cache[doc_id] = result
                    pbar.update(1)
                    pbar.set_postfix({"loaded": len([k for k, v in document_cache.items() if v[0]])})

                pbar.close()
            except ImportError:
                # Fallback without progress bar
                for future in concurrent.futures.as_completed(futures):
                    doc_id, result, error = future.result()
                    if error:
                        print(f"Error loading document {doc_id}: {error}")
                        document_cache[doc_id] = ([], 0)
                    else:
                        document_cache[doc_id] = result

        # Report loading statistics
        successful_loads = len([k for k, v in document_cache.items() if v[0]])
        failed_loads = len(document_cache) - successful_loads
        print(f"Document loading completed: {successful_loads} successful, {failed_loads} failed")

        return document_cache

    # ===== MAIN PROCESSING PIPELINE =====

    async def process_dataset_async(self,
                                   data_path: str,
                                   model_name: str,
                                   num_samples: Optional[int] = None,
                                   judge_llm: Optional[Any] = None,
                                   **kwargs) -> Dict[str, Any]:
        """
        Async version of process_dataset.
        """
        # Load data (keep sync)
        print(f"Loading {num_samples if num_samples else 'all'} samples from {self.get_dataset_name()} dataset...")
        qa_data = self.load_data(data_path, num_samples)

        if not qa_data:
            raise ValueError(f"No data loaded from {data_path}")

        # Print document information
        self._print_document_info(qa_data)

        # Batch load all documents in parallel
        print("Batch loading documents...")
        loop = asyncio.get_running_loop()
        doc_load_start = loop.time()
        document_cache = await self._batch_load_documents_async(qa_data)
        doc_load_time = loop.time() - doc_load_start
        print(f"Loaded {len(document_cache)} unique documents in {doc_load_time:.1f} seconds")

        # Question preprocessing (async version)
        question_improvement_tokens = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0}
        if kwargs.get('preprocess_questions', False):
            print("Preprocessing questions...")
            qa_data, question_improvement_tokens = await self._preprocess_questions_async(qa_data)

        loop = asyncio.get_running_loop()
        t1 = loop.time()
        print(f"Processing {len(qa_data)} QA pairs")

        tasks = [
            asyncio.create_task(self.process_single_qa_async(qa, document_cache))
            for qa in qa_data
        ]

        results = []
        from tqdm import tqdm
        pbar = tqdm(total=len(tasks), desc="Processing QA pairs")
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            pbar.update(1)
        pbar.close()

        # Update qa_data with results
        qa_data[:] = results

        process_time = loop.time() - t1
        print(f"QA processing completed in {process_time:.1f} seconds ({process_time/len(qa_data):.1f}s per question)")

        # Evaluate with judge using async method
        print("Evaluating answers using LLM judge...")
        judge = judge_llm if judge_llm is not None else self.judge_llm
        evaluation_results = await self._evaluate_with_judge_async(judge, qa_data)

        # Get judge model name
        judge_model_name = judge.get_model_name() if hasattr(judge, 'get_model_name') else model_name

        # Compile and save results
        results = self._compile_results(
            qa_data, evaluation_results, model_name, judge_model_name,
            process_time, doc_load_time, question_improvement_tokens=question_improvement_tokens, **kwargs
        )

        # Save results (run in executor to avoid blocking)
        _ = await loop.run_in_executor(None, self._save_results, results)

        # Print summary
        self._print_summary(evaluation_results, judge_model_name)

        return results

    # ===== LLM INTERACTION METHODS =====

    async def ainvoke_llm_judge(self, judge_prompt: str) -> Any:
        """
        Async version of invoke_llm_judge with global semaphore.
        """
        async with self.global_semaphore:
            return await self.judge_llm.ainvoke(judge_prompt)

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

    # ===== EVALUATION METHODS =====

    async def _evaluate_with_judge_async(self, judge_llm: Any, qa_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Async version of evaluation with judge using AsyncLLMJudgeEvaluator.
        """
        return await self.judge_evaluator.evaluate_async(qa_data)

    # ===== ERROR HANDLING =====

    def _handle_error(self, qa_pair: Dict[str, Any], error_msg: str, error_type: str = "processing") -> Dict[str, Any]:
        """Handle pipeline errors with appropriate messages based on error type."""
        error_messages = {
            "document": "Error: Could not load document",
            "processing": "Error: Processing failed",
            "llm": "Error: LLM call failed"
        }
        
        qa_pair["llm_answer"] = error_messages.get(error_type, "Error: Unknown failure")
        qa_pair["llm_reasoning"] = error_msg if error_type == "processing" else f"{error_type.title()} failed: {error_msg}"
        qa_pair["llm_evidence"] = []
        qa_pair["error"] = error_msg
        qa_pair["token_stats"] = self._empty_token_stats()
        return qa_pair
    
    def _handle_document_error(self, qa_pair: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        """Handle document loading errors - delegates to _handle_error."""
        return self._handle_error(qa_pair, error_msg, "document")
    
    def _handle_processing_error(self, qa_pair: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        """Handle processing errors - delegates to _handle_error."""
        return self._handle_error(qa_pair, error_msg, "processing")

    def _print_document_info(self, qa_data: List[Dict[str, Any]]):
        """Print information about documents to be processed."""
        print("\n=== Documents to be processed ===")
        doc_names = [self.get_document_identifier(qa_pair) for qa_pair in qa_data]
        for i, doc_name in enumerate(doc_names[:10]):  # Show first 10
            print(f"{i+1}/{len(doc_names)}: {os.path.basename(doc_name)}")
        if len(doc_names) > 10:
            print(f"... and {len(doc_names) - 10} more documents")
        print("===============================\n")

    # ===== RESULT COMPILATION & OUTPUT =====

    def _compile_results(self, qa_data: List[Dict], evaluation_results: Dict,
                        model_name: str, judge_model_name: str, process_time: float, doc_load_time: float, **kwargs) -> Dict:
        """Compile final results dictionary."""
        from utils import calculate_token_usage_summary, calculate_accuracy_by_question_type, calculate_accuracy_by_question_reasoning

        # Extract question improvement tokens from kwargs
        question_improvement_tokens = kwargs.pop('question_improvement_tokens', {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0})

        token_summary = calculate_token_usage_summary(qa_data)

        # Get pipeline-specific statistics from subclass
        pipeline_stats = self.compile_statistics(qa_data)

        # Add evaluation tokens if available
        evaluation_tokens = evaluation_results.get("evaluation_tokens", {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0})
        pipeline_stats["evaluation_phase_total"] = evaluation_tokens

        # Add document loading time to timing
        if "timing_summary" in pipeline_stats:
            pipeline_stats["timing_summary"]["document_loading_time"] = doc_load_time
            pipeline_stats["timing_summary"]["total_pipeline_time"] = process_time + doc_load_time

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
                "max_total_requests": self.max_total_requests,
                "llm_configuration": llm_config
            },
            "execution_time": datetime.now().isoformat(),
            "time_taken": process_time,
            "document_loading_time": doc_load_time,
            "num_samples": len(qa_data),
            "token_usage_summary": token_summary,
            "question_improvement_tokens": question_improvement_tokens,
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

        # Add pipeline-specific statistics
        results.update(pipeline_stats)

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
        # Build filename with available config
        file_parts = [prompt_name]
        if 'chunk_size' in self.config:
            file_parts.append(f"chunk{self.config['chunk_size']}")
        if 'chunk_overlap' in self.config:
            file_parts.append(f"overlap{self.config['chunk_overlap']}")
        file_parts.extend([str(results['num_samples']), dataset_name])
        file_prefix = "_".join(file_parts)

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

    # ===== QUESTION PREPROCESSING METHODS =====

    async def _preprocess_questions_async(self, qa_data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Preprocess questions using async requests to make them clearer and more effective.

        Args:
            qa_data: List of QA pairs with original questions

        Returns:
            Tuple of (updated qa_data with improved questions, aggregated token usage)
        """
        print(f"Preprocessing {len(qa_data)} questions using async requests...")

        async def improve_single_question_async(qa_pair):
            try:
                improved_question, tokens = await self.improve_question_async(qa_pair["question"])
                qa_pair["original_question"] = qa_pair["question"]
                qa_pair["question"] = improved_question
                return qa_pair, tokens
            except Exception as e:
                print(f"Error improving question: {e}")
                empty_tokens = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0}
                return qa_pair, empty_tokens

        # Track total token usage
        total_tokens = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0}

        # Create async tasks for all questions
        tasks = [
            asyncio.create_task(improve_single_question_async(qa_pair))
            for qa_pair in qa_data
        ]

        # Process tasks with progress bar
        results = []
        from tqdm import tqdm
        pbar = tqdm(total=len(tasks), desc="Improving questions", unit="question")
        for task in asyncio.as_completed(tasks):
            try:
                improved_qa_pair, tokens = await task
                results.append((improved_qa_pair, tokens))
                # Aggregate token usage
                total_tokens["input_tokens"] += tokens["input_tokens"]
                total_tokens["output_tokens"] += tokens["output_tokens"]
                total_tokens["cache_read_tokens"] += tokens["cache_read_tokens"]
                pbar.update(1)
            except Exception as e:
                print(f"Error processing question: {e}")
                pbar.update(1)
        pbar.close()

        # Update qa_data with results in original order
        for i, (improved_qa_pair, _) in enumerate(results):
            qa_data[i] = improved_qa_pair

        print("Question preprocessing completed.")
        print(f"Question improvement tokens: {total_tokens['input_tokens']} input, {total_tokens['output_tokens']} output, {total_tokens['cache_read_tokens']} cache read")
        return qa_data, total_tokens

    async def improve_question_async(self, original_question: str) -> Tuple[str, Dict[str, int]]:
        """
        Improve a single question to make it clearer and more effective using async request.
        Uses the question_improvement_prompt if available, otherwise returns original.

        Args:
            original_question: The original question text

        Returns:
            Tuple of (improved question text, token usage dict)
        """
        empty_tokens = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0}

        # Check if question improvement prompt is available
        if 'question_improvement_prompt' not in self.prompts_dict:
            print("Warning: question_improvement_prompt not available, keeping original questions")
            return original_question, empty_tokens

        try:
            # Use async LLM call with global semaphore for rate limiting
            prompt_template = self.prompts_dict['question_improvement_prompt']
            
            # Handle both string templates and PromptTemplate objects
            if hasattr(prompt_template, 'format'):
                prompt = prompt_template.format(question=original_question)
            else:
                prompt = str(prompt_template).format(question=original_question)
            
            # Choose appropriate LLM (allows subclasses to override via getattr)
            llm_to_use = getattr(self, 'question_improvement_llm', None) or getattr(self, 'reduce_llm', None) or self.llm
            
            async with self.global_semaphore:
                response = await llm_to_use.ainvoke(prompt)

            # Extract token usage
            tokens = self._extract_token_usage_from_response(response)

            # Extract improved question from JSON response
            if isinstance(response, dict) and 'json' in response:
                json_data = response['json']
                if isinstance(json_data, dict) and 'improved_question' in json_data:
                    improved = json_data['improved_question'].strip()
                    if improved:
                        return improved, tokens

            # Fallback to original if JSON parsing fails
            print(f"Warning: Could not parse improved question, using original")
            return original_question, tokens

        except Exception as e:
            print(f"Error improving question, using original: {e}")
            return original_question, empty_tokens

    # ===== UTILITY & SERIALIZATION =====

    def _serialize_prompts(self, prompts_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize prompts dictionary to preserve LangChain structure for easy loading.

        Args:
            prompts_dict: Dictionary containing prompt objects

        Returns:
            Serialized prompts dictionary with metadata for reconstruction
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

    @staticmethod
    def load_prompts_from_json(json_file_path: str) -> Dict[str, Any]:
        """
        Load prompts from a JSON results file and reconstruct LangChain objects.

        Args:
            json_file_path: Path to the JSON results file

        Returns:
            Dictionary with reconstructed prompt objects
        """
        from langchain.prompts import PromptTemplate

        with open(json_file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)

        prompts_data = results.get('prompts_dict', {})
        prompts_dict = {}

        for key, prompt_data in prompts_data.items():
            if isinstance(prompt_data, dict) and 'type' in prompt_data:
                if prompt_data['type'] == 'langchain_prompt_template':
                    # Reconstruct LangChain PromptTemplate
                    prompts_dict[key] = PromptTemplate(
                        template=prompt_data['template'],
                        input_variables=prompt_data.get('input_variables', [])
                    )
                else:
                    # String prompt
                    prompts_dict[key] = prompt_data['template']
            else:
                # Backward compatibility with old format
                prompts_dict[key] = prompt_data if isinstance(prompt_data, str) else str(prompt_data)

        return prompts_dict