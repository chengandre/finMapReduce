from typing import Dict, List, Any, Tuple, Optional, Union
from base_pipeline import BasePipeline
from dataset_loader import DatasetLoader
from truncation_formatter import TruncationFormatter
import asyncio
import time


class TruncationPipeline(BasePipeline):
    """
    Truncation-based QA pipeline using composition for dataset loading and truncation formatting.

    This pipeline composes a DatasetLoader with a TruncationFormatter to handle
    different datasets (FinanceBench, FinQA) with document truncation strategies.
    """

    def __init__(self,
                 llm: Any,
                 prompts_dict: Dict[str, Any],
                 dataset_loader: DatasetLoader,
                 truncation_formatter: TruncationFormatter,
                 **kwargs):
        """
        Initialize Truncation pipeline with composed strategies.

        Args:
            llm: Primary LLM instance
            prompts_dict: Dictionary containing prompt templates
            dataset_loader: DatasetLoader instance for loading data and documents
            truncation_formatter: TruncationFormatter instance for truncation operations
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(
            llm=llm,
            prompts_dict=prompts_dict,
            **kwargs
        )

        # Store composed components
        self.dataset_loader = dataset_loader
        self.truncation_formatter = truncation_formatter

        # Set LLM instance for the formatter
        self.truncation_formatter.set_llm(self.llm)

    # ===== Pipeline Implementation (Abstract Methods) =====

    async def process_single_qa_async(self, qa_pair: Dict[str, Any], document_cache: Optional[Dict[str, Tuple[str, int]]] = None) -> Dict[str, Any]:
        """
        Truncation-specific implementation of single QA processing.
        """
        question = qa_pair["question"]

        # Step 1: Load full document (from cache or load)
        try:
            if document_cache is not None:
                doc_identifier = self.get_document_identifier(qa_pair)
                if doc_identifier in document_cache:
                    doc_text, original_token_count = document_cache[doc_identifier]
                else:
                    return self._handle_document_error(qa_pair, f"Document {doc_identifier} not found in cache")
            else:
                # Fallback to loading individually
                doc_text, original_token_count = await self._load_document_for_qa_async(qa_pair)
        except Exception as e:
            print(f"Error loading document for {self.get_document_identifier(qa_pair)}: {e}")
            return self._handle_document_error(qa_pair, str(e))

        if not doc_text:
            return self._handle_document_error(qa_pair, "No document content loaded")

        # Step 2: Truncate document using formatter
        try:
            start_time = time.time()
            truncated_text, truncation_stats = self.truncation_formatter.truncate_document(doc_text, question)
            truncation_time = time.time() - start_time
        except Exception as e:
            print(f"Error truncating document for {self.get_document_identifier(qa_pair)}: {e}")
            return self._handle_processing_error(qa_pair, f"Document truncation failed: {str(e)}")

        # Step 3: Single LLM call with truncated document
        try:
            llm_start_time = time.time()
            llm_result = await self.truncation_formatter.invoke_llm_truncation_async(truncated_text, question)
            llm_time = time.time() - llm_start_time
        except Exception as e:
            print(f"Error in LLM call for {self.get_document_identifier(qa_pair)}: {e}")
            return self._handle_processing_error(qa_pair, f"LLM call failed: {str(e)}")

        # Step 4: Parse result using formatter
        try:
            parsed_results = self.truncation_formatter.parse_final_result(llm_result)
            qa_pair.update(parsed_results)
        except Exception as e:
            print(f"Error parsing result for {self.get_document_identifier(qa_pair)}: {e}")
            return self._handle_processing_error(qa_pair, f"Result parsing failed: {str(e)}")

        # Step 5: Store token statistics and timing
        llm_tokens = self._extract_token_usage_from_response(llm_result)
        qa_pair["token_stats"] = self._compile_token_stats(
            original_token_count, truncation_stats, llm_tokens, llm_time, truncation_time
        )

        return qa_pair

    def compile_statistics(self, qa_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compile Truncation-specific statistics.
        """
        # Calculate truncation summary
        truncation_summary = self._calculate_truncation_summary(qa_data)

        # Calculate timing averages
        timing_averages = self._calculate_timing_averages(qa_data)

        return {
            "truncation_summary": truncation_summary,
            "timing_summary": timing_averages,
        }

    # ===== Dataset Loader Delegation =====

    def load_data(self, data_path: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Delegate to dataset loader."""
        return self.dataset_loader.load_data(data_path, num_samples)

    def get_document_identifier(self, qa_pair: Dict[str, Any]) -> str:
        """Delegate to dataset loader."""
        return self.dataset_loader.get_document_identifier(qa_pair)

    def get_results_directory(self) -> str:
        """Delegate to dataset loader."""
        return "truncation_results"  # Override for truncation-specific directory

    def get_dataset_name(self) -> str:
        """Get dataset name with truncation suffix."""
        base_name = self.dataset_loader.get_dataset_name()
        strategy = getattr(self.truncation_formatter, 'strategy', 'truncation')
        return f"{base_name}_{strategy}"

    def load_document_chunks(self, qa_pair: Dict[str, Any]) -> Tuple[List[Any], int]:
        """Load document chunks - for compatibility with batch loading in base class."""
        return self.dataset_loader.load_document_chunks(qa_pair, self.chunk_size, self.chunk_overlap)

    async def _load_document_for_qa_async(self, qa_pair: Dict[str, Any]) -> Tuple[str, int]:
        """Load full document for truncation approach."""
        return self.dataset_loader.load_full_document(qa_pair)

    # ===== Truncation Formatter Delegation =====

    def get_judge_prompt_key(self) -> str:
        """Delegate to truncation formatter."""
        return self.truncation_formatter.get_judge_prompt_key()

    def get_evaluation_formatter_type(self) -> Optional[str]:
        """Delegate to truncation formatter."""
        return self.truncation_formatter.get_evaluation_formatter_type()

    # ===== Batch Document Loading Override =====

    async def _batch_load_documents_async(self, qa_data: List[Dict[str, Any]]) -> Dict[str, Tuple[str, int]]:
        """
        Override to load full documents for truncation (not chunks).
        """
        # Extract unique document identifiers
        unique_docs = {}
        for qa_pair in qa_data:
            doc_id = self.get_document_identifier(qa_pair)
            if doc_id not in unique_docs:
                unique_docs[doc_id] = qa_pair

        print(f"Loading {len(unique_docs)} unique documents (full text) in parallel...")

        # Load documents in parallel using ThreadPool
        document_cache = {}

        def load_single_document(doc_item):
            doc_id, qa_pair = doc_item
            try:
                doc_text, token_count = self.dataset_loader.load_full_document(qa_pair)
                return doc_id, (doc_text, token_count), None
            except Exception as e:
                return doc_id, None, str(e)

        # Use ThreadPoolExecutor for I/O bound document loading
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(unique_docs)) as executor:
            # Submit all loading tasks
            futures = {
                executor.submit(load_single_document, doc_item): doc_item[0]
                for doc_item in unique_docs.items()
            }

            # Collect results with progress bar
            try:
                from tqdm import tqdm
                pbar = tqdm(total=len(futures), desc="Loading full documents")

                for future in concurrent.futures.as_completed(futures):
                    doc_id, result, error = future.result()
                    if error:
                        print(f"Error loading document {doc_id}: {error}")
                        # Store empty result for failed documents
                        document_cache[doc_id] = ("", 0)
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
                        document_cache[doc_id] = ("", 0)
                    else:
                        document_cache[doc_id] = result

        # Report loading statistics
        successful_loads = len([k for k, v in document_cache.items() if v[0]])
        failed_loads = len(document_cache) - successful_loads
        print(f"Document loading completed: {successful_loads} successful, {failed_loads} failed")

        return document_cache

    # ===== Results Compilation =====

    def _compile_results(self, qa_data: List[Dict], evaluation_results: Dict,
                        model_name: str, judge_model_name: str, process_time: float, doc_load_time: float, **kwargs) -> Dict:
        """
        Override to add dataset and truncation-specific configuration.
        """
        results = super()._compile_results(qa_data, evaluation_results, model_name, judge_model_name, process_time, doc_load_time, **kwargs)

        # Add dataset-specific config
        results["configuration"] = self.dataset_loader.add_dataset_config(results["configuration"])

        # Add truncation-specific config
        results["configuration"] = self.truncation_formatter.add_truncation_config(results["configuration"])

        return results

    # ===== Truncation-specific Internal Methods =====

    def _empty_token_stats(self) -> Dict[str, Any]:
        """Return empty token statistics structure for Truncation."""
        return {
            "original_document_tokens": 0,
            "truncation_stats": {
                "strategy": getattr(self.truncation_formatter, 'strategy', 'unknown'),
                "truncated_tokens": 0,
                "retention_rate": 0.0,
                "truncation_applied": False
            },
            "llm_call": {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0},
            "timing": {
                "llm_call_time": 0.0,
                "truncation_time": 0.0,
                "total_time": 0.0
            },
            "total": {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0}
        }

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
                            llm_tokens: Dict, llm_time: float, truncation_time: float) -> Dict:
        """Compile token statistics and timing for truncation."""
        return {
            "original_document_tokens": original_tokens,
            "truncation_stats": truncation_stats,
            "llm_call": llm_tokens,
            "timing": {
                "llm_call_time": llm_time,
                "truncation_time": truncation_time,
                "total_time": llm_time + truncation_time
            },
            "total": llm_tokens.copy()
        }

    def _calculate_truncation_summary(self, qa_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate dataset-level truncation statistics."""
        import statistics

        original_tokens = []
        truncated_tokens = []
        retention_rates = []
        truncations_applied = 0

        strategy = getattr(self.truncation_formatter, 'strategy', 'unknown')
        context_window = getattr(self.truncation_formatter, 'context_window', 0)
        truncation_buffer = getattr(self.truncation_formatter, 'buffer', 0)

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
            "strategy": strategy,
            "context_window": context_window,
            "truncation_buffer": truncation_buffer,
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
        truncation_times = []
        total_times = []

        for qa_pair in qa_data:
            token_stats = qa_pair.get('token_stats', {})
            timing = token_stats.get('timing', {})

            if timing:
                llm_time = timing.get('llm_call_time', 0.0)
                truncation_time = timing.get('truncation_time', 0.0)
                total_time = timing.get('total_time', 0.0)

                if llm_time > 0:
                    llm_times.append(llm_time)
                if truncation_time > 0:
                    truncation_times.append(truncation_time)
                if total_time > 0:
                    total_times.append(total_time)

        return {
            "average_llm_call_time": sum(llm_times) / len(llm_times) if llm_times else 0.0,
            "average_truncation_time": sum(truncation_times) / len(truncation_times) if truncation_times else 0.0,
            "average_total_time": sum(total_times) / len(total_times) if total_times else 0.0,
            "median_llm_call_time": statistics.median(llm_times) if llm_times else 0.0,
            "median_truncation_time": statistics.median(truncation_times) if truncation_times else 0.0,
            "median_total_time": statistics.median(total_times) if total_times else 0.0,
            "samples_with_timing": len(total_times)
        }