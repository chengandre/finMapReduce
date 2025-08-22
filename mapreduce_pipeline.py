from typing import Dict, List, Any, Tuple, Optional, Union
from base_pipeline import BasePipeline
from output_formatter import OutputFormatter
from dataset_loader import DatasetLoader
from json_formatter import JSONFormatter
from plain_text_formatter import PlainTextFormatter
from hybrid_formatter import HybridFormatter


class MapReducePipeline(BasePipeline):
    """
    Unified MapReduce pipeline using composition for dataset loading and output formatting.

    This pipeline composes a DatasetLoader with a configurable OutputFormatter
    to handle different datasets (FinanceBench, FinQA) and output formats (plain text, JSON, hybrid).
    """

    def __init__(self,
                 llm: Any,
                 prompts_dict: Dict[str, Any],
                 dataset_loader: DatasetLoader,
                 format_type: str = "json",
                 score_threshold: int = 50,
                 map_llm: Optional[Any] = None,
                 reduce_llm: Optional[Any] = None,
                 question_improvement_llm: Optional[Any] = None,
                 **kwargs):
        """
        Initialize MapReduce pipeline with composed strategies.

        Args:
            llm: Primary LLM instance
            prompts_dict: Dictionary containing prompt templates
            dataset_loader: DatasetLoader instance for loading data and documents
            format_type: Output format type ("json", "plain_text", "hybrid")
            score_threshold: Score threshold for filtering (used by plain_text and hybrid)
            map_llm: LLM for map phase (hybrid format only)
            reduce_llm: LLM for reduce phase (hybrid format only)
            question_improvement_llm: LLM for question preprocessing (hybrid format only)
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(
            llm=llm,
            prompts_dict=prompts_dict,
            map_llm=map_llm,
            reduce_llm=reduce_llm,
            **kwargs
        )

        # Store dataset loader
        self.dataset_loader = dataset_loader

        # Initialize output formatter based on format_type
        self.output_formatter = self._create_formatter(
            format_type, prompts_dict, score_threshold, question_improvement_llm
        )

        # Set LLM instances for the formatter
        self.output_formatter.set_llms(self.map_llm, self.reduce_llm)

        self.format_type = format_type

    def _create_formatter(self,
                         format_type: str,
                         prompts_dict: Dict[str, Any],
                         score_threshold: int,
                         question_improvement_llm: Optional[Any]) -> OutputFormatter:
        """Create appropriate formatter based on format_type."""
        if format_type == "json":
            return JSONFormatter(prompts_dict)
        elif format_type == "plain_text":
            return PlainTextFormatter(prompts_dict, score_threshold)
        elif format_type == "hybrid":
            return HybridFormatter(prompts_dict, question_improvement_llm, score_threshold)
        else:
            raise ValueError(f"Unknown format_type: {format_type}. Must be 'json', 'plain_text', or 'hybrid'")

    # ===== Dataset Loader Delegation =====

    def load_data(self, data_path: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Delegate to dataset loader."""
        return self.dataset_loader.load_data(data_path, num_samples)

    def load_document_chunks(self, qa_pair: Dict[str, Any]) -> Tuple[List[Any], int]:
        """Delegate to dataset loader."""
        return self.dataset_loader.load_document_chunks(qa_pair, self.chunk_size, self.chunk_overlap)

    def get_document_identifier(self, qa_pair: Dict[str, Any]) -> str:
        """Delegate to dataset loader."""
        return self.dataset_loader.get_document_identifier(qa_pair)

    def get_results_directory(self) -> str:
        """Delegate to dataset loader."""
        return self.dataset_loader.get_results_directory()

    def get_dataset_name(self) -> str:
        """Get dataset name with format suffix."""
        base_name = self.dataset_loader.get_dataset_name()
        return f"{base_name}_{self.format_type}"

    def get_map_question(self, qa_pair: Dict[str, Any]) -> str:
        """Delegate to dataset loader."""
        return self.dataset_loader.get_map_question(qa_pair)

    # ===== Output Formatter Delegation =====

    def preprocess_map_results(self, results: List[Dict[str, Any]]) -> List[Any]:
        """Delegate to output formatter."""
        return self.output_formatter.preprocess_map_results(results)

    def format_map_results_for_reduce(self, results: List[Any], question: str) -> Union[str, Dict]:
        """Delegate to output formatter."""
        return self.output_formatter.format_map_results_for_reduce(results, question)

    def parse_final_result(self, result: Any) -> Dict[str, Any]:
        """Delegate to output formatter."""
        return self.output_formatter.parse_final_result(result)

    def parse_final_result_with_map_data(self, reduce_result: Any, map_results: List[Any]) -> Dict[str, Any]:
        """Delegate to output formatter."""
        return self.output_formatter.parse_final_result_with_map_data(reduce_result, map_results)

    def get_judge_prompt_key(self) -> str:
        """Delegate to output formatter."""
        return self.output_formatter.get_judge_prompt_key()

    def get_evaluation_formatter_type(self) -> Optional[str]:
        """Delegate to output formatter."""
        return self.output_formatter.get_evaluation_formatter_type()

    # ===== Pipeline Implementation (Abstract Methods) =====

    async def process_single_qa_async(self, qa_pair: Dict[str, Any], document_cache: Optional[Dict[str, Tuple[List[Any], int]]] = None) -> Dict[str, Any]:
        """
        MapReduce-specific implementation of single QA processing.
        """
        question = qa_pair["question"]
        map_question = self.get_map_question(qa_pair)

        # Step 1: Get document chunks (from cache or load)
        try:
            if document_cache is not None:
                doc_identifier = self.get_document_identifier(qa_pair)
                if doc_identifier in document_cache:
                    docs, _ = document_cache[doc_identifier]
                else:
                    return self._handle_document_error(qa_pair, f"Document {doc_identifier} not found in cache")
            else:
                # Fallback to loading individually
                docs, _ = self.load_document_chunks(qa_pair)
        except Exception as e:
            print(f"Error loading document for {self.get_document_identifier(qa_pair)}: {e}")
            return self._handle_document_error(qa_pair, str(e))

        if not docs:
            return self._handle_document_error(qa_pair, "No documents loaded")

        # Step 2: Async map phase
        map_results, map_tokens, map_time = await self._map_phase_async(docs, map_question)

        if not map_results:
            return self._handle_processing_error(qa_pair, "No results from map phase")

        # Step 3: Preprocess/filter results (keep sync)
        chunks_before_filtering = len(map_results)
        score_analysis = self._extract_score_analysis(map_results)
        filtered_results = self.preprocess_map_results(map_results)
        chunks_after_filtering = len(filtered_results)

        if not filtered_results:
            return self._handle_processing_error(qa_pair, "No results after preprocessing")

        # Step 4: Async reduce phase
        final_result, reduce_tokens, reduce_time = await self._reduce_phase_async(filtered_results, question)

        # Step 5: Parse and store results
        parsed_results = self.parse_final_result_with_map_data(final_result, filtered_results)
        qa_pair.update(parsed_results)

        # Step 6: Store token statistics and timing
        filtering_stats = self._calculate_filtering_stats(chunks_before_filtering, chunks_after_filtering, score_analysis)
        qa_pair["token_stats"] = self._compile_token_stats(
            len(docs), map_tokens, reduce_tokens, map_time, reduce_time, filtering_stats
        )

        return qa_pair

    def compile_statistics(self, qa_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compile MapReduce-specific statistics.
        """
        # Calculate phase-level token totals across dataset
        phase_token_totals = self._calculate_phase_token_totals(qa_data)

        # Calculate timing averages
        timing_averages = self._calculate_timing_averages(qa_data)

        # Calculate filtering effectiveness
        filtering_effectiveness = self._calculate_filtering_effectiveness(qa_data)

        return {
            "phase_token_totals": phase_token_totals,
            "timing_summary": timing_averages,
            "filtering_effectiveness": filtering_effectiveness,
        }

    def get_map_question(self, qa_pair: Dict[str, Any]) -> str:
        """Get the question to use for the map phase."""
        return self.dataset_loader.get_map_question(qa_pair)

    # ===== Async Method Overrides =====

    async def invoke_llm_map_async(self, chunk: Any, question: str) -> Dict[str, Any]:
        """Async version of invoke_llm_map, delegate to formatter."""
        return await self.output_formatter.invoke_llm_map_async(chunk, question)

    async def invoke_llm_reduce_async(self, formatted_results: Any, question: str) -> Any:
        """Async version of invoke_llm_reduce, delegate to formatter."""
        return await self.output_formatter.invoke_llm_reduce_async(formatted_results, question)

    # ===== Question Improvement =====

    def improve_question(self, original_question: str) -> Tuple[str, Dict[str, int]]:
        """
        Improve question if formatter supports it.

        Args:
            original_question: The original question text

        Returns:
            Tuple of (improved question text, token usage dict)
        """
        if hasattr(self.output_formatter, 'improve_question'):
            return self.output_formatter.improve_question(original_question)
        else:
            # Default behavior from base class
            return super().improve_question(original_question)

    # ===== Results Compilation =====

    def _compile_results(self, qa_data: List[Dict], evaluation_results: Dict,
                        model_name: str, judge_model_name: str, process_time: float, doc_load_time: float, **kwargs) -> Dict:
        """
        Override to add dataset and format-specific configuration.
        """
        results = super()._compile_results(qa_data, evaluation_results, model_name, judge_model_name, process_time, doc_load_time, **kwargs)

        # Add dataset-specific config
        results["configuration"] = self.dataset_loader.add_dataset_config(results["configuration"])

        # Add format-specific config
        results["configuration"] = self.output_formatter.add_format_config(results["configuration"])

        return results

    # ===== MapReduce-specific Internal Methods =====

    def _empty_token_stats(self) -> Dict[str, Any]:
        """Return empty token statistics structure for MapReduce."""
        return {
            "len_docs": 0,
            "map_phase": {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0},
            "reduce_phase": {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0},
            "timing": {
                "map_phase_time": 0.0,
                "reduce_phase_time": 0.0,
                "total_time": 0.0
            },
            "total": {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0},
            "filtering_stats": {
                "chunks_before_filtering": 0,
                "chunks_after_filtering": 0,
                "filtering_retention_rate": 0.0
            }
        }

    async def _map_phase_async(self, docs: List[Any], question: str) -> Tuple[List[Dict], Dict[str, int], float]:
        """Async map phase with semaphore management."""
        import asyncio
        start = asyncio.get_running_loop().time()

        async def process_chunk(chunk, idx):
            try:
                result = await self.invoke_llm_map_async(chunk, question)
                return idx, result
            except Exception as e:
                return idx, {"error": str(e), "content": ""}

        results = await asyncio.gather(
            *(process_chunk(c, i) for i, c in enumerate(docs))
        )

        results_sorted = [r for _, r in sorted(results, key=lambda x: x[0])]
        return results_sorted, self._calculate_map_tokens(results_sorted), asyncio.get_running_loop().time() - start

    async def _reduce_phase_async(self, map_results: List[Any], question: str) -> Tuple[Any, Dict[str, int], float]:
        """Async reduce phase."""
        import asyncio
        loop = asyncio.get_running_loop()
        reduce_start_time = loop.time()

        # Format results for reduce
        formatted_results = self.format_map_results_for_reduce(map_results, question)

        # Invoke reduce
        result_final = await self.invoke_llm_reduce_async(formatted_results, question)

        reduce_time = loop.time() - reduce_start_time

        # Get token usage
        tokens = self._calculate_reduce_tokens(result_final)

        return result_final, tokens, reduce_time

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

    def _compile_token_stats(self, num_docs: int, map_tokens: Dict, reduce_tokens: Dict, map_time: float, reduce_time: float, filtering_stats: Optional[Dict] = None) -> Dict:
        """Compile token statistics, timing, and filtering stats."""
        stats = {
            "len_docs": num_docs,
            "map_phase": map_tokens,
            "reduce_phase": reduce_tokens,
            "timing": {
                "map_phase_time": map_time,
                "reduce_phase_time": reduce_time,
                "total_time": map_time + reduce_time
            },
            "total": {
                "input_tokens": map_tokens["input_tokens"] + reduce_tokens["input_tokens"],
                "output_tokens": map_tokens["output_tokens"] + reduce_tokens["output_tokens"],
                "cache_read_tokens": map_tokens.get("cache_read_tokens", 0) + reduce_tokens.get("cache_read_tokens", 0)
            }
        }

        if filtering_stats:
            stats["filtering_stats"] = filtering_stats

        return stats

    def _extract_score_analysis(self, map_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract score distribution from map results if available."""
        import re

        scores = []
        score_threshold = None

        # Check if this pipeline uses score-based filtering
        if hasattr(self, 'output_formatter'):
            formatter = getattr(self, 'output_formatter', None)
            if formatter and hasattr(formatter, 'score_threshold'):
                score_threshold = getattr(formatter, 'score_threshold', None)

        for result in map_results:
            try:
                content = result.get('content', '')
                if "Score:" in content:
                    score_match = re.search(r'Score:\s*(\d+)', content)
                    if score_match:
                        score = int(score_match.group(1))
                        scores.append(score)
            except Exception:
                pass  # Ignore score extraction errors

        if scores:
            return {
                "score_distribution": {
                    "scores": scores,
                    "count": len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "avg": sum(scores) / len(scores)
                },
                "score_threshold_used": score_threshold
            }

        return {}

    def _calculate_filtering_stats(self, chunks_before: int, chunks_after: int, score_analysis: Optional[Dict] = None) -> Dict[str, Any]:
        """Calculate filtering statistics for a single QA pair."""
        retention_rate = chunks_after / chunks_before if chunks_before > 0 else 0.0
        stats = {
            "chunks_before_filtering": chunks_before,
            "chunks_after_filtering": chunks_after,
            "filtering_retention_rate": retention_rate
        }

        if score_analysis:
            stats.update(score_analysis)

        return stats

    def _calculate_phase_token_totals(self, qa_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        """Calculate dataset-level token totals for map and reduce phases separately."""
        map_totals = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0}
        reduce_totals = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0}

        for qa_pair in qa_data:
            token_stats = qa_pair.get('token_stats', {})

            # Aggregate map phase tokens
            map_phase = token_stats.get('map_phase', {})
            map_totals["input_tokens"] += map_phase.get("input_tokens", 0)
            map_totals["output_tokens"] += map_phase.get("output_tokens", 0)
            map_totals["cache_read_tokens"] += map_phase.get("cache_read_tokens", 0)

            # Aggregate reduce phase tokens
            reduce_phase = token_stats.get('reduce_phase', {})
            reduce_totals["input_tokens"] += reduce_phase.get("input_tokens", 0)
            reduce_totals["output_tokens"] += reduce_phase.get("output_tokens", 0)
            reduce_totals["cache_read_tokens"] += reduce_phase.get("cache_read_tokens", 0)

        return {
            "map_phase_total": map_totals,
            "reduce_phase_total": reduce_totals
        }

    def _calculate_timing_averages(self, qa_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate average and median timing statistics from all QA pairs."""
        import statistics

        map_times = []
        reduce_times = []
        total_times = []

        for qa_pair in qa_data:
            token_stats = qa_pair.get('token_stats', {})
            timing = token_stats.get('timing', {})

            if timing:
                map_time = timing.get('map_phase_time', 0.0)
                reduce_time = timing.get('reduce_phase_time', 0.0)
                total_time = timing.get('total_time', 0.0)

                if map_time > 0:
                    map_times.append(map_time)
                if reduce_time > 0:
                    reduce_times.append(reduce_time)
                if total_time > 0:
                    total_times.append(total_time)

        return {
            "average_map_phase_time": sum(map_times) / len(map_times) if map_times else 0.0,
            "average_reduce_phase_time": sum(reduce_times) / len(reduce_times) if reduce_times else 0.0,
            "average_total_mapreduce_time": sum(total_times) / len(total_times) if total_times else 0.0,
            "median_map_phase_time": statistics.median(map_times) if map_times else 0.0,
            "median_reduce_phase_time": statistics.median(reduce_times) if reduce_times else 0.0,
            "median_total_mapreduce_time": statistics.median(total_times) if total_times else 0.0,
            "samples_with_timing": len(total_times)
        }

    def _calculate_filtering_effectiveness(self, qa_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate dataset-level filtering effectiveness statistics."""
        import statistics

        retention_rates = []
        total_chunks_processed = 0
        total_chunks_retained = 0
        all_scores = []
        score_threshold_used = None

        for qa_pair in qa_data:
            token_stats = qa_pair.get('token_stats', {})
            filtering_stats = token_stats.get('filtering_stats', {})

            if filtering_stats:
                chunks_before = filtering_stats.get('chunks_before_filtering', 0)
                chunks_after = filtering_stats.get('chunks_after_filtering', 0)
                retention_rate = filtering_stats.get('filtering_retention_rate', 0.0)

                if chunks_before > 0:
                    retention_rates.append(retention_rate)
                    total_chunks_processed += chunks_before
                    total_chunks_retained += chunks_after

                # Collect score data if available
                score_distribution = filtering_stats.get('score_distribution', {})
                if score_distribution and 'scores' in score_distribution:
                    all_scores.extend(score_distribution['scores'])

                # Get score threshold (should be consistent across QA pairs)
                if score_threshold_used is None:
                    score_threshold_used = filtering_stats.get('score_threshold_used')

        effectiveness = {
            "dataset_avg_retention_rate": sum(retention_rates) / len(retention_rates) if retention_rates else 0.0,
            "dataset_median_retention_rate": statistics.median(retention_rates) if retention_rates else 0.0,
            "total_chunks_processed": total_chunks_processed,
            "total_chunks_retained": total_chunks_retained
        }

        # Add score analysis if scores were found
        if all_scores:
            effectiveness["score_distribution"] = {
                "total_scores": len(all_scores),
                "min": min(all_scores),
                "max": max(all_scores),
                "avg": sum(all_scores) / len(all_scores),
                "median": statistics.median(all_scores)
            }

        if score_threshold_used is not None:
            effectiveness["score_threshold_used"] = score_threshold_used

        return effectiveness