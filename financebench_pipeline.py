from typing import Dict, List, Any, Tuple, Optional, Union
from base_mapreduce_qa import BaseMapReduceQA
from output_formatter import OutputFormatter
from financebench_loader import FinanceBenchLoader
from json_formatter import JSONFormatter
from plain_text_formatter import PlainTextFormatter
from hybrid_formatter import HybridFormatter


class FinanceBenchPipeline(BaseMapReduceQA):
    """
    FinanceBench pipeline using composition for dataset loading and output formatting.

    This pipeline composes a FinanceBenchLoader with a configurable OutputFormatter
    to handle different output formats (plain text, JSON, hybrid) for the same dataset.
    """

    def __init__(self,
                 llm: Any,
                 prompts_dict: Dict[str, Any],
                 format_type: str = "json",
                 pdf_parser: str = "marker",
                 score_threshold: int = 50,
                 map_llm: Optional[Any] = None,
                 reduce_llm: Optional[Any] = None,
                 question_improvement_llm: Optional[Any] = None,
                 **kwargs):
        """
        Initialize FinanceBench pipeline with composed strategies.

        Args:
            llm: Primary LLM instance
            prompts_dict: Dictionary containing prompt templates
            format_type: Output format type ("json", "plain_text", "hybrid")
            pdf_parser: PDF parsing method for FinanceBenchLoader
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
            pdf_parser=pdf_parser,
            **kwargs
        )

        # Initialize dataset loader
        self.dataset_loader = FinanceBenchLoader(pdf_parser=pdf_parser)

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

    # Delegate to dataset loader
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

    # Async method overrides for better performance when using async LLM clients
    async def invoke_llm_map_async(self, chunk: Any, question: str) -> Dict[str, Any]:
        """Async version of invoke_llm_map, delegate to formatter."""
        return await self.output_formatter.invoke_llm_map_async(chunk, question)

    async def invoke_llm_reduce_async(self, formatted_results: Any, question: str) -> Any:
        """Async version of invoke_llm_reduce, delegate to formatter."""
        return await self.output_formatter.invoke_llm_reduce_async(formatted_results, question)

    def get_judge_prompt_key(self) -> str:
        """Delegate to output formatter."""
        return self.output_formatter.get_judge_prompt_key()

    def get_evaluation_formatter_type(self) -> Optional[str]:
        """Delegate to output formatter."""
        return self.output_formatter.get_evaluation_formatter_type()

    # Question improvement for hybrid format
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