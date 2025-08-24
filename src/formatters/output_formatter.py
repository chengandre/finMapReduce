from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union, Optional, Tuple
import asyncio


class OutputFormatter(ABC):
    """
    Abstract base class for formatting map/reduce outputs and LLM interactions.

    This strategy class handles all output format-specific operations:
    - LLM invocation patterns (direct vs wrapper)
    - Result parsing and preprocessing
    - Format-specific configurations
    """

    def __init__(self, prompts_dict: Dict[str, Any]):
        """
        Initialize the output formatter.

        Args:
            prompts_dict: Dictionary containing prompt templates
        """
        self.prompts_dict = prompts_dict

    def set_llms(self, map_llm: Any, reduce_llm: Any):
        """
        Set the LLM instances for this formatter.

        Called by the pipeline to inject LLM dependencies.

        Args:
            map_llm: LLM instance for map phase
            reduce_llm: LLM instance for reduce phase
        """
        self.map_llm = map_llm
        self.reduce_llm = reduce_llm

    @abstractmethod
    def preprocess_map_results(self, results: List[Dict[str, Any]]) -> List[Any]:
        """
        Filter/preprocess map phase results before reduce phase.

        Args:
            results: List of map phase results

        Returns:
            Filtered/processed list of results
        """
        pass

    @abstractmethod
    def format_map_results_for_reduce(self, results: List[Any], question: str) -> Union[str, Dict]:
        """
        Format map results for the reduce phase input.

        Args:
            results: List of preprocessed map results
            question: The original question

        Returns:
            Formatted input for reduce phase (string or dict)
        """
        pass

    @abstractmethod
    def parse_final_result(self, result: Any) -> Dict[str, Any]:
        """
        Parse the final result from reduce phase into standardized format.

        Args:
            result: Result from reduce phase

        Returns:
            Dictionary with llm_answer, llm_reasoning, llm_evidence
        """
        pass

    def parse_final_result_with_map_data(self, reduce_result: Any, map_results: List[Any]) -> Dict[str, Any]:
        """
        Parse final result with access to both map and reduce results.

        Default implementation just calls parse_final_result for backward compatibility.
        Override to use map_results for llm_evidence.

        Args:
            reduce_result: Result from reduce phase
            map_results: Filtered results from map phase (after preprocessing)

        Returns:
            Dictionary with llm_answer, llm_reasoning, llm_evidence
        """
        return self.parse_final_result(reduce_result)

    def get_judge_prompt_key(self) -> str:
        """
        Get the key for judge prompt in prompts_dict.

        Returns:
            Key string for judge prompt
        """
        return 'judge_prompt'

    def get_evaluation_formatter_type(self) -> Optional[str]:
        """
        Get the evaluation formatter type for this output format.

        Override in subclasses to specify a particular formatter.
        Returns None to use auto-detection.

        Returns:
            Formatter type string or None for auto-detection
        """
        return None

    def improve_question(self, original_question: str) -> Tuple[str, Dict[str, int]]:
        """
        Improve a single question to make it clearer and more effective.

        Default implementation returns the original question unchanged.
        Override in formatters that support question improvement.

        Args:
            original_question: The original question text

        Returns:
            Tuple of (improved question text, token usage dict)
        """
        empty_tokens = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0}
        return original_question, empty_tokens

    def add_format_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add format-specific configuration to results.

        Default implementation adds basic approach info.
        Override to add format-specific configuration.

        Args:
            config: Existing configuration dictionary

        Returns:
            Updated configuration dictionary
        """
        config["approach"] = "MapReduce"
        return config

    # Async methods with default implementations
    @abstractmethod
    async def ainvoke_llm_map(self, chunk: Any, question: str) -> Dict[str, Any]:
        """
        Async version of invoke_llm_map.

        Args:
            chunk: Document chunk with page_content attribute
            question: The question to answer

        Returns:
            Dictionary with format-specific response structure
        """
        pass

    @abstractmethod
    async def ainvoke_llm_reduce(self, formatted_results: Any, question: str) -> Any:
        """
        Async version of invoke_llm_reduce.

        Args:
            formatted_results: Formatted map results
            question: The original question

        Returns:
            Format-specific response object
        """
        pass