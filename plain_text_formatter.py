import re
import asyncio
from typing import Dict, List, Any, Optional
from output_formatter import OutputFormatter


class PlainTextFormatter(OutputFormatter):
    """
    Output formatter for plain text map/reduce operations.

    Features:
    - Direct LLM invocation (no JSON wrapper)
    - Score-based filtering from text content
    - String concatenation for reduce phase
    - Simple text parsing for final results
    """

    def __init__(self, prompts_dict: Dict[str, Any], score_threshold: int = 50):
        """
        Initialize plain text formatter.

        Args:
            prompts_dict: Dictionary containing prompt templates
            score_threshold: Minimum score threshold for filtering results
        """
        super().__init__(prompts_dict)
        self.score_threshold = score_threshold


    def preprocess_map_results(self, results: List[Dict[str, Any]]) -> List[str]:
        """
        Filter based on score extraction from text content.

        Only keeps results with Score > score_threshold.

        Args:
            results: List of map phase results

        Returns:
            List of filtered content strings
        """
        modified_results = []

        for result in results:
            try:
                content = result.get('content', '')
                if "Score:" in content:
                    score_match = re.search(r'Score:\s*(\d+)', content)
                    if score_match:
                        score = int(score_match.group(1))
                        if score > self.score_threshold:
                            modified_results.append(content)
            except Exception as e:
                print(f"Error preprocessing result: {e}")

        return modified_results

    def format_map_results_for_reduce(self, results: List[str], question: str) -> str:
        """
        Simple string concatenation for reduce phase input.

        Args:
            results: List of filtered content strings
            question: The original question (unused)

        Returns:
            Concatenated string of results
        """
        return "\n".join(results)

    def parse_final_result(self, result: Any) -> Dict[str, Any]:
        """
        Parse plain text result from reduce phase.

        Args:
            result: LLM response object

        Returns:
            Dictionary with llm_answer, llm_reasoning, llm_evidence
        """
        if hasattr(result, 'content'):
            answer = result.content
        else:
            answer = str(result)

        return {
            "llm_answer": answer,
            "llm_reasoning": "Score-based evaluation",
            "llm_evidence": []
        }

    def parse_final_result_with_map_data(self, reduce_result: Any, map_results: List[str]) -> Dict[str, Any]:
        """
        Parse final result with map results included in llm_evidence.

        Args:
            reduce_result: Result from reduce phase
            map_results: Filtered map results (strings with Score > threshold)

        Returns:
            Dictionary with llm_answer, llm_reasoning, and map_results as llm_evidence
        """
        if hasattr(reduce_result, 'content'):
            answer = reduce_result.content
        else:
            answer = str(reduce_result)

        return {
            "llm_answer": answer,
            "llm_reasoning": "Score-based evaluation",
            "llm_evidence": map_results
        }

    def get_evaluation_formatter_type(self) -> Optional[str]:
        """Use last year evaluation formatter for plain text."""
        return 'baseline'

    def add_format_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Add plain text formatter configuration."""
        config = super().add_format_config(config)
        config["approach"] = "Plain Text MapReduce"
        config["score_threshold"] = self.score_threshold
        return config

    # Async method overrides for better performance
    async def invoke_llm_map_async(self, chunk: Any, question: str) -> Dict[str, Any]:
        """
        Async version of invoke_llm_map.

        Args:
            chunk: Document chunk with page_content attribute
            question: The question to answer

        Returns:
            Dictionary with 'content' and 'usage' keys
        """
        prompt = self.prompts_dict['map_prompt'].format(
            context=chunk.page_content,
            question=question
        )

        response = await self.map_llm.invoke(prompt)

        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            return {
                'content': response.content,
                'usage': response.usage_metadata
            }
        else:
            return {
                'content': response.content if hasattr(response, 'content') else str(response),
                'usage': {}
            }

    async def invoke_llm_reduce_async(self, formatted_results: Any, question: str) -> Any:
        """
        Async version of invoke_llm_reduce.

        Args:
            formatted_results: String concatenation of map results
            question: The original question

        Returns:
            Response object from LLM
        """
        prompt = self.prompts_dict['reduce_prompt'].format(
            summaries=formatted_results,
            question_final=question
        )

        return await self.reduce_llm.invoke(prompt)