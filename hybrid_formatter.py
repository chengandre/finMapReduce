import re
from typing import Dict, List, Any, Optional, Tuple
from output_formatter import OutputFormatter


class HybridFormatter(OutputFormatter):
    """
    Output formatter for hybrid map/reduce operations.

    Combines:
    - Map phase: Plain text output with score-based filtering
    - Reduce phase: JSON output format
    - Supports multiple LLM instances for different phases
    """

    def __init__(self,
                 prompts_dict: Dict[str, Any],
                 question_improvement_llm: Optional[Any] = None,
                 score_threshold: int = 50):
        """
        Initialize hybrid formatter.

        Args:
            prompts_dict: Dictionary containing prompt templates
            question_improvement_llm: LLM for question preprocessing
            score_threshold: Minimum score threshold for filtering results
        """
        super().__init__(prompts_dict)
        self.question_improvement_llm = question_improvement_llm
        self.score_threshold = score_threshold

    def invoke_llm_map(self, chunk: Any, question: str) -> Dict[str, Any]:
        """
        Map phase using direct LLM invocation for text output.

        Args:
            chunk: Document chunk with page_content attribute
            question: The question to answer

        Returns:
            Dictionary with 'content' and 'usage' keys
        """
        prompt = self.prompts_dict['map_prompt'].format(
            context=chunk.page_content,
            question_int=question
        )

        response = self.map_llm.invoke(prompt)

        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            return {
                'content': response.content,
                'usage': response.usage_metadata
            }
        return {'content': response.content, 'usage': None}

    def invoke_llm_reduce(self, formatted_results: Any, question: str) -> Any:
        """
        Reduce phase using GPT wrapper for JSON output.

        Args:
            formatted_results: Formatted string of map results
            question: The original question

        Returns:
            Dictionary with 'json' and 'raw_response' keys
        """
        reduce_prompt = self.prompts_dict['reduce_prompt']
        return self.reduce_llm.invoke(reduce_prompt, context=formatted_results, question=question)

    def preprocess_map_results(self, results: List[Dict[str, Any]]) -> List[str]:
        """
        Filter based on score extraction from text.

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
        Parse JSON result from reduce phase.

        Expects JSON with keys: "reasoning" and "answer"

        Args:
            result: Result from reduce phase (should have 'json' key)

        Returns:
            Dictionary with llm_answer, llm_reasoning, and llm_evidence
        """
        reduce_json = result.get('json', {})
        if reduce_json:
            return {
                "llm_answer": reduce_json.get("answer", ""),
                "llm_reasoning": reduce_json.get("reasoning", ""),
                "llm_evidence": []
            }
        else:
            raw_response = result.get('raw_response')
            if raw_response:
                answer = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
            else:
                answer = str(result)

            return {
                "llm_answer": answer,
                "llm_reasoning": "Score-based filtering applied to map results",
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
        reduce_json = reduce_result.get('json', {})
        if reduce_json:
            return {
                "llm_answer": reduce_json.get("answer", ""),
                "llm_reasoning": reduce_json.get("reasoning", ""),
                "llm_evidence": map_results
            }
        else:
            raw_response = reduce_result.get('raw_response')
            if raw_response:
                answer = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
            else:
                answer = str(reduce_result)

            return {
                "llm_answer": answer,
                "llm_reasoning": "Score-based filtering applied to map results",
                "llm_evidence": map_results
            }

    def get_evaluation_formatter_type(self) -> Optional[str]:
        """Use last year evaluation formatter for hybrid."""
        return "last_year"

    def improve_question(self, original_question: str) -> Tuple[str, Dict[str, int]]:
        """
        Improve a single question using the question improvement LLM.

        Args:
            original_question: The original question text

        Returns:
            Tuple of (improved question text, token usage dict)
        """
        empty_tokens = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0}

        if 'question_improvement_prompt' not in self.prompts_dict:
            print("Warning: question_improvement_prompt not available, keeping original questions")
            return original_question, empty_tokens

        try:
            prompt = self.prompts_dict['question_improvement_prompt']
            if self.question_improvement_llm:
                response = self.question_improvement_llm.invoke(prompt, question=original_question)
            else:
                response = self.reduce_llm.invoke(prompt, question=original_question)

            tokens = self._extract_token_usage_from_response(response)

            if isinstance(response, dict) and 'json' in response:
                json_data = response['json']
                if isinstance(json_data, dict) and 'improved_question' in json_data:
                    improved = json_data['improved_question'].strip()
                    if improved:
                        return improved, tokens

            print(f"Warning: Could not parse improved question, using original")
            return original_question, tokens

        except Exception as e:
            print(f"Error improving question, using original: {e}")
            return original_question, empty_tokens

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

    def add_format_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Add hybrid formatter configuration."""
        config = super().add_format_config(config)
        config["approach"] = "Hybrid MapReduce (Text Map + JSON Reduce)"
        config["score_threshold"] = self.score_threshold

        if self.question_improvement_llm and hasattr(self.question_improvement_llm, 'get_model_name'):
            config["question_improvement_llm"] = {
                "model_name": self.question_improvement_llm.get_model_name(),
                "temperature": self.question_improvement_llm.get_temperature() if hasattr(self.question_improvement_llm, 'get_temperature') else None,
                "max_tokens": self.question_improvement_llm.get_max_tokens() if hasattr(self.question_improvement_llm, 'get_max_tokens') else None,
                "provider": self.question_improvement_llm.get_provider() if hasattr(self.question_improvement_llm, 'get_provider') else None,
                "key": self.question_improvement_llm.get_key() if hasattr(self.question_improvement_llm, 'get_key') else None
            }

        return config