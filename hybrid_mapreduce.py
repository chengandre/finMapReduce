import re
from typing import Dict, List, Any, Tuple, Optional
from base_mapreduce_qa import BaseMapReduceQA
from utils import load_pdf_chunk
from financebench_mapreduce import load_financebench_data


class HybridMapReduce(BaseMapReduceQA):
    """
    Hybrid MapReduce implementation combining:
    - Map phase: Text output with score-based filtering (like LastYearMapReduce)
    - Reduce phase: JSON output format (like LastYearJSONMapReduce)

    Uses prompts/map_prompt_test.yml for map phase and prompts/reduce_prompt_test.yml for reduce phase.
    """

    def __init__(self,
                 llm,
                 prompts_dict: Dict[str, Any],
                 map_llm=None,
                 reduce_llm=None,
                 question_improvement_llm=None,
                 pdf_parser: str = "marker",
                 score_threshold: int = 50,
                 **kwargs):
        """
        Initialize hybrid pipeline.

        Args:
            llm: Primary LLM instance (used as fallback and for base class compatibility)
            prompts_dict: Dictionary containing prompt templates
            map_llm: LLM for map phase (RateLimitedRetryLLM for direct invocation). If None, uses llm.
            reduce_llm: LLM for reduce phase (RateLimitedGPT for JSON parsing). If None, uses llm.
            question_improvement_llm: LLM for question preprocessing. If None, uses llm.
            pdf_parser: PDF parsing method ('marker', 'pypdf', etc.)
            score_threshold: Minimum score to keep map results (default: 50)
            **kwargs: Additional arguments for parent class
        """
        super().__init__(llm=llm, prompts_dict=prompts_dict, **kwargs)
        self.pdf_parser = pdf_parser
        self.score_threshold = score_threshold
        # Store separate LLMs for map and reduce phases
        self.map_llm = map_llm if map_llm is not None else llm
        self.reduce_llm = reduce_llm if reduce_llm is not None else llm
        self.question_improvement_llm = question_improvement_llm if question_improvement_llm is not None else llm

    def load_data(self, data_path: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load FinanceBench data for hybrid pipeline."""
        return load_financebench_data(data_path, num_samples)

    def load_document_chunks(self, qa_pair: Dict[str, Any]) -> Tuple[List[Any], int]:
        """
        Load PDF with configurable parser.

        Args:
            qa_pair: Dictionary containing 'doc_name' key

        Returns:
            Tuple of (list of document chunks, total token count)
        """
        doc_name = qa_pair["doc_name"]

        documents, token_count = load_pdf_chunk(
            doc_name,
            self.chunk_size,
            self.chunk_overlap,
            method=self.pdf_parser
        )

        # Handle the case where load_pdf_chunk returns None values
        if documents is None or token_count is None:
            return [], 0

        return documents, token_count

    def invoke_llm_map(self, chunk: Any, question: str) -> Dict[str, Any]:
        """
        Map phase using direct LLM invocation for text output (like LastYearMapReduce).

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

        # Use map_llm (RateLimitedRetryLLM) for direct invoke with retry and rate limiting
        response = self.map_llm.invoke(prompt)

        # Return in standardized format
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            return {
                'content': response.content,
                'usage': response.usage_metadata
            }
        return {'content': response.content, 'usage': None}

    def invoke_llm_reduce(self, formatted_results: Any, question: str) -> Any:
        """
        Reduce phase using GPT wrapper for JSON output (like LastYearJSONMapReduce).

        Args:
            formatted_results: Formatted string of map results
            question: The original question

        Returns:
            Dictionary with 'json' and 'raw_response' keys
        """
        reduce_prompt = self.prompts_dict['reduce_prompt']
        return self.reduce_llm(reduce_prompt, context=formatted_results, question=question)

    def preprocess_map_results(self, results: List[Dict[str, Any]]) -> List[str]:
        """
        Filter based on score extraction from text (like LastYearMapReduce).

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
            question: The original question (unused in this implementation)

        Returns:
            Concatenated string of results
        """
        return "\n".join(results)

    def parse_final_result(self, result: Any) -> Dict[str, Any]:
        """
        Parse JSON result from reduce phase (like LastYearJSONMapReduce).

        Expects JSON with 2 keys: "reasoning" and "answer"

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
            # Fallback to raw response content
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

    def parse_final_result_with_map_data(self, reduce_result: Any, map_results: List[Any]) -> Dict[str, Any]:
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
                "llm_evidence": map_results  # Include filtered map results
            }
        else:
            # Fallback to raw response content
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

    def get_document_identifier(self, qa_pair: Dict[str, Any]) -> str:
        """Get document name for display."""
        return qa_pair.get("doc_name", "unknown")

    def get_results_directory(self) -> str:
        """Get directory for saving results."""
        return "financebench_results"

    def get_dataset_name(self) -> str:
        """Get dataset name."""
        return "financebench_hybrid"

    def get_judge_prompt_key(self) -> str:
        """Get the key for judge prompt."""
        return 'judge_prompt'

    def get_evaluation_formatter_type(self) -> Optional[str]:
        """Use default evaluation formatter."""
        return "last_year"

    def get_map_question(self, qa_pair) -> str:
        """Get the map question from QA pair."""
        return qa_pair["question"]

    def improve_question(self, original_question: str) -> Tuple[str, Dict[str, int]]:
        """
        Override to use separate question improvement LLM.

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
            # Use the question improvement LLM instead of the default llm
            prompt = self.prompts_dict['question_improvement_prompt']
            response = self.question_improvement_llm(prompt, question=original_question)

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

    def _compile_results(self, qa_data: List[Dict], evaluation_results: Dict,
                        model_name: str, judge_model_name: str, process_time: float, **kwargs) -> Dict:
        """
        Override to add pdf_parser, approach, and question improvement LLM configuration.
        """
        results = super()._compile_results(qa_data, evaluation_results, model_name, judge_model_name, process_time, **kwargs)
        results["configuration"]["pdf_parser"] = self.pdf_parser
        results["configuration"]["score_threshold"] = self.score_threshold
        results["configuration"]["approach"] = "Hybrid MapReduce (Text Map + JSON Reduce)"

        # Add question improvement LLM configuration
        if hasattr(self.question_improvement_llm, 'get_model_name'):
            results["configuration"]["question_improvement_llm"] = {
                "model_name": self.question_improvement_llm.get_model_name(),
                "temperature": self.question_improvement_llm.get_temperature() if hasattr(self.question_improvement_llm, 'get_temperature') else None,
                "max_tokens": self.question_improvement_llm.get_max_tokens() if hasattr(self.question_improvement_llm, 'get_max_tokens') else None,
                "provider": self.question_improvement_llm.get_provider() if hasattr(self.question_improvement_llm, 'get_provider') else None,
                "key": self.question_improvement_llm.get_key() if hasattr(self.question_improvement_llm, 'get_key') else None
            }

        return results