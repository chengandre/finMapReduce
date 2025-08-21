from typing import Dict, List, Any, Union, Optional
import asyncio
from output_formatter import OutputFormatter


class JSONFormatter(OutputFormatter):
    """
    Output formatter for JSON-based map/reduce operations.

    Features:
    - GPT wrapper with JSON parsing
    - Relevance score filtering from JSON
    - XML formatting for reduce phase
    - Structured JSON result parsing
    """

    def preprocess_map_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter map results based on relevance score from JSON.

        Only keeps results with relevance_score > 5.

        Args:
            results: List of map phase results

        Returns:
            Filtered list of results
        """
        modified_results = []
        for result in results:
            score = result.get("json", {}).get("relevance_score", 0)
            if score > 5:
                modified_results.append(result)
        return modified_results

    def format_map_results_for_reduce(self, results: List[Dict[str, Any]], question: str) -> str:
        """
        Format map results as XML for the reduce phase.

        Args:
            results: List of filtered map results
            question: The original question (unused)

        Returns:
            XML-formatted string of results
        """
        processed_results = []
        for i, result in enumerate(results, 1):
            chunk_xml = self._format_single_result_as_xml(result, i)
            if chunk_xml:
                processed_results.append(chunk_xml)

        return "\n".join(processed_results)

    def _format_single_result_as_xml(self, result: Dict[str, Any], index: int) -> str:
        """
        Format a single result as XML chunk.

        Args:
            result: Single map result
            index: Chunk index number

        Returns:
            XML string for the chunk
        """
        result_json = result.get('json', {})
        if result_json:
            chunk_xml = f"      <chunk_{index}>\n"
            chunk_xml += f"        <summary>{self._escape_xml(result_json.get('summary', ''))}</summary>\n"
            chunk_xml += f"        <terms>{self._escape_xml(str(result_json.get('terms', [])))}</terms>\n"
            chunk_xml += f"        <evidence>{self._escape_xml(str(result_json.get('evidence', [])))}</evidence>\n"
            chunk_xml += f"        <answer>{self._escape_xml(result_json.get('answer', ''))}</answer>\n"
            chunk_xml += f"        <relevance_score>{result_json.get('relevance_score', 0)}</relevance_score>\n"
            chunk_xml += f"      </chunk_{index}>"
            return chunk_xml
        else:
            raw_response = result.get('raw_response')
            if raw_response:
                content = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
                chunk_xml = f"      <chunk_{index}>\n"
                chunk_xml += f"        <summary>Raw response content</summary>\n"
                chunk_xml += f"        <terms>[]</terms>\n"
                chunk_xml += f"        <evidence>[\"{self._escape_xml(content)}\"]</evidence>\n"
                chunk_xml += f"        <answer>{self._escape_xml(content)}</answer>\n"
                chunk_xml += f"        <relevance_score>0</relevance_score>\n"
                chunk_xml += f"      </chunk_{index}>"
                return chunk_xml
        return ""

    def _escape_xml(self, text: str) -> str:
        """Escape special XML characters."""
        if not isinstance(text, str):
            text = str(text)
        return (text.replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;")
                   .replace("'", "&apos;")
                   .replace('"', "&quot;"))

    def parse_final_result(self, result: Any) -> Dict[str, Any]:
        """
        Parse JSON result from reduce phase.

        Args:
            result: Result from reduce phase (should have 'json' key)

        Returns:
            Dictionary with llm_answer, llm_reasoning, and llm_evidence
        """
        reduce_json = result.get('json', {})
        if reduce_json:
            return {
                "llm_answer": reduce_json.get("answer", ""),
                "llm_reasoning": reduce_json.get("reasoning", "No reasoning provided"),
                "llm_evidence": reduce_json.get("evidence", [])
            }
        else:
            raw_response = result.get('raw_response')
            if raw_response:
                answer = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
            else:
                answer = str(result)

            return {
                "llm_answer": answer,
                "llm_reasoning": "No reasoning provided",
                "llm_evidence": []
            }

    def add_format_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Add JSON formatter configuration."""
        config = super().add_format_config(config)
        config["approach"] = "JSON MapReduce"
        return config

    # Async method overrides for better performance
    async def invoke_llm_map_async(self, chunk: Any, question: str) -> Dict[str, Any]:
        """
        Async version of invoke_llm_map.

        Args:
            chunk: Document chunk with page_content attribute
            question: The question to answer

        Returns:
            Dictionary with 'json' and 'raw_response' keys
        """
        map_prompt = self.prompts_dict['map_prompt']

        return await self.map_llm.invoke(map_prompt, context=chunk.page_content, final_query=question)

    async def invoke_llm_reduce_async(self, formatted_results: Any, question: str) -> Any:
        """
        Async version of invoke_llm_reduce.

        Args:
            formatted_results: XML-formatted string of map results
            question: The original question

        Returns:
            Dictionary with 'json' and 'raw_response' keys
        """
        reduce_prompt = self.prompts_dict['reduce_prompt']

        return await self.reduce_llm.invoke(reduce_prompt, map_results=formatted_results, final_query=question)