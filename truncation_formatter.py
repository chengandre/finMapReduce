from typing import Dict, Any, Optional, Tuple, List
import asyncio


class TruncationFormatter:
    """
    Formatter for truncation-based QA pipelines.

    This class handles document truncation, LLM invocation, and result parsing
    for truncation-based approaches.
    """

    def __init__(self,
                 prompts_dict: Dict[str, Any],
                 strategy: str = "start",
                 context_window: int = 128000,
                 buffer: int = 2000,
                 max_document_tokens: Optional[int] = None):
        """
        Initialize the truncation formatter.

        Args:
            prompts_dict: Dictionary containing prompt templates
            strategy: Truncation strategy ("start", "end", "smart")
            context_window: Maximum context window size for the model
            buffer: Safety buffer for response tokens
            max_document_tokens: Override for max document tokens (None = auto-calculate)
        """
        self.prompts_dict = prompts_dict
        self.strategy = strategy
        self.context_window = context_window
        self.buffer = buffer
        self.max_document_tokens = max_document_tokens
        self.llm = None  # Set via set_llm()

    def set_llm(self, llm: Any):
        """
        Set the LLM instance for this formatter.

        Args:
            llm: LLM instance for truncation calls
        """
        self.llm = llm

    def truncate_document(self, doc_text: str, question: str) -> Tuple[str, Dict[str, Any]]:
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
            prompt_tokens = num_tokens_from_string(question, "cl100k_base") + self.buffer
            max_doc_tokens = self.context_window - prompt_tokens

        # Create truncation manager and truncate
        manager = TruncationManager(
            strategy=self.strategy,
            max_tokens=max_doc_tokens
        )

        return manager.truncate_document(doc_text)

    async def invoke_llm_truncation_async(self, document_text: str, question: str) -> Any:
        """
        Invoke LLM with truncated document and question.

        Args:
            document_text: Truncated document content
            question: Question to answer

        Returns:
            LLM response
        """
        if self.llm is None:
            raise ValueError("LLM instance not set. Call set_llm() first.")

        # Use direct prompt template for truncation approach
        prompt_key = self.get_prompt_key()
        prompt_template = self.prompts_dict[prompt_key]

        # Format prompt with document and question
        formatted_prompt = prompt_template.format(context=document_text, question=question)

        # Invoke LLM
        return await self.llm.invoke(formatted_prompt)

    def parse_final_result(self, llm_result: Any) -> Dict[str, Any]:
        """
        Parse the LLM result into standardized format.

        Args:
            llm_result: Raw LLM response

        Returns:
            Dictionary with llm_answer, llm_reasoning, llm_evidence
        """
        # Handle different response formats
        if isinstance(llm_result, dict):
            # Check if it's a JSON response
            if 'json' in llm_result:
                json_data = llm_result['json']
                if isinstance(json_data, dict):
                    return {
                        "llm_answer": json_data.get("answer", ""),
                        "llm_reasoning": json_data.get("reasoning", ""),
                        "llm_evidence": json_data.get("evidence", [])
                    }

            # Check if it's a content response
            if 'content' in llm_result:
                content = llm_result['content']
                return self._parse_text_response(content)

        # Handle string responses
        if isinstance(llm_result, str):
            return self._parse_text_response(llm_result)

        # Handle LangChain-style responses
        if hasattr(llm_result, 'content'):
            return self._parse_text_response(llm_result.content)

        # Fallback
        return {
            "llm_answer": str(llm_result),
            "llm_reasoning": "No reasoning provided",
            "llm_evidence": []
        }

    def _parse_text_response(self, text: str) -> Dict[str, Any]:
        """
        Parse a text response to extract answer, reasoning, and evidence.

        Args:
            text: Raw text response

        Returns:
            Dictionary with parsed components
        """
        # Try to parse JSON if text looks like JSON
        text = text.strip()
        if text.startswith('{') and text.endswith('}'):
            try:
                import json5
                parsed = json5.loads(text)
                if isinstance(parsed, dict):
                    return {
                        "llm_answer": parsed.get("answer", text),
                        "llm_reasoning": parsed.get("reasoning", "JSON response"),
                        "llm_evidence": parsed.get("evidence", [])
                    }
            except Exception:
                pass  # Fall back to text parsing

        # Simple text parsing - use entire response as answer
        return {
            "llm_answer": text,
            "llm_reasoning": "Single-pass truncation response",
            "llm_evidence": []
        }

    def get_judge_prompt_key(self) -> str:
        """
        Get the key for judge prompt in prompts_dict.

        Returns:
            Key string for judge prompt
        """
        return 'judge_prompt'

    def get_evaluation_formatter_type(self) -> Optional[str]:
        """
        Get the evaluation formatter type for truncation approach.

        Returns:
            Formatter type string or None for auto-detection
        """
        return None  # Auto-detect by default

    def get_prompt_key(self) -> str:
        """
        Get the key for the main prompt in prompts_dict.

        Returns:
            Key string for the truncation prompt
        """
        # Use map_prompt as the direct prompt for truncation approach
        return 'map_prompt'

    def add_truncation_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add truncation-specific configuration to results.

        Args:
            config: Existing configuration dictionary

        Returns:
            Updated configuration dictionary
        """
        config.update({
            "approach": "Truncation",
            "truncation_strategy": self.strategy,
            "context_window": self.context_window,
            "truncation_buffer": self.buffer,
            "max_document_tokens": self.max_document_tokens
        })
        return config


