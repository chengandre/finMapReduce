"""
FinanceBench-specific truncation implementation.

Handles FinanceBench dataset format with PDF document processing
and single-pass question answering using document truncation.
"""

import json
import sys
import os
from typing import Dict, List, Any, Tuple, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_truncation_qa import BaseTruncationQA


class FinanceBenchTruncation(BaseTruncationQA):
    """
    FinanceBench-specific implementation of truncation-based QA.
    
    This implementation:
    - Loads data from FinanceBench JSONL format
    - Processes PDF documents using existing utilities
    - Uses single LLM call with truncated document content
    - Supports various truncation strategies (start, end, smart)
    """

    def __init__(self,
                 llm: Any,
                 prompts_dict: Dict[str, Any],
                 pdf_parser: str = "marker",
                 **kwargs):
        """
        Initialize FinanceBench truncation pipeline.

        Args:
            llm: Language model instance
            prompts_dict: Dictionary containing prompt templates
            pdf_parser: PDF parsing method ("marker", "pypdf", etc.)
            **kwargs: Additional arguments passed to BaseTruncationQA
        """
        super().__init__(llm, prompts_dict, **kwargs)
        self.pdf_parser = pdf_parser

    def load_data(self, data_path: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load FinanceBench data from JSONL file.

        Args:
            data_path: Path to FinanceBench JSONL file
            num_samples: Number of samples to load (None for all)

        Returns:
            List of QA pairs
        """
        qa_data = []
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if num_samples and len(qa_data) >= num_samples:
                        break
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        qa_item = json.loads(line)
                        qa_data.append(qa_item)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed JSON on line {line_num}: {e}")
                        continue
        
        except FileNotFoundError:
            raise FileNotFoundError(f"FinanceBench data file not found: {data_path}")
        except Exception as e:
            raise Exception(f"Error loading FinanceBench data: {e}")

        print(f"Loaded {len(qa_data)} QA pairs from FinanceBench dataset")
        return qa_data

    def load_full_document(self, qa_pair: Dict[str, Any]) -> Tuple[str, int]:
        """
        Load the full document content for a QA pair.

        Args:
            qa_pair: Dictionary containing 'doc_name' key

        Returns:
            Tuple of (document_text, original_token_count)
        """
        from utils import load_document_chunk, num_tokens_from_string

        doc_name = qa_pair.get("doc_name")
        if not doc_name:
            raise ValueError("QA pair missing 'doc_name' field")

        # Load full document without chunking (use very large chunk size)
        try:
            documents, _ = load_document_chunk(
                doc_name,
                chunk_size=10000000,  # Very large to get full document
                chunk_overlap=0,
                method=self.pdf_parser
            )
            
            if not documents:
                raise ValueError(f"No document content loaded for {doc_name}")
            
            # Combine all chunks into single text (should typically be just one chunk)
            full_text = "\n\n".join([doc.page_content if hasattr(doc, 'page_content') else str(doc) 
                                   for doc in documents])
            
            # Count tokens in full document
            token_count = num_tokens_from_string(full_text, "cl100k_base")
            
            return full_text, token_count
            
        except Exception as e:
            raise Exception(f"Failed to load document {doc_name}: {e}")

    def invoke_llm_direct(self, document_text: str, question: str) -> Any:
        """
        Invoke LLM directly with truncated document and question.

        Args:
            document_text: Truncated document content
            question: Question to answer

        Returns:
            LLM response
        """
        # Get the prompt template
        prompt_key = self.get_prompt_key()
        if prompt_key not in self.prompts_dict:
            # Fallback to reduce prompt if truncation prompt not available
            prompt_key = 'map_prompt'
            if prompt_key not in self.prompts_dict:
                raise ValueError(f"No suitable prompt found. Need '{self.get_prompt_key()}' or 'reduce_prompt'")

        prompt_template = self.prompts_dict[prompt_key]

        try:
            # Use the LLM wrapper to invoke with the prompt
            if hasattr(prompt_template, 'format'):
                # LangChain PromptTemplate
                formatted_prompt = prompt_template.format(
                    context=document_text,
                    question=question
                )
                # Use direct string invocation for truncation
                response = self.llm.invoke(formatted_prompt)
            else:
                # String template
                if hasattr(self.llm, 'invoke'):
                    response = self.llm.invoke(prompt_template,
                                            context=document_text,
                                            question=question)
                else:
                    # Fallback for basic LLM interface
                    formatted_prompt = prompt_template.format(
                        context=document_text,
                        question=question
                    )
                    response = self.llm.invoke(formatted_prompt)

            return response

        except Exception as e:
            print(f"Error in LLM invocation: {e}")
            raise

    def parse_result(self, llm_result: Any) -> Dict[str, Any]:
        """
        Parse the LLM result into standardized format.

        Args:
            llm_result: Raw LLM response

        Returns:
            Dictionary with llm_answer, llm_reasoning, llm_evidence
        """
        try:
            # Handle LangChain AIMessage format
            if hasattr(llm_result, 'content'):
                content = llm_result.content
                return self._parse_text_response(content)

            # Handle GPT wrapper response format
            if isinstance(llm_result, dict):
                # Check for JSON response first
                if 'json' in llm_result and isinstance(llm_result['json'], dict):
                    json_data = llm_result['json']
                    return {
                        "llm_answer": json_data.get("answer", "No answer provided"),
                        "llm_reasoning": json_data.get("reasoning", "No reasoning provided"),
                        "llm_evidence": json_data.get("evidence", [])
                    }
                # Check for content field
                elif 'content' in llm_result:
                    content = llm_result['content']
                    return self._parse_text_response(content)
                # Direct dict response
                else:
                    return {
                        "llm_answer": llm_result.get("answer", str(llm_result)),
                        "llm_reasoning": llm_result.get("reasoning", "Direct response"),
                        "llm_evidence": llm_result.get("evidence", [])
                    }
            
            # Handle string response
            elif isinstance(llm_result, str):
                return self._parse_text_response(llm_result)
            
            # Handle other response types
            else:
                content = str(llm_result)
                return self._parse_text_response(content)

        except Exception as e:
            print(f"Error parsing LLM result: {e}")
            return {
                "llm_answer": "Error parsing response",
                "llm_reasoning": f"Parsing failed: {str(e)}",
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

    # ===== Template Method Overrides =====

    def get_dataset_name(self) -> str:
        """Get the name of the dataset."""
        return "financebench"

    def get_results_directory(self) -> str:
        """Get the directory name for saving results."""
        return "truncation_results"

    def get_evaluation_formatter_type(self) -> Optional[str]:
        """Get the evaluation formatter type for this pipeline."""
        return "standard"  # Use standard formatter for FinanceBench

    def get_prompt_key(self) -> str:
        """Get the key for the main prompt in prompts_dict."""
        # Try truncation-specific prompt first, fallback to reduce_prompt
        if 'truncation_prompt' in self.prompts_dict:
            return 'map_prompt'
        else:
            return 'map_prompt'

    def get_document_identifier(self, qa_pair: Dict[str, Any]) -> str:
        """Get a display name for the document being processed."""
        return qa_pair.get("doc_name", "unknown_document")