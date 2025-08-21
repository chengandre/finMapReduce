import sys
import os
import tempfile
from typing import Dict, List, Any, Tuple, Optional, Union
import concurrent.futures
from pathlib import Path

# Add parent directory to path to import existing modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from base_mapreduce_qa import BaseMapReduceQA
from json_formatter import JSONFormatter
from hybrid_formatter import HybridFormatter
from plain_text_formatter import PlainTextFormatter
from utils import load_document_chunk
import time


class SingleDocPipeline(BaseMapReduceQA):
    """
    MapReduce pipeline for single document processing in the webapp.

    This class adapts the existing BaseMapReduceQA architecture for web requests,
    handling uploaded files and returning structured results without persistence.
    """

    def __init__(self,
                 llm: Any,
                 prompts_dict: Dict[str, Any],
                 format_type: str = "hybrid",
                 pdf_parser: str = "marker",
                 max_file_size: int = 50 * 1024 * 1024,  # 50MB
                 **kwargs):
        """
        Initialize the SingleDocPipeline.

        Args:
            llm: Primary language model instance
            prompts_dict: Dictionary containing prompt templates
            format_type: Output format type ("json", "hybrid", "plain")
            pdf_parser: PDF parsing method ("marker", "pypdf", etc.)
            max_file_size: Maximum file size in bytes
            **kwargs: Additional arguments passed to BaseMapReduceQA
        """
        super().__init__(llm=llm, prompts_dict=prompts_dict, **kwargs)
        self.format_type = format_type
        self.pdf_parser = pdf_parser
        self.max_file_size = max_file_size

        # Create output formatter
        self.output_formatter = self._create_formatter(format_type, prompts_dict)
        self.output_formatter.set_llms(self.map_llm, self.reduce_llm)

    def _create_formatter(self, format_type: str, prompts_dict: Dict[str, Any]):
        """Create appropriate output formatter based on format type."""
        if format_type == "json":
            return JSONFormatter(prompts_dict)
        elif format_type == "hybrid":
            return HybridFormatter(prompts_dict)
        elif format_type == "plain":
            return PlainTextFormatter(prompts_dict)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

    def load_data(self, data_path: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Not used in webapp - raise NotImplementedError."""
        raise NotImplementedError("SingleDocPipeline does not support dataset loading. Use process_uploaded_file instead.")

    def load_document_chunks(self, qa_pair: Dict[str, Any]) -> Tuple[List[Any], int]:
        """
        Load and chunk an uploaded document.

        Handles both PDF and text files with size limits and cleanup.
        """
        file_path = qa_pair.get("file_path")
        if not file_path or not os.path.exists(file_path):
            raise ValueError("Invalid file path provided")

        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > self.max_file_size:
            raise ValueError(f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds maximum allowed size ({self.max_file_size / 1024 / 1024:.1f}MB)")

        # Determine file type
        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.pdf':
            # Use existing PDF processing
            documents, token_count = load_document_chunk(
                file_path,
                self.chunk_size,
                self.chunk_overlap,
                method=self.pdf_parser
            )
        elif file_ext in ['.txt', '.md']:
            # Simple text chunking for non-PDF files
            documents, token_count = self._chunk_text_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}. Only PDF and TXT files are supported.")

        if documents is None or token_count is None:
            return [], 0

        return documents, token_count

    def _chunk_text_file(self, file_path: str) -> Tuple[List[Any], int]:
        """
        Simple text file chunking with overlap.

        Returns chunks compatible with the existing pipeline.
        """
        from langchain.schema import Document
        import tiktoken

        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except:
            # Fallback to character-based chunking
            encoding = None

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        if encoding:
            # Token-based chunking
            tokens = encoding.encode(text)
            total_tokens = len(tokens)

            chunks = []
            start = 0
            while start < len(tokens):
                end = min(start + self.chunk_size, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = encoding.decode(chunk_tokens)

                chunks.append(Document(page_content=chunk_text))

                # Move start position with overlap
                if end < len(tokens):
                    start = end - self.chunk_overlap
                else:
                    break
        else:
            # Character-based chunking fallback
            total_tokens = len(text) // 4  # Rough estimation

            chunks = []
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size * 4, len(text))  # Rough char equivalent
                chunk_text = text[start:end]

                chunks.append(Document(page_content=chunk_text))

                # Move start position with overlap
                if end < len(text):
                    start = end - self.chunk_overlap * 4
                else:
                    break

        return chunks, total_tokens

    def process_uploaded_file(self, file_path: str, question: str, cleanup: bool = True) -> Dict[str, Any]:
        """
        Process an uploaded file with a question.

        Args:
            file_path: Path to the uploaded file
            question: User question
            cleanup: Whether to clean up temp files after processing

        Returns:
            Dictionary with answer, reasoning, evidence, and metadata
        """
        try:
            # Create qa_pair structure
            qa_pair = {
                "file_path": file_path,
                "question": question,
                "doc_name": os.path.basename(file_path)
            }

            # Process through MapReduce pipeline
            result = self.process_single_qa(qa_pair)

            # Map fields for webapp response
            webapp_result = {
                "answer": result.get("llm_answer", ""),
                "reasoning": result.get("llm_reasoning", ""),
                "evidence": result.get("llm_evidence", []),
                "token_stats": result.get("token_stats", {}),
                "timing_stats": result.get("timing_stats", {}),
                "chunk_stats": result.get("chunk_stats", {}),
                "request_id": f"req_{int(time.time())}_{hash(question) % 10000}"
            }
            # print(webapp_result)
            # print(type(webapp_result["evidence"]))
            # print("Evidence content:", webapp_result["evidence"])
            return webapp_result

        finally:
            # Clean up temp file if requested
            if cleanup and os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except Exception as e:
                    print(f"Warning: Could not clean up temp file {file_path}: {e}")

    # Delegate formatting methods to output_formatter
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
        """Delegate to output formatter with map data support."""
        if hasattr(self.output_formatter, 'parse_final_result_with_map_data'):
            return self.output_formatter.parse_final_result_with_map_data(reduce_result, map_results)
        else:
            return self.output_formatter.parse_final_result(reduce_result)

    def invoke_llm_map(self, chunk: Any, question: str) -> Dict[str, Any]:
        """Delegate to output formatter."""
        return self.output_formatter.invoke_llm_map(chunk, question)

    def invoke_llm_reduce(self, formatted_results: Any, question: str) -> Any:
        """Delegate to output formatter."""
        return self.output_formatter.invoke_llm_reduce(formatted_results, question)

    # Override methods that are not needed for webapp
    def get_results_directory(self) -> str:
        """Not used in webapp."""
        return ""

    def get_dataset_name(self) -> str:
        """Return webapp identifier."""
        return "webapp"