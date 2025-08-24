import os
import sys
import tempfile
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.loaders.dataset_loader import DatasetLoader
from src.utils.document_processing import load_document_chunk


class WebappDatasetLoader(DatasetLoader):
    """
    Dataset loader for webapp uploaded files.

    Handles single uploaded files for question answering instead of batch datasets.
    Supports both PDF and text file uploads with proper document processing.
    """

    def __init__(self, pdf_parser: str = "marker", max_file_size: int = 50 * 1024 * 1024):
        """
        Initialize the WebappDatasetLoader.

        Args:
            pdf_parser: PDF parsing method ("marker", "pypdf", etc.)
            max_file_size: Maximum file size in bytes (default: 50MB)
        """
        self.pdf_parser = pdf_parser
        self.max_file_size = max_file_size

    def load_data(self, data_path: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        For webapp, this creates a single QA pair from the uploaded file.

        Args:
            data_path: Path to uploaded file
            num_samples: Ignored for webapp (always returns 1 item)

        Returns:
            List with single QA pair containing file_path and placeholder question
        """
        if not os.path.exists(data_path):
            raise ValueError(f"File not found: {data_path}")

        # Check file size
        file_size = os.path.getsize(data_path)
        if file_size > self.max_file_size:
            raise ValueError(f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds maximum allowed size ({self.max_file_size / 1024 / 1024:.1f}MB)")

        # Create a single QA pair for the uploaded file
        qa_pair = {
            "file_path": data_path,
            "doc_name": os.path.basename(data_path),
            "question": "",  # Will be set by the API endpoint
            "answer": "",   # Not used for webapp processing
            "evidence": []  # Not used for webapp processing
        }

        return [qa_pair]

    def load_document_chunks(self, qa_pair: Dict[str, Any], chunk_size: int, chunk_overlap: int) -> Tuple[List[Any], int]:
        """
        Load and chunk an uploaded document.

        Args:
            qa_pair: Dictionary containing file_path
            chunk_size: Size of document chunks for processing
            chunk_overlap: Overlap between consecutive chunks

        Returns:
            Tuple of (list of document chunks, total token count)
        """
        file_path = qa_pair.get("file_path")
        if not file_path or not os.path.exists(file_path):
            raise ValueError("Invalid file path provided")

        # Determine file type
        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.pdf':
            # Use existing PDF processing
            documents, token_count = load_document_chunk(
                file_path,
                chunk_size,
                chunk_overlap,
                method=self.pdf_parser
            )
        elif file_ext in ['.txt', '.md']:
            # Simple text chunking for non-PDF files
            documents, token_count = self._chunk_text_file(file_path, chunk_size, chunk_overlap)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}. Only PDF and TXT files are supported.")

        if documents is None or token_count is None:
            return [], 0

        return documents, token_count

    def load_full_document(self, qa_pair: Dict[str, Any]) -> Tuple[str, int]:
        """
        Load the full document content for truncation approaches.

        Args:
            qa_pair: Dictionary containing file_path

        Returns:
            Tuple of (full document text, total token count)
        """
        file_path = qa_pair.get("file_path")
        if not file_path or not os.path.exists(file_path):
            raise ValueError("Invalid file path provided")

        # Determine file type
        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.pdf':
            # Use existing PDF processing to get full document
            # Load with very large chunk size to get the whole document
            documents, token_count = load_document_chunk(
                file_path,
                chunk_size=1000000,  # Very large chunk to get full document
                chunk_overlap=0,
                method=self.pdf_parser
            )

            if documents:
                # Combine all chunks into single text
                full_text = "\n\n".join([doc.page_content for doc in documents])
                return full_text, token_count
            else:
                return "", 0

        elif file_ext in ['.txt', '.md']:
            # Read text file directly
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Estimate token count (rough approximation)
            token_count = len(text) // 4

            return text, token_count
        else:
            raise ValueError(f"Unsupported file type: {file_ext}. Only PDF and TXT files are supported.")

    def _chunk_text_file(self, file_path: str, chunk_size: int, chunk_overlap: int) -> Tuple[List[Any], int]:
        """
        Simple text file chunking with overlap.

        Args:
            file_path: Path to text file
            chunk_size: Size of chunks in tokens
            chunk_overlap: Overlap between chunks in tokens

        Returns:
            Tuple of (list of document chunks, total token count)
        """
        from langchain.schema import Document

        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
        except ImportError:
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
                end = min(start + chunk_size, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = encoding.decode(chunk_tokens)

                chunks.append(Document(page_content=chunk_text))

                # Move start position with overlap
                if end < len(tokens):
                    start = end - chunk_overlap
                else:
                    break
        else:
            # Character-based chunking fallback
            total_tokens = len(text) // 4  # Rough estimation

            chunks = []
            start = 0
            char_chunk_size = chunk_size * 4  # Rough char equivalent
            char_overlap = chunk_overlap * 4

            while start < len(text):
                end = min(start + char_chunk_size, len(text))
                chunk_text = text[start:end]

                chunks.append(Document(page_content=chunk_text))

                # Move start position with overlap
                if end < len(text):
                    start = end - char_overlap
                else:
                    break

        return chunks, total_tokens

    def get_document_identifier(self, qa_pair: Dict[str, Any]) -> str:
        """Get document identifier from qa_pair."""
        return qa_pair.get("doc_name", qa_pair.get("file_path", "unknown_doc"))

    def get_results_directory(self) -> str:
        """Get the directory name for saving results."""
        return "webapp_results"

    def get_dataset_name(self) -> str:
        """Get the name of the dataset."""
        return "webapp"

    def add_dataset_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Add webapp-specific configuration."""
        config = super().add_dataset_config(config)
        config["pdf_parser"] = self.pdf_parser
        config["max_file_size"] = self.max_file_size
        return config