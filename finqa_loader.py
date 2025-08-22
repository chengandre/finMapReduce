import json
import os
from typing import Dict, List, Any, Tuple, Optional
from dataset_loader import DatasetLoader
from utils import load_document_chunk


class FinQALoader(DatasetLoader):
    """
    Dataset loader for FinQA data.

    Handles:
    - Loading JSON format data
    - Markdown document processing
    - FinQA-specific paths and configurations
    """

    def __init__(self, doc_dir: str):
        """
        Initialize FinQA loader.

        Args:
            doc_dir: Directory containing FinQA markdown documents
        """
        self.doc_dir = doc_dir

    def load_data(self, data_path: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load FinQA data from JSON file.

        Args:
            data_path: Path to the JSON file
            num_samples: Number of samples to load (None for all)

        Returns:
            List of QA dictionaries
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        def transform_item(item):
            return {
                "doc_name": item["doc_name"],
                "question": item["question"],
                "answer": item["answer"],
                "filename": item.get("filename", ""),
                "explanation": item.get("explanation", "")
            }

        return self._process_data_samples(data, num_samples, transform_item)

    def load_document_chunks(self, qa_pair: Dict[str, Any], chunk_size: int, chunk_overlap: int) -> Tuple[List[Any], int]:
        """
        Load and chunk Markdown document.

        Args:
            qa_pair: Dictionary containing 'doc_name' key
            chunk_size: Size of document chunks for processing
            chunk_overlap: Overlap between consecutive chunks

        Returns:
            Tuple of (list of document chunks, total token count)
        """
        doc_name = qa_pair["doc_name"]
        markdown_file = os.path.join(self.doc_dir, doc_name)
        documents, token_count = load_document_chunk(markdown_file, chunk_size, chunk_overlap, method="markdown")

        # Ensure we return proper types as expected by the interface
        if documents is None:
            documents = []
        if token_count is None:
            token_count = 0

        return documents, token_count

    def load_full_document(self, qa_pair: Dict[str, Any]) -> Tuple[str, int]:
        """
        Load the full document content for a QA pair (for truncation approaches).

        Args:
            qa_pair: Dictionary containing 'doc_name' key

        Returns:
            Tuple of (full document text, total token count)
        """
        from utils import num_tokens_from_string

        doc_name = qa_pair["doc_name"]
        markdown_file = os.path.join(self.doc_dir, doc_name)

        try:
            # Read the full markdown file
            with open(markdown_file, 'r', encoding='utf-8') as f:
                full_text = f.read()

            # Calculate token count
            token_count = num_tokens_from_string(full_text, "cl100k_base")

            return full_text, token_count

        except Exception as e:
            print(f"Error loading full document for {doc_name}: {e}")
            return "", 0

    def get_document_identifier(self, qa_pair: Dict[str, Any]) -> str:
        """Get document name for display."""
        return qa_pair.get("doc_name", "unknown")

    def get_results_directory(self) -> str:
        """Get directory for saving results."""
        return "finqa_results"

    def get_dataset_name(self) -> str:
        """Get dataset name."""
        return "finqa"

    def add_dataset_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Add FinQA-specific configuration."""
        config = super().add_dataset_config(config)
        config["pdf_parser"] = "markdown"
        config["doc_dir"] = self.doc_dir
        return config