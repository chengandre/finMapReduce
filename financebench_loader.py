import json
from typing import Dict, List, Any, Tuple, Optional
from dataset_loader import DatasetLoader
from utils import load_document_chunk


def load_financebench_data(jsonl_path: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load QA pairs from financebench_open_source.jsonl file

    Args:
        jsonl_path: Path to the financebench jsonl file
        num_samples: Number of samples to load

    Returns:
        List of QA dictionaries with necessary information
    """
    qa_data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    count = 0
    for line in lines:
        if num_samples is not None and count >= num_samples:
            break

        item = json.loads(line)

        qa_pair = {
            "doc_name": item["doc_name"],
            "question": item["question"],
            "answer": item["answer"],
            "justification": item["justification"] if item["justification"] else "No justification provided",
            "evidence": [ev["evidence_text"] for ev in item["evidence"]],
            "question_type": item["question_type"],
            "question_reasoning": item["question_reasoning"]
        }

        qa_data.append(qa_pair)
        count += 1

    return qa_data


class FinanceBenchLoader(DatasetLoader):
    """
    Dataset loader for FinanceBench data.

    Handles:
    - Loading JSONL format data
    - PDF document processing with configurable parsers
    - FinanceBench-specific paths and configurations
    """

    def __init__(self, pdf_parser: str = "marker"):
        """
        Initialize FinanceBench loader.

        Args:
            pdf_parser: PDF parsing method ('marker', 'pypdf', 'pymu', 'unstructured', 'default')
        """
        self.pdf_parser = pdf_parser

    def load_data(self, data_path: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load FinanceBench data from JSONL file.

        Args:
            data_path: Path to the JSONL file
            num_samples: Number of samples to load (None for all)

        Returns:
            List of QA dictionaries with question, answer, evidence, etc.
        """
        return load_financebench_data(data_path, num_samples)

    def load_document_chunks(self, qa_pair: Dict[str, Any], chunk_size: int, chunk_overlap: int) -> Tuple[List[Any], int]:
        """
        Load and chunk PDF document.

        Args:
            qa_pair: Dictionary containing 'doc_name' key
            chunk_size: Size of document chunks for processing
            chunk_overlap: Overlap between consecutive chunks

        Returns:
            Tuple of (list of document chunks, total token count)
        """
        doc_name = qa_pair["doc_name"]
        documents, token_count = load_document_chunk(
            doc_name,
            chunk_size,
            chunk_overlap,
            method=self.pdf_parser
        )

        if documents is None or token_count is None:
            return [], 0

        return documents, token_count

    def get_document_identifier(self, qa_pair: Dict[str, Any]) -> str:
        """Get document name for display."""
        return qa_pair.get("doc_name", "unknown")

    def get_results_directory(self) -> str:
        """Get directory for saving results."""
        return "financebench_results"

    def get_dataset_name(self) -> str:
        """Get dataset name."""
        return "financebench"

    def add_dataset_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Add FinanceBench-specific configuration."""
        config = super().add_dataset_config(config)
        config["pdf_parser"] = self.pdf_parser
        return config