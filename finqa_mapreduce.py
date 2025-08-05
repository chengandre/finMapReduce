from typing import Dict, List, Any, Tuple, Optional
import json
import os
from json_based_mapreduce import JSONBasedMapReduce
from utils import load_markdown_chunk


def load_finqa_data(json_path, num_samples=None):
    """
    Load QA pairs from FinQA json file

    Args:
        json_path (str): Path to the FinQA json file
        num_samples (int): Number of samples to load

    Returns:
        list: List of QA dictionaries with necessary information
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    qa_data = []
    count = 0

    for item in data:
        if num_samples is not None and count >= num_samples:
            break

        # Create a QA pair with the necessary information
        qa_pair = {
            "doc_name": item["doc_name"],
            "question": item["question"],
            "answer": item["answer"],
            "filename": item.get("filename", ""),
            "explanation": item.get("explanation", "")
        }

        qa_data.append(qa_pair)
        count += 1

    return qa_data


class FinQAMapReduce(JSONBasedMapReduce):
    """
    FinQA-specific implementation of MapReduce QA pipeline.

    Features:
    - Loads data from JSON format
    - Processes Markdown documents
    - Uses JSON-based prompts and responses
    - No question_type analysis (FinQA doesn't have this field)
    """

    def __init__(self, *args, doc_dir: str, **kwargs):
        """
        Initialize FinQA pipeline.

        Args:
            doc_dir: Directory containing FinQA markdown documents
            *args, **kwargs: Arguments passed to parent class
        """
        super().__init__(*args, **kwargs)
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
        return load_finqa_data(data_path, num_samples)

    def load_document_chunks(self, qa_pair: Dict[str, Any]) -> Tuple[List[Any], int]:
        """
        Load and chunk Markdown document.

        Args:
            qa_pair: Dictionary containing 'doc_name' key

        Returns:
            Tuple of (list of document chunks, total token count)
        """
        doc_name = qa_pair["doc_name"]
        markdown_file = os.path.join(self.doc_dir, doc_name)
        return load_markdown_chunk(markdown_file, self.chunk_size, self.chunk_overlap)

    def get_document_identifier(self, qa_pair: Dict[str, Any]) -> str:
        """Get document name for display."""
        return qa_pair.get("doc_name", "unknown")

    def get_results_directory(self) -> str:
        """Get directory for saving results."""
        return "finqa_results"

    def get_dataset_name(self) -> str:
        """Get dataset name."""
        return "finqa"

    def _compile_results(self, qa_data: List[Dict], evaluation_results: Dict,
                        model_name: str, judge_model_name: str, process_time: float, **kwargs) -> Dict:
        """
        Override to set pdf_parser as 'markdown' and skip question_type analysis.
        """
        results = super()._compile_results(qa_data, evaluation_results, model_name, judge_model_name, process_time, **kwargs)
        results["configuration"]["pdf_parser"] = "markdown"
        # Clear question_type analysis since FinQA doesn't have this field
        results["evaluation_summary"]["accuracy_by_question_type"] = {}
        return results