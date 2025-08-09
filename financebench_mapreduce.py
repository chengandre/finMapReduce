from typing import Dict, List, Any, Tuple, Optional
import json
from json_based_mapreduce import JSONBasedMapReduce
from utils import load_pdf_chunk


def load_financebench_data(jsonl_path, num_samples=None):
    """
    Load QA pairs from financebench_open_source.jsonl file

    Args:
        jsonl_path (str): Path to the financebench jsonl file
        num_samples (int): Number of samples to load

    Returns:
        list: List of QA dictionaries with necessary information
    """
    qa_data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    count = 0
    for line in lines:
        if num_samples is not None and count >= num_samples:
            break

        item = json.loads(line)

        # Create a QA pair with the necessary information
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


class FinanceBenchMapReduce(JSONBasedMapReduce):
    """
    FinanceBench-specific implementation of MapReduce QA pipeline.

    Features:
    - Loads data from JSONL format
    - Processes PDF documents with configurable parsers
    - Uses JSON-based prompts and responses
    - Includes question_type analysis in evaluation
    """

    def __init__(self, *args, pdf_parser: str = "marker", **kwargs):
        """
        Initialize FinanceBench pipeline.

        Args:
            pdf_parser: PDF parsing method ('marker', 'pypdf', 'pymu', 'unstructured', 'default')
            *args, **kwargs: Arguments passed to parent class
        """
        super().__init__(*args, **kwargs)
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



    def get_results_directory(self) -> str:
        """Get directory for saving results."""
        return "financebench_results"

    def get_dataset_name(self) -> str:
        """Get dataset name."""
        return "financebench"

    def _compile_results(self, qa_data: List[Dict], evaluation_results: Dict,
                        model_name: str, judge_model_name: str, process_time: float, **kwargs) -> Dict:
        """
        Override to add pdf_parser to configuration.
        """
        results = super()._compile_results(qa_data, evaluation_results, model_name, judge_model_name,process_time, **kwargs)
        results["configuration"]["pdf_parser"] = self.pdf_parser
        return results