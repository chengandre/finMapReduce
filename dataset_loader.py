from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional


class DatasetLoader(ABC):
    """
    Abstract base class for loading dataset-specific data and documents.

    This strategy class handles all dataset-specific operations:
    - Loading QA data from files
    - Loading and chunking documents
    - Providing dataset metadata
    """

    @abstractmethod
    def load_data(self, data_path: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load QA data from dataset-specific format.

        Args:
            data_path: Path to the dataset file
            num_samples: Number of samples to load (None for all)

        Returns:
            List of QA dictionaries with question, answer, evidence, etc.
        """
        pass

    @abstractmethod
    def load_document_chunks(self, qa_pair: Dict[str, Any], chunk_size: int, chunk_overlap: int) -> Tuple[List[Any], int]:
        """
        Load and chunk a document for the given QA pair.

        Args:
            qa_pair: Dictionary containing document information
            chunk_size: Size of document chunks for processing
            chunk_overlap: Overlap between consecutive chunks

        Returns:
            Tuple of (list of document chunks, total token count)
        """
        pass

    @abstractmethod
    def load_full_document(self, qa_pair: Dict[str, Any]) -> Tuple[str, int]:
        """
        Load the full document content for a QA pair (for truncation approaches).

        Args:
            qa_pair: Dictionary containing document information

        Returns:
            Tuple of (full document text, total token count)
        """
        pass

    @abstractmethod
    def get_document_identifier(self, qa_pair: Dict[str, Any]) -> str:
        """
        Get a display name for the document being processed.

        Args:
            qa_pair: Dictionary containing document information

        Returns:
            String identifier for the document
        """
        pass

    @abstractmethod
    def get_results_directory(self) -> str:
        """
        Get the directory name for saving results.

        Returns:
            Directory name for this dataset's results
        """
        pass

    @abstractmethod
    def get_dataset_name(self) -> str:
        """
        Get the name of the dataset.

        Returns:
            Dataset name string
        """
        pass

    def get_map_question(self, qa_pair: Dict[str, Any]) -> str:
        """
        Get the question to use in the map phase.

        Default implementation returns the original question.
        Override if dataset needs question transformation.

        Args:
            qa_pair: Dictionary containing question information

        Returns:
            Question string for map phase
        """
        return qa_pair["question"]

    def add_dataset_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add dataset-specific configuration to results.

        Default implementation adds basic dataset info.
        Override to add dataset-specific configuration.

        Args:
            config: Existing configuration dictionary

        Returns:
            Updated configuration dictionary
        """
        config["dataset"] = self.get_dataset_name()
        return config