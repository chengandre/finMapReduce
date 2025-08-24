from typing import Type, Dict, Optional, List, Union, Any
from src.core.base_pipeline import BasePipeline
from src.core.mapreduce_pipeline import MapReducePipeline
from src.core.truncation_pipeline import TruncationPipeline
from src.formatters.truncation_formatter import TruncationFormatter
from src.loaders.financebench_loader import FinanceBenchLoader
from src.loaders.finqa_loader import FinQALoader
from src.loaders.dataset_loader import DatasetLoader


class PipelineFactory:
    """
    Factory for creating both MapReduce and Truncation pipelines with clean dataset/format separation.

    This factory provides a centralized way to create pipeline instances
    using composition-based architecture with dataset loaders and output formatters (MapReduce)
    or truncation formatters (Truncation). Creates loaders internally for simplified usage.
    """

    # Registry of available dataset types and their loader classes
    _dataset_registry: Dict[str, Type[DatasetLoader]] = {
        'financebench': FinanceBenchLoader,
        'finqa': FinQALoader,
        'webapp': None,  # Will be handled by direct loader instance
    }

    @classmethod
    def create_pipeline(cls,
                       dataset: str,
                       format_type: str = "json",
                       dataset_loader: Optional[DatasetLoader] = None,
                       **config) -> BasePipeline:
        """
        Create a configured pipeline instance with internally created loader.

        Args:
            dataset: Dataset type ('financebench' or 'finqa')
            format_type: Output format ('json', 'plain_text', 'hybrid')
            **config: Configuration parameters specific to the pipeline type

        Returns:
            Configured MapReducePipeline instance

        Raises:
            ValueError: If dataset or format_type is invalid

        Example:
            >>> pipeline = PipelineFactory.create_pipeline(
            ...     dataset='financebench',
            ...     format_type='hybrid',
            ...     llm=llm_instance,
            ...     prompts_dict=prompts,
            ...     pdf_parser='marker'
            ... )
        """
        if dataset not in cls._dataset_registry:
            available = list(cls._dataset_registry.keys())
            raise ValueError(
                f"Unknown dataset '{dataset}'. "
                f"Available datasets: {available}"
            )

        valid_formats = ["json", "plain_text", "hybrid"]
        if format_type not in valid_formats:
            raise ValueError(
                f"Unknown format_type '{format_type}'. "
                f"Available formats: {valid_formats}"
            )

        # Validate required parameters
        if 'llm' not in config:
            raise ValueError(f"{dataset} requires 'llm' parameter")
        if 'prompts_dict' not in config:
            raise ValueError(f"{dataset} requires 'prompts_dict' parameter")

        # Create dataset loader with dataset-specific parameters (if not provided)
        if dataset_loader is None:
            dataset_loader = cls._create_dataset_loader(dataset, config)

        # Extract loader-specific parameters from config before passing to pipeline
        cleaned_config = cls._clean_config_for_pipeline(dataset, config)

        # Add format_type and dataset_loader to config
        cleaned_config['format_type'] = format_type
        cleaned_config['dataset_loader'] = dataset_loader

        return MapReducePipeline(**cleaned_config)

    @classmethod
    def create_truncation_pipeline(cls,
                                  dataset: str,
                                  strategy: str = "start",
                                  context_window: int = 128000,
                                  buffer: int = 2000,
                                  max_document_tokens: Optional[int] = None,
                                  dataset_loader: Optional[DatasetLoader] = None,
                                  **config) -> TruncationPipeline:
        """
        Create a configured truncation pipeline instance.

        Args:
            dataset: Dataset type ('financebench' or 'finqa')
            strategy: Truncation strategy ('start', 'end', 'smart')
            context_window: Maximum context window size for the model
            buffer: Safety buffer for response tokens
            max_document_tokens: Override for max document tokens (None = auto-calculate)
            **config: Configuration parameters specific to the pipeline type

        Returns:
            Configured TruncationPipeline instance

        Raises:
            ValueError: If dataset or configuration is invalid

        Example:
            >>> pipeline = PipelineFactory.create_truncation_pipeline(
            ...     dataset='financebench',
            ...     strategy='start',
            ...     context_window=128000,
            ...     llm=llm_instance,
            ...     prompts_dict=prompts,
            ...     pdf_parser='marker'
            ... )
        """
        if dataset not in cls._dataset_registry:
            available = list(cls._dataset_registry.keys())
            raise ValueError(
                f"Unknown dataset '{dataset}'. "
                f"Available datasets: {available}"
            )

        # Validate required parameters
        if 'llm' not in config:
            raise ValueError(f"{dataset} truncation pipeline requires 'llm' parameter")
        if 'prompts_dict' not in config:
            raise ValueError(f"{dataset} truncation pipeline requires 'prompts_dict' parameter")

        # Validate dataset-specific requirements
        if dataset == 'finqa' and 'doc_dir' not in config:
            raise ValueError("FinQA truncation pipeline requires 'doc_dir' parameter")

        # Create dataset loader with dataset-specific parameters (if not provided)
        if dataset_loader is None:
            dataset_loader = cls._create_dataset_loader(dataset, config)

        # Create truncation formatter
        truncation_formatter = TruncationFormatter(
            prompts_dict=config['prompts_dict'],
            strategy=strategy,
            context_window=context_window,
            buffer=buffer,
            max_document_tokens=max_document_tokens
        )

        # Extract loader-specific parameters from config before passing to pipeline
        cleaned_config = cls._clean_config_for_pipeline(dataset, config)

        # Add truncation-specific components to config
        cleaned_config['dataset_loader'] = dataset_loader
        cleaned_config['truncation_formatter'] = truncation_formatter

        return TruncationPipeline(**cleaned_config)

    @classmethod
    def _create_dataset_loader(cls, dataset: str, config: Dict[str, Any]) -> DatasetLoader:
        """Create dataset loader with dataset-specific parameters."""
        if dataset == 'webapp':
            # Webapp should be handled by direct loader instance, not created here
            raise ValueError("Webapp dataset loader should be provided directly, not created by factory")

        loader_class = cls._dataset_registry[dataset]
        if loader_class is None:
            raise ValueError(f"No loader class registered for dataset '{dataset}'")

        if dataset == 'financebench':
            # FinanceBench loader needs pdf_parser
            pdf_parser = config.get('pdf_parser', 'marker')
            return loader_class(pdf_parser=pdf_parser)
        elif dataset == 'finqa':
            # FinQA loader needs doc_dir
            if 'doc_dir' not in config:
                raise ValueError("FinQA pipeline requires 'doc_dir' parameter")
            return loader_class(doc_dir=config['doc_dir'])
        else:
            # Default construction
            return loader_class()

    @classmethod
    def _clean_config_for_pipeline(cls, dataset: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Remove loader-specific parameters from config before passing to pipeline."""
        cleaned_config = config.copy()

        # Remove dataset-specific loader parameters
        if dataset == 'financebench':
            # pdf_parser is handled by base class, keep it
            pass
        elif dataset == 'finqa':
            # Remove doc_dir as it's handled by the loader
            cleaned_config.pop('doc_dir', None)

        return cleaned_config

    @classmethod
    def register_dataset(cls, name: str, loader_class: Type[DatasetLoader]):
        """
        Register a new dataset loader type.

        Args:
            name: Name to register the dataset under
            loader_class: DatasetLoader class (must inherit from DatasetLoader)

        Raises:
            TypeError: If loader_class doesn't inherit from DatasetLoader
        """
        if not issubclass(loader_class, DatasetLoader):
            raise TypeError(
                f"{loader_class} must inherit from DatasetLoader"
            )

        cls._dataset_registry[name] = loader_class

    @classmethod
    def get_available_datasets(cls) -> List[str]:
        """Get list of available dataset types."""
        return list(cls._dataset_registry.keys())

    @classmethod
    def get_available_formats(cls) -> List[str]:
        """Get list of available output formats for MapReduce."""
        return ["json", "plain_text", "hybrid"]

    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Get list of available truncation strategies."""
        return ["start", "end", "smart"]

    @classmethod
    def get_available_approaches(cls) -> List[str]:
        """Get list of available pipeline approaches."""
        return ["mapreduce", "truncation"]

    @classmethod
    def get_dataset_loader_class(cls, dataset: str) -> Optional[Type[DatasetLoader]]:
        """Get the loader class for a dataset type."""
        return cls._dataset_registry.get(dataset)

    @classmethod
    def get_pipeline_info(cls, dataset: str, approach: str = "mapreduce", format_type: str = "json", strategy: str = "start") -> Dict[str, Union[str, List[str]]]:
        """
        Get information about a pipeline configuration.

        Args:
            dataset: Dataset type
            approach: Pipeline approach ("mapreduce" or "truncation")
            format_type: Output format type (for MapReduce)
            strategy: Truncation strategy (for Truncation)

        Returns:
            Dictionary with pipeline information

        Raises:
            ValueError: If dataset, approach, format_type, or strategy is invalid
        """
        if dataset not in cls._dataset_registry:
            available = list(cls._dataset_registry.keys())
            raise ValueError(
                f"Unknown dataset '{dataset}'. "
                f"Available datasets: {available}"
            )

        valid_approaches = cls.get_available_approaches()
        if approach not in valid_approaches:
            raise ValueError(
                f"Unknown approach '{approach}'. "
                f"Available approaches: {valid_approaches}"
            )

        if approach == "mapreduce":
            valid_formats = cls.get_available_formats()
            if format_type not in valid_formats:
                raise ValueError(
                    f"Unknown format_type '{format_type}'. "
                    f"Available formats: {valid_formats}"
                )
        elif approach == "truncation":
            valid_strategies = cls.get_available_strategies()
            if strategy not in valid_strategies:
                raise ValueError(
                    f"Unknown strategy '{strategy}'. "
                    f"Available strategies: {valid_strategies}"
                )

        loader_class = cls._dataset_registry[dataset]

        info: Dict[str, Union[str, List[str]]] = {
            "dataset": dataset,
            "approach": approach.title(),
            "pipeline_class": "MapReducePipeline" if approach == "mapreduce" else "TruncationPipeline",
            "loader_class": loader_class.__name__,
            "loader_module": loader_class.__module__,
            "docstring": loader_class.__doc__ or "No description available"
        }

        if approach == "mapreduce":
            info["format_type"] = format_type
        elif approach == "truncation":
            info["strategy"] = strategy

        # Add dataset-specific information
        if dataset == 'financebench':
            info["data_format"] = "JSONL"
            info["document_type"] = "PDF"
            if approach == "mapreduce":
                info["required_params"] = ["llm", "prompts_dict"]
                info["optional_params"] = ["pdf_parser", "score_threshold", "map_llm", "reduce_llm", "question_improvement_llm"]
            else:  # truncation
                info["required_params"] = ["llm", "prompts_dict"]
                info["optional_params"] = ["pdf_parser", "context_window", "buffer", "max_document_tokens"]
        elif dataset == 'finqa':
            info["data_format"] = "JSON"
            info["document_type"] = "Markdown"
            if approach == "mapreduce":
                info["required_params"] = ["llm", "prompts_dict", "doc_dir"]
                info["optional_params"] = ["score_threshold", "map_llm", "reduce_llm", "question_improvement_llm"]
            else:  # truncation
                info["required_params"] = ["llm", "prompts_dict", "doc_dir"]
                info["optional_params"] = ["context_window", "buffer", "max_document_tokens"]

        # Add approach-specific information
        if approach == "mapreduce":
            # Add format-specific information
            if format_type == "json":
                info["map_output"] = "JSON with relevance scoring"
                info["reduce_input"] = "XML formatted"
                info["llm_wrapper"] = "GPT wrapper with JSON parsing"
            elif format_type == "plain_text":
                info["map_output"] = "Plain text with score extraction"
                info["reduce_input"] = "String concatenation"
                info["llm_wrapper"] = "Direct LLM invocation"
            elif format_type == "hybrid":
                info["map_output"] = "Plain text with score extraction"
                info["reduce_input"] = "String concatenation"
                info["reduce_output"] = "JSON format"
                info["llm_wrapper"] = "Mixed (direct for map, GPT wrapper for reduce)"
        elif approach == "truncation":
            # Add strategy-specific information
            if strategy == "start":
                info["description"] = "Keeps the beginning of documents, truncates the end"
                info["use_case"] = "Good when important context is at document start"
            elif strategy == "end":
                info["description"] = "Keeps the end of documents, truncates the beginning"
                info["use_case"] = "Good when conclusions/summaries are at document end"
            elif strategy == "smart":
                info["description"] = "Intelligent truncation preserving important sections"
                info["use_case"] = "Best overall strategy when available (currently uses start strategy)"

        return info