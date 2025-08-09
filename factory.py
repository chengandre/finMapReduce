from typing import Type, Dict, Optional, List, Union, Any
from base_mapreduce_qa import BaseMapReduceQA
from financebench_pipeline import FinanceBenchPipeline
from finqa_pipeline import FinQAPipeline


class MapReducePipelineFactory:
    """
    Factory for creating MapReduce pipelines with clean dataset/format separation.

    This factory provides a centralized way to create pipeline instances
    using composition-based architecture with dataset loaders and output formatters.
    """

    # Registry of available pipeline types
    _pipeline_registry: Dict[str, Type[BaseMapReduceQA]] = {
        'financebench': FinanceBenchPipeline,
        'finqa': FinQAPipeline,
    }

    @classmethod
    def create_pipeline(cls,
                       dataset: str,
                       format_type: str = "json",
                       **config) -> BaseMapReduceQA:
        """
        Create a configured pipeline instance.

        Args:
            dataset: Dataset type ('financebench' or 'finqa')
            format_type: Output format ('json', 'plain_text', 'hybrid')
            **config: Configuration parameters specific to the pipeline type

        Returns:
            Configured pipeline instance

        Raises:
            ValueError: If dataset or format_type is invalid

        Example:
            >>> pipeline = MapReducePipelineFactory.create_pipeline(
            ...     dataset='financebench',
            ...     format_type='hybrid',
            ...     llm=llm_instance,
            ...     prompts_dict=prompts,
            ...     pdf_parser='marker'
            ... )
        """
        if dataset not in cls._pipeline_registry:
            available = list(cls._pipeline_registry.keys())
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

        pipeline_class = cls._pipeline_registry[dataset]

        # Validate required parameters
        if 'llm' not in config:
            raise ValueError(f"{dataset} requires 'llm' parameter")
        if 'prompts_dict' not in config:
            raise ValueError(f"{dataset} requires 'prompts_dict' parameter")

        # Validate dataset-specific requirements
        if dataset == 'finqa' and 'doc_dir' not in config:
            raise ValueError("FinQA pipeline requires 'doc_dir' parameter")

        # Add format_type to config
        config['format_type'] = format_type

        return pipeline_class(**config)

    @classmethod
    def register_pipeline(cls, name: str, pipeline_class: Type[BaseMapReduceQA]):
        """
        Register a new pipeline type.

        Args:
            name: Name to register the pipeline under
            pipeline_class: Pipeline class (must inherit from BaseMapReduceQA)

        Raises:
            TypeError: If pipeline_class doesn't inherit from BaseMapReduceQA
        """
        if not issubclass(pipeline_class, BaseMapReduceQA):
            raise TypeError(
                f"{pipeline_class} must inherit from BaseMapReduceQA"
            )

        cls._pipeline_registry[name] = pipeline_class

    @classmethod
    def get_available_datasets(cls) -> List[str]:
        """Get list of available dataset types."""
        return list(cls._pipeline_registry.keys())

    @classmethod
    def get_available_formats(cls) -> List[str]:
        """Get list of available output formats."""
        return ["json", "plain_text", "hybrid"]

    @classmethod
    def get_pipeline_class(cls, dataset: str) -> Optional[Type[BaseMapReduceQA]]:
        """Get the class for a dataset type."""
        return cls._pipeline_registry.get(dataset)

    # Convenience methods for backward compatibility
    @classmethod
    def create_financebench_pipeline(cls,
                                   llm: Any,
                                   prompts_dict: Dict[str, Any],
                                   format_type: str = "json",
                                   pdf_parser: str = "marker",
                                   **kwargs) -> BaseMapReduceQA:
        """
        Convenience method to create a FinanceBench pipeline.

        Args:
            llm: Language model instance
            prompts_dict: Dictionary containing prompt templates
            format_type: Output format type
            pdf_parser: PDF parsing method
            **kwargs: Additional configuration parameters

        Returns:
            Configured FinanceBench pipeline
        """
        return cls.create_pipeline(
            dataset='financebench',
            format_type=format_type,
            llm=llm,
            prompts_dict=prompts_dict,
            pdf_parser=pdf_parser,
            **kwargs
        )

    @classmethod
    def create_finqa_pipeline(cls,
                            llm: Any,
                            prompts_dict: Dict[str, Any],
                            doc_dir: str,
                            format_type: str = "json",
                            **kwargs) -> BaseMapReduceQA:
        """
        Convenience method to create a FinQA pipeline.

        Args:
            llm: Language model instance
            prompts_dict: Dictionary containing prompt templates
            doc_dir: Directory containing FinQA markdown documents
            format_type: Output format type
            **kwargs: Additional configuration parameters

        Returns:
            Configured FinQA pipeline
        """
        return cls.create_pipeline(
            dataset='finqa',
            format_type=format_type,
            llm=llm,
            prompts_dict=prompts_dict,
            doc_dir=doc_dir,
            **kwargs
        )

    @classmethod
    def get_pipeline_info(cls, dataset: str, format_type: str = "json") -> Dict[str, Union[str, List[str]]]:
        """
        Get information about a pipeline configuration.

        Args:
            dataset: Dataset type
            format_type: Output format type

        Returns:
            Dictionary with pipeline information

        Raises:
            ValueError: If dataset or format_type is invalid
        """
        if dataset not in cls._pipeline_registry:
            available = list(cls._pipeline_registry.keys())
            raise ValueError(
                f"Unknown dataset '{dataset}'. "
                f"Available datasets: {available}"
            )

        valid_formats = cls.get_available_formats()
        if format_type not in valid_formats:
            raise ValueError(
                f"Unknown format_type '{format_type}'. "
                f"Available formats: {valid_formats}"
            )

        pipeline_class = cls._pipeline_registry[dataset]

        info: Dict[str, Union[str, List[str]]] = {
            "dataset": dataset,
            "format_type": format_type,
            "class": pipeline_class.__name__,
            "module": pipeline_class.__module__,
            "docstring": pipeline_class.__doc__ or "No description available"
        }

        # Add dataset-specific information
        if dataset == 'financebench':
            info["data_format"] = "JSONL"
            info["document_type"] = "PDF"
            info["required_params"] = ["llm", "prompts_dict"]
            info["optional_params"] = ["pdf_parser", "score_threshold", "map_llm", "reduce_llm", "question_improvement_llm"]
        elif dataset == 'finqa':
            info["data_format"] = "JSON"
            info["document_type"] = "Markdown"
            info["required_params"] = ["llm", "prompts_dict", "doc_dir"]
            info["optional_params"] = ["score_threshold", "map_llm", "reduce_llm", "question_improvement_llm"]

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

        return info