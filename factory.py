from typing import Type, Dict, Optional, List, Union
from base_mapreduce_qa import BaseMapReduceQA
from financebench_mapreduce import FinanceBenchMapReduce
from finqa_mapreduce import FinQAMapReduce
from last_year_mapreduce import LastYearMapReduce


class MapReducePipelineFactory:
    """
    Factory for creating MapReduce pipelines with proper configuration.

    This factory provides a centralized way to create pipeline instances
    and allows for easy registration of new pipeline types.
    """

    # Registry of available pipeline types
    _pipeline_registry: Dict[str, Type[BaseMapReduceQA]] = {
        'financebench': FinanceBenchMapReduce,
        'finqa': FinQAMapReduce,
        'last_year': LastYearMapReduce,
    }

    @classmethod
    def create_pipeline(cls, pipeline_type: str, **config) -> BaseMapReduceQA:
        """
        Create a configured pipeline instance.

        Args:
            pipeline_type: Type of pipeline to create
            **config: Configuration parameters specific to the pipeline type

        Returns:
            Configured pipeline instance

        Raises:
            ValueError: If pipeline_type is not registered

        Example:
            >>> pipeline = MapReducePipelineFactory.create_pipeline(
            ...     'financebench',
            ...     llm=llm_instance,
            ...     prompts_dict=prompts,
            ...     pdf_parser='marker'
            ... )
        """
        if pipeline_type not in cls._pipeline_registry:
            available = list(cls._pipeline_registry.keys())
            raise ValueError(
                f"Unknown pipeline type '{pipeline_type}'. "
                f"Available types: {available}"
            )

        pipeline_class = cls._pipeline_registry[pipeline_type]

        # Validate required parameters for each pipeline type
        if pipeline_type in ['financebench', 'finqa', 'last_year']:
            if 'llm' not in config:
                raise ValueError(f"{pipeline_type} requires 'llm' parameter")
            if 'prompts_dict' not in config:
                raise ValueError(f"{pipeline_type} requires 'prompts_dict' parameter")

        if pipeline_type == 'finqa' and 'doc_dir' not in config:
            raise ValueError("FinQA pipeline requires 'doc_dir' parameter")

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

        Example:
            >>> MapReducePipelineFactory.register_pipeline(
            ...     'custom',
            ...     CustomMapReducePipeline
            ... )
        """
        if not issubclass(pipeline_class, BaseMapReduceQA):
            raise TypeError(
                f"{pipeline_class} must inherit from BaseMapReduceQA"
            )

        cls._pipeline_registry[name] = pipeline_class

    @classmethod
    def get_available_pipelines(cls) -> List[str]:
        """Get list of available pipeline types."""
        return list(cls._pipeline_registry.keys())

    @classmethod
    def get_pipeline_class(cls, pipeline_type: str) -> Optional[Type[BaseMapReduceQA]]:
        """Get the class for a pipeline type."""
        return cls._pipeline_registry.get(pipeline_type)

    @classmethod
    def create_financebench_pipeline(cls, llm, prompts_dict, pdf_parser: str = "marker", **kwargs) -> BaseMapReduceQA:
        """
        Convenience method to create a FinanceBench pipeline.

        Args:
            llm: Language model instance
            prompts_dict: Dictionary containing prompt templates
            pdf_parser: PDF parsing method
            **kwargs: Additional configuration parameters

        Returns:
            Configured FinanceBench pipeline
        """
        return cls.create_pipeline(
            'financebench',
            llm=llm,
            prompts_dict=prompts_dict,
            pdf_parser=pdf_parser,
            **kwargs
        )

    @classmethod
    def create_finqa_pipeline(cls, llm, prompts_dict, doc_dir: str, **kwargs) -> BaseMapReduceQA:
        """
        Convenience method to create a FinQA pipeline.

        Args:
            llm: Language model instance
            prompts_dict: Dictionary containing prompt templates
            doc_dir: Directory containing FinQA markdown documents
            **kwargs: Additional configuration parameters

        Returns:
            Configured FinQA pipeline
        """
        return cls.create_pipeline(
            'finqa',
            llm=llm,
            prompts_dict=prompts_dict,
            doc_dir=doc_dir,
            **kwargs
        )

    @classmethod
    def create_last_year_pipeline(cls, llm, prompts_dict, parser_method: str = "pypdf", **kwargs) -> BaseMapReduceQA:
        """
        Convenience method to create a Last Year evaluation pipeline.

        Args:
            llm: Language model instance (should be RateLimitedRetryLLM)
            prompts_dict: Dictionary containing last year prompt templates
            parser_method: PDF parsing method
            **kwargs: Additional configuration parameters

        Returns:
            Configured Last Year pipeline
        """
        return cls.create_pipeline(
            'last_year',
            llm=llm,
            prompts_dict=prompts_dict,
            parser_method=parser_method,
            **kwargs
        )

    @classmethod
    def get_pipeline_info(cls, pipeline_type: str) -> Dict[str, Union[str, List[str]]]:
        """
        Get information about a pipeline type.

        Args:
            pipeline_type: Type of pipeline

        Returns:
            Dictionary with pipeline information

        Raises:
            ValueError: If pipeline_type is not registered
        """
        if pipeline_type not in cls._pipeline_registry:
            available = list(cls._pipeline_registry.keys())
            raise ValueError(
                f"Unknown pipeline type '{pipeline_type}'. "
                f"Available types: {available}"
            )

        pipeline_class = cls._pipeline_registry[pipeline_type]

        info: Dict[str, Union[str, List[str]]] = {
            "name": pipeline_type,
            "class": pipeline_class.__name__,
            "module": pipeline_class.__module__,
            "docstring": pipeline_class.__doc__ or "No description available"
        }

        # Add specific information based on pipeline type
        if pipeline_type == 'financebench':
            info["data_format"] = "JSONL"
            info["document_type"] = "PDF"
            info["required_params"] = ["llm", "prompts_dict"]
            info["optional_params"] = ["pdf_parser"]
        elif pipeline_type == 'finqa':
            info["data_format"] = "JSON"
            info["document_type"] = "Markdown"
            info["required_params"] = ["llm", "prompts_dict", "doc_dir"]
            info["optional_params"] = []
        elif pipeline_type == 'last_year':
            info["data_format"] = "JSONL (FinanceBench)"
            info["document_type"] = "PDF"
            info["required_params"] = ["llm", "prompts_dict"]
            info["optional_params"] = ["parser_method"]

        return info