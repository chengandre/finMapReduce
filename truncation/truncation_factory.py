"""
Factory for creating truncation-based QA pipelines.

Provides a centralized way to create and configure truncation pipelines
for different datasets with various truncation strategies.
"""

from typing import Type, Dict, Optional, List, Union, Any
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_truncation_qa import BaseTruncationQA
from financebench_truncation import FinanceBenchTruncation
from finqa_truncation import FinQATruncation
from truncation_utils import validate_truncation_config, TruncationManager


class TruncationPipelineFactory:
    """
    Factory for creating truncation-based QA pipelines.

    This factory provides a centralized way to create pipeline instances
    using different truncation strategies for various datasets.
    """

    # Registry of available pipeline types
    _pipeline_registry: Dict[str, Type[BaseTruncationQA]] = {
        'financebench': FinanceBenchTruncation,
        'finqa': FinQATruncation,
    }

    @classmethod
    def create_pipeline(cls,
                       dataset: str,
                       truncation_strategy: str = "start",
                       context_window: int = 128000,
                       truncation_buffer: int = 2000,
                       max_document_tokens: Optional[int] = None,
                       **config) -> BaseTruncationQA:
        """
        Create a configured truncation pipeline instance.

        Args:
            dataset: Dataset type ('financebench' or 'finqa')
            truncation_strategy: Truncation strategy ('start', 'end', 'smart')
            context_window: Maximum context window size for the model
            truncation_buffer: Safety buffer for response tokens
            max_document_tokens: Override for max document tokens (None = auto-calculate)
            **config: Additional configuration parameters specific to the pipeline type

        Returns:
            Configured truncation pipeline instance

        Raises:
            ValueError: If dataset or configuration is invalid

        Example:
            >>> pipeline = TruncationPipelineFactory.create_pipeline(
            ...     dataset='financebench',
            ...     truncation_strategy='start',
            ...     context_window=128000,
            ...     llm=llm_instance,
            ...     prompts_dict=prompts,
            ...     pdf_parser='marker'
            ... )
        """
        # Validate dataset
        if dataset not in cls._pipeline_registry:
            available = list(cls._pipeline_registry.keys())
            raise ValueError(
                f"Unknown dataset '{dataset}'. "
                f"Available datasets: {available}"
            )

        # Validate truncation configuration
        validation = validate_truncation_config(
            truncation_strategy, 
            context_window, 
            max_document_tokens
        )
        
        if not validation["valid"]:
            errors = "; ".join(validation["errors"])
            raise ValueError(f"Invalid truncation configuration: {errors}")
        
        # Print warnings if any
        for warning in validation["warnings"]:
            print(f"Warning: {warning}")

        pipeline_class = cls._pipeline_registry[dataset]

        # Validate required parameters
        if 'llm' not in config:
            raise ValueError(f"{dataset} pipeline requires 'llm' parameter")
        if 'prompts_dict' not in config:
            raise ValueError(f"{dataset} pipeline requires 'prompts_dict' parameter")

        # Validate dataset-specific requirements
        if dataset == 'finqa' and 'doc_dir' not in config:
            raise ValueError("FinQA pipeline requires 'doc_dir' parameter")

        # Add truncation-specific parameters to config
        config.update({
            'truncation_strategy': truncation_strategy,
            'context_window': context_window,
            'truncation_buffer': truncation_buffer,
            'max_document_tokens': max_document_tokens
        })

        return pipeline_class(**config)

    @classmethod
    def register_pipeline(cls, name: str, pipeline_class: Type[BaseTruncationQA]):
        """
        Register a new truncation pipeline type.

        Args:
            name: Name to register the pipeline under
            pipeline_class: Pipeline class (must inherit from BaseTruncationQA)

        Raises:
            TypeError: If pipeline_class doesn't inherit from BaseTruncationQA
        """
        if not issubclass(pipeline_class, BaseTruncationQA):
            raise TypeError(
                f"{pipeline_class} must inherit from BaseTruncationQA"
            )

        cls._pipeline_registry[name] = pipeline_class

    @classmethod
    def get_available_datasets(cls) -> List[str]:
        """Get list of available dataset types."""
        return list(cls._pipeline_registry.keys())

    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Get list of available truncation strategies."""
        return TruncationManager.get_available_strategies()

    @classmethod
    def get_pipeline_class(cls, dataset: str) -> Optional[Type[BaseTruncationQA]]:
        """Get the class for a dataset type."""
        return cls._pipeline_registry.get(dataset)

    @classmethod
    def get_pipeline_info(cls, dataset: str, truncation_strategy: str = "start") -> Dict[str, Union[str, List[str], int]]:
        """
        Get information about a truncation pipeline configuration.

        Args:
            dataset: Dataset type
            truncation_strategy: Truncation strategy

        Returns:
            Dictionary with pipeline information

        Raises:
            ValueError: If dataset or strategy is invalid
        """
        if dataset not in cls._pipeline_registry:
            available = list(cls._pipeline_registry.keys())
            raise ValueError(
                f"Unknown dataset '{dataset}'. "
                f"Available datasets: {available}"
            )

        if truncation_strategy not in cls.get_available_strategies():
            available = cls.get_available_strategies()
            raise ValueError(
                f"Unknown truncation_strategy '{truncation_strategy}'. "
                f"Available strategies: {available}"
            )

        pipeline_class = cls._pipeline_registry[dataset]

        info: Dict[str, Union[str, List[str], int]] = {
            "dataset": dataset,
            "truncation_strategy": truncation_strategy,
            "approach": "Truncation",
            "class": pipeline_class.__name__,
            "module": pipeline_class.__module__,
            "docstring": pipeline_class.__doc__ or "No description available"
        }

        # Add dataset-specific information
        if dataset == 'financebench':
            info["data_format"] = "JSONL"
            info["document_type"] = "PDF"
            info["required_params"] = ["llm", "prompts_dict"]
            info["optional_params"] = ["pdf_parser", "truncation_buffer", "max_document_tokens"]
        elif dataset == 'finqa':
            info["data_format"] = "JSON"
            info["document_type"] = "Markdown"
            info["required_params"] = ["llm", "prompts_dict", "doc_dir"]
            info["optional_params"] = ["truncation_buffer", "max_document_tokens"]

        # Add strategy-specific information
        if truncation_strategy == "start":
            info["description"] = "Keeps the beginning of documents, truncates the end"
            info["use_case"] = "Good when important context is at document start"
        elif truncation_strategy == "end":
            info["description"] = "Keeps the end of documents, truncates the beginning"
            info["use_case"] = "Good when conclusions/summaries are at document end"
        elif truncation_strategy == "smart":
            info["description"] = "Intelligent truncation preserving important sections"
            info["use_case"] = "Best overall strategy when available (currently uses start strategy)"

        return info

    @classmethod
    def get_recommended_config(cls, 
                              dataset: str, 
                              model_name: str, 
                              document_complexity: str = "medium") -> Dict[str, Any]:
        """
        Get recommended configuration for a dataset and model combination.

        Args:
            dataset: Dataset type
            model_name: Model name (used to infer context window)
            document_complexity: Document complexity level ("simple", "medium", "complex")

        Returns:
            Dictionary with recommended configuration

        Example:
            >>> config = TruncationPipelineFactory.get_recommended_config(
            ...     'financebench', 'gpt-4o-mini', 'complex'
            ... )
        """
        # Infer context window from model name
        context_window = cls._infer_context_window(model_name)
        
        # Base configuration
        config = {
            "context_window": context_window,
            "truncation_strategy": "start",  # Safe default
            "max_concurrent_qa": 20
        }

        # Adjust based on document complexity
        if document_complexity == "simple":
            config["truncation_buffer"] = 1000
            config["max_concurrent_qa"] = 40
        elif document_complexity == "medium":
            config["truncation_buffer"] = 2000
            config["max_concurrent_qa"] = 20
        else:  # complex
            config["truncation_buffer"] = 3000
            config["max_concurrent_qa"] = 10
            config["truncation_strategy"] = "smart"  # Try smart for complex docs

        # Dataset-specific adjustments
        if dataset == 'financebench':
            config["pdf_parser"] = "marker"  # Better for financial PDFs
        elif dataset == 'finqa':
            pass  # No special requirements for FinQA

        return config

    @classmethod
    def _infer_context_window(cls, model_name: str) -> int:
        """
        Infer context window size from model name.

        Args:
            model_name: Model name

        Returns:
            Estimated context window size
        """
        model_name_lower = model_name.lower()
        
        # GPT models
        if "gpt-4o" in model_name_lower:
            return 128000
        elif "gpt-4" in model_name_lower:
            if "32k" in model_name_lower:
                return 32768
            else:
                return 8192
        elif "gpt-3.5" in model_name_lower:
            if "16k" in model_name_lower:
                return 16384
            else:
                return 4096
        
        # Claude models
        elif "claude-3" in model_name_lower or "claude-sonnet" in model_name_lower:
            return 200000
        elif "claude-2" in model_name_lower:
            return 100000
        elif "claude" in model_name_lower:
            return 100000
        
        # DeepSeek models
        elif "deepseek" in model_name_lower:
            if "r1" in model_name_lower:
                return 128000
            else:
                return 64000
        
        # Gemini models
        elif "gemini" in model_name_lower:
            if "pro" in model_name_lower:
                return 1000000  # Very large context
            else:
                return 32768
        
        # Default fallback
        else:
            print(f"Warning: Unknown model '{model_name}', using default context window of 32K")
            return 32768

    @classmethod
    def validate_configuration(cls, 
                              dataset: str, 
                              truncation_strategy: str,
                              context_window: int,
                              **config) -> Dict[str, Union[bool, List[str]]]:
        """
        Validate a complete pipeline configuration.

        Args:
            dataset: Dataset type
            truncation_strategy: Truncation strategy
            context_window: Context window size
            **config: Additional configuration parameters

        Returns:
            Dictionary with validation results
        """
        errors = []
        warnings = []

        # Check dataset
        if dataset not in cls._pipeline_registry:
            available = list(cls._pipeline_registry.keys())
            errors.append(f"Unknown dataset '{dataset}'. Available: {available}")

        # Check truncation configuration
        truncation_validation = validate_truncation_config(
            truncation_strategy, context_window, config.get('max_document_tokens')
        )
        errors.extend(truncation_validation['errors'])
        warnings.extend(truncation_validation['warnings'])

        # Check required parameters
        if 'llm' not in config:
            errors.append("Missing required parameter: 'llm'")
        if 'prompts_dict' not in config:
            errors.append("Missing required parameter: 'prompts_dict'")

        # Dataset-specific validation
        if dataset == 'finqa' and 'doc_dir' not in config:
            errors.append("FinQA pipeline requires 'doc_dir' parameter")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }