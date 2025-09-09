# API Reference

This document provides detailed API documentation for all classes and methods in the FinMapReduce system.

## Core Pipeline Classes

### BasePipeline

Base abstract class for all pipeline types.

```python
class BasePipeline(ABC):
    """Abstract base class for all pipeline implementations."""

    def __init__(self, dataset_loader: DatasetLoader, llm: AsyncLLMClient,
                 prompts_dict: Dict[str, Any], max_total_requests: int = 20)

    @abstractmethod
    async def process_single_qa_async(self, qa_pair: Dict[str, Any],
                                    document_cache: Optional[Dict] = None) -> Dict[str, Any]:
        """Process single QA pair (abstract method)."""

    @abstractmethod
    def compile_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compile pipeline-specific statistics (abstract method)."""

    async def process_dataset_async(self, data_path: str, model_name: str,
                                  num_samples: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """Main async processing method."""

    async def _batch_load_documents_async(self, qa_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Batch load documents in parallel with caching."""
```

### MapReducePipeline

MapReduce implementation extending BasePipeline.

```python
class MapReducePipeline(BasePipeline):
    """MapReduce pipeline with parallel map phase processing."""

    def __init__(self, dataset_loader: DatasetLoader, output_formatter: OutputFormatter,
                 llm: AsyncLLMClient, prompts_dict: Dict[str, Any],
                 chunk_size: int = 32768, chunk_overlap: int = 4096,
                 max_total_requests: int = 20)

    async def process_single_qa_async(self, qa_pair: Dict[str, Any],
                                    document_cache: Optional[Dict] = None) -> Dict[str, Any]:
        """Process single QA pair using MapReduce approach."""

    def compile_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compile MapReduce-specific statistics including token usage and timing."""
```

### TruncationPipeline

Truncation implementation extending BasePipeline.

```python
class TruncationPipeline(BasePipeline):
    """Truncation pipeline with single-pass document processing."""

    def __init__(self, dataset_loader: DatasetLoader, formatter: TruncationFormatter,
                 llm: AsyncLLMClient, prompts_dict: Dict[str, Any],
                 strategy: str = "start", context_window: int = 128000,
                 buffer: int = 2000, max_total_requests: int = 20)

    async def process_single_qa_async(self, qa_pair: Dict[str, Any],
                                    document_cache: Optional[Dict] = None) -> Dict[str, Any]:
        """Process single QA pair using truncation approach."""

    def compile_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compile truncation-specific statistics including retention rates."""
```

## Factory Classes

### PipelineFactory

Centralized factory for creating pipeline instances.

```python
class PipelineFactory:
    """Factory for creating and configuring pipeline instances."""

    @classmethod
    def create_pipeline(cls, dataset: str, format_type: str, llm: AsyncLLMClient,
                       prompts_dict: Dict[str, Any], chunk_size: int = 32768,
                       chunk_overlap: int = 4096, **kwargs) -> BasePipeline:
        """Create MapReduce pipeline with specified configuration."""

    @classmethod
    def create_truncation_pipeline(cls, dataset: str, strategy: str,
                                 context_window: int, llm: AsyncLLMClient,
                                 prompts_dict: Dict[str, Any], **kwargs) -> TruncationPipeline:
        """Create Truncation pipeline with specified strategy."""

    @classmethod
    def get_available_datasets(cls) -> List[str]:
        """Get list of available dataset types."""

    @classmethod
    def get_available_formats(cls) -> List[str]:
        """Get list of available output formats."""

    @classmethod
    def get_pipeline_info(cls, dataset: str, approach: str, format_type: str) -> Dict:
        """Get detailed information about pipeline configuration."""

    @classmethod
    def register_dataset(cls, name: str, loader_class: Type[DatasetLoader]):
        """Register new dataset loader."""
```

## Dataset Loader Classes

### DatasetLoader

Abstract base class for dataset-specific operations.

```python
class DatasetLoader(ABC):
    """Abstract base class for dataset loading and processing."""

    @abstractmethod
    def load_data(self, data_path: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load QA data from dataset file."""

    @abstractmethod
    def load_document_chunks(self, qa_pair: Dict[str, Any], chunk_size: int,
                           chunk_overlap: int, pdf_parser: str = "marker") -> Tuple[List[Any], int]:
        """Load and chunk document for MapReduce processing."""

    @abstractmethod
    def load_full_document(self, qa_pair: Dict[str, Any],
                         pdf_parser: str = "marker") -> Tuple[str, int]:
        """Load complete document for Truncation processing."""

    @abstractmethod
    def get_document_identifier(self, qa_pair: Dict[str, Any]) -> str:
        """Generate unique identifier for document."""

    @abstractmethod
    def get_results_directory(self) -> str:
        """Get directory for saving results."""
```

### FinanceBenchLoader

FinanceBench dataset loader implementation.

```python
class FinanceBenchLoader(DatasetLoader):
    """Loader for FinanceBench dataset with PDF document processing."""

    def load_data(self, data_path: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load FinanceBench JSONL data."""

    def load_document_chunks(self, qa_pair: Dict[str, Any], chunk_size: int,
                           chunk_overlap: int, pdf_parser: str = "marker") -> Tuple[List[Any], int]:
        """Load and chunk PDF documents."""

    def load_full_document(self, qa_pair: Dict[str, Any],
                         pdf_parser: str = "marker") -> Tuple[str, int]:
        """Load complete PDF document."""

    def get_document_identifier(self, qa_pair: Dict[str, Any]) -> str:
        """Generate identifier from doc_name field."""

    def get_results_directory(self) -> str:
        """Return 'results/financebench_results'."""
```

### FinQALoader

FinQA dataset loader implementation.

```python
class FinQALoader(DatasetLoader):
    """Loader for FinQA dataset with markdown document processing."""

    def __init__(self, doc_dir: str):
        """Initialize with document directory path."""

    def load_data(self, data_path: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load FinQA JSON data."""

    def load_document_chunks(self, qa_pair: Dict[str, Any], chunk_size: int,
                           chunk_overlap: int, pdf_parser: str = "marker") -> Tuple[List[Any], int]:
        """Load and chunk markdown documents."""

    def load_full_document(self, qa_pair: Dict[str, Any],
                         pdf_parser: str = "marker") -> Tuple[str, int]:
        """Load complete markdown document."""

    def get_document_identifier(self, qa_pair: Dict[str, Any]) -> str:
        """Generate identifier from filename."""

    def get_results_directory(self) -> str:
        """Return 'results/finqa_results'."""
```

### WebappDatasetLoader

Webapp file upload loader implementation.

```python
class WebappDatasetLoader(DatasetLoader):
    """Loader for webapp file uploads with multi-format support."""

    def __init__(self, file_path: str, question: str):
        """Initialize with uploaded file path and question."""

    def load_data(self, data_path: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Create single QA pair from file and question."""

    def load_document_chunks(self, qa_pair: Dict[str, Any], chunk_size: int,
                           chunk_overlap: int, pdf_parser: str = "marker") -> Tuple[List[Any], int]:
        """Load and chunk uploaded file."""

    def load_full_document(self, qa_pair: Dict[str, Any],
                         pdf_parser: str = "marker") -> Tuple[str, int]:
        """Load complete uploaded document."""

    def get_document_identifier(self, qa_pair: Dict[str, Any]) -> str:
        """Generate identifier from file path."""

    def get_results_directory(self) -> str:
        """Return 'results/webapp_results'."""
```

## Output Formatter Classes

### OutputFormatter

Abstract base class for output format handling.

```python
class OutputFormatter(ABC):
    """Abstract base class for output formatting and LLM interaction."""

    def __init__(self, llm: AsyncLLMClient, prompts_dict: Dict[str, Any]):
        """Initialize with LLM client and prompts."""

    @abstractmethod
    async def ainvoke_llm_map(self, chunk: Any, question: str, **kwargs) -> Dict[str, Any]:
        """Invoke LLM for map phase processing."""

    @abstractmethod
    async def ainvoke_llm_reduce(self, formatted_results: Any, question: str, **kwargs) -> Any:
        """Invoke LLM for reduce phase processing."""

    @abstractmethod
    def preprocess_map_results(self, map_results: List[Dict[str, Any]], **kwargs) -> Any:
        """Preprocess map results for reduce phase."""

    @abstractmethod
    def parse_final_result(self, reduce_result: Any) -> Tuple[str, str, List[str]]:
        """Parse final result into answer, reasoning, evidence."""
```

### JSONFormatter

JSON-based output formatter implementation.

```python
class JSONFormatter(OutputFormatter):
    """JSON-based formatter with structured input/output."""

    async def ainvoke_llm_map(self, chunk: Any, question: str, **kwargs) -> Dict[str, Any]:
        """Process chunk with JSON-structured prompts."""

    async def ainvoke_llm_reduce(self, formatted_results: Any, question: str, **kwargs) -> Any:
        """Synthesize results with XML-formatted reduce prompts."""

    def preprocess_map_results(self, map_results: List[Dict[str, Any]], **kwargs) -> Any:
        """Format map results as XML for reduce phase."""

    def parse_final_result(self, reduce_result: Any) -> Tuple[str, str, List[str]]:
        """Parse JSON response into components."""
```

### HybridFormatter

Hybrid text/JSON formatter implementation.

```python
class HybridFormatter(OutputFormatter):
    """Hybrid formatter with text map phase and JSON reduce phase."""

    def __init__(self, map_llm: AsyncLLMClient, reduce_llm: AsyncLLMClient,
                 prompts_dict: Dict[str, Any]):
        """Initialize with separate map and reduce LLM clients."""

    async def ainvoke_llm_map(self, chunk: Any, question: str, **kwargs) -> Dict[str, Any]:
        """Process chunk with text-based prompts and score extraction."""

    async def ainvoke_llm_reduce(self, formatted_results: Any, question: str, **kwargs) -> Any:
        """Synthesize with JSON-structured reduce prompts."""

    def preprocess_map_results(self, map_results: List[Dict[str, Any]],
                             score_threshold: int = 5, **kwargs) -> Any:
        """Filter by score and format for reduce phase."""

    def parse_final_result(self, reduce_result: Any) -> Tuple[str, str, List[str]]:
        """Parse JSON response with fallback strategies."""
```

### PlainTextFormatter

Plain text formatter implementation.

```python
class PlainTextFormatter(OutputFormatter):
    """Plain text formatter with text processing throughout."""

    async def ainvoke_llm_map(self, chunk: Any, question: str, **kwargs) -> Dict[str, Any]:
        """Process chunk with text prompts and score extraction."""

    async def ainvoke_llm_reduce(self, formatted_results: Any, question: str, **kwargs) -> Any:
        """Synthesize with text-based reduce prompts."""

    def preprocess_map_results(self, map_results: List[Dict[str, Any]],
                             score_threshold: int = 5, **kwargs) -> Any:
        """Filter by score and concatenate text results."""

    def parse_final_result(self, reduce_result: Any) -> Tuple[str, str, List[str]]:
        """Parse text response with pattern matching."""
```

### TruncationFormatter

Truncation-specific formatter implementation.

```python
class TruncationFormatter:
    """Formatter for truncation pipeline with single-pass processing."""

    def __init__(self, llm: AsyncLLMClient, prompts_dict: Dict[str, Any]):
        """Initialize with LLM client and prompts."""

    async def ainvoke_llm(self, document: str, question: str, **kwargs) -> Dict[str, Any]:
        """Process full document with single LLM call."""

    def parse_result(self, llm_result: Any) -> Tuple[str, str, List[str]]:
        """Parse response into answer, reasoning, evidence."""
```

## LLM Client Classes

### AsyncLLMClient

Async LLM client with rate limiting and error handling.

```python
class AsyncLLMClient:
    """Async LLM client with comprehensive rate limiting and error handling."""

    def __init__(self, model_name: str, provider: str = "openai",
                 temperature: float = 0.0, rate_limit_config: Optional[RateLimitConfig] = None,
                 response_processor: Optional[ResponseProcessor] = None):
        """Initialize client with model and rate limiting configuration."""

    async def ainvoke(self, prompt: str, **kwargs) -> Any:
        """Async LLM invocation with rate limiting and retry logic."""

    async def get_stats(self) -> Optional[Dict[str, Any]]:
        """Get current rate limiting statistics."""

    async def reset_stats(self):
        """Reset rate limiting statistics."""
```

### Rate Limiting Classes

```python
@dataclass
class RateLimitConfig:
    """Configuration for rate limiting behavior."""
    requests_per_minute: int = 5000
    tokens_per_minute: int = 4000000
    request_burst_size: int = 500
    token_burst_size: Optional[int] = None

class TokenBucketRateLimiter:
    """Token bucket implementation for rate limiting."""

    def __init__(self, requests_per_minute: int, tokens_per_minute: int,
                 request_burst_size: int, token_burst_size: Optional[int] = None):
        """Initialize rate limiter with specified limits."""

    async def acquire_request(self) -> None:
        """Acquire permission for a request."""

    async def acquire_tokens(self, token_count: int) -> None:
        """Acquire permission for specified token count."""

    def get_stats(self) -> Dict[str, Any]:
        """Get current rate limiting statistics."""
```

### Response Processor Classes

```python
class ResponseProcessor(ABC):
    """Abstract base for response processing strategies."""

    @abstractmethod
    def process_response(self, response: str) -> Any:
        """Process LLM response according to strategy."""

class JSONResponseProcessor(ResponseProcessor):
    """Processor for JSON responses with multiple parsing strategies."""

    def process_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON with multiple fallback strategies."""

class RawResponseProcessor(ResponseProcessor):
    """Processor for raw text responses."""

    def process_response(self, response: str) -> str:
        """Return response as-is."""
```

## Evaluation Classes

### AsyncLLMJudgeEvaluator

LLM-based evaluation system with batch processing.

```python
class AsyncLLMJudgeEvaluator:
    """Async LLM judge evaluation with batch processing."""

    def __init__(self, judge_llm: AsyncLLMClient, prompts_dict: Dict[str, Any]):
        """Initialize with judge LLM and evaluation prompts."""

    async def evaluate_batch_async(self, results: List[Dict[str, Any]],
                                 dataset_name: str, model_name: str,
                                 batch_size: int = 10) -> Dict[str, Any]:
        """Evaluate results in batches for efficiency."""

    def compile_evaluation_statistics(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compile evaluation statistics and accuracy metrics."""
```

### Evaluation Formatter Classes

```python
class EvaluationFormatter(ABC):
    """Abstract base for evaluation formatting strategies."""

    @abstractmethod
    def extract_sample_data(self, sample: Dict[str, Any]) -> Tuple[str, str, str]:
        """Extract question, expected answer, actual answer from sample."""

    @abstractmethod
    def format_item(self, question: str, expected: str, actual: str, index: int) -> str:
        """Format single evaluation item for judge prompt."""

class JSONEvaluationFormatter(EvaluationFormatter):
    """Formatter for JSON-based evaluation results."""

class HybridEvaluationFormatter(EvaluationFormatter):
    """Formatter for hybrid-based evaluation results."""

class TruncationEvaluationFormatter(EvaluationFormatter):
    """Formatter for truncation-based evaluation results."""
```

## Utility Functions

### Document Processing

```python
def load_document_chunk(doc_path: str, chunk_size: int, chunk_overlap: int,
                       method: str = "marker") -> Tuple[List[Document], int]:
    """
    Load and chunk a document using specified parsing method.

    Args:
        doc_path: Path to document file
        chunk_size: Size of chunks in tokens
        chunk_overlap: Overlap between chunks
        method: Parser method ('marker', 'pypdf', 'pymu', etc.)

    Returns:
        Tuple of (document chunks, total token count)
    """

def load_full_document(doc_path: str, method: str = "marker") -> Tuple[str, int]:
    """
    Load complete document using specified parsing method.

    Args:
        doc_path: Path to document file
        method: Parser method ('marker', 'pypdf', 'pymu', etc.)

    Returns:
        Tuple of (document text, token count)
    """
```

### Prompt Management

```python
def load_prompt_set(prompt_set_name: str) -> Dict[str, Any]:
    """
    Load prompts from YAML configuration files.

    Args:
        prompt_set_name: Name of prompt set from prompt_config.yml

    Returns:
        Dictionary containing loaded prompts
    """

def load_yaml_prompt(file_path: str) -> str:
    """
    Load individual YAML prompt file.

    Args:
        file_path: Path to YAML prompt file

    Returns:
        Loaded prompt as string
    """
```

### LLM Utilities

```python
def create_async_rate_limited_llm(model_name: str, provider: str = "openai",
                                temperature: float = 0.0, parse_json: bool = False,
                                rate_limit_config: Optional[RateLimitConfig] = None) -> AsyncLLMClient:
    """
    Create configured AsyncLLMClient with rate limiting.

    Args:
        model_name: Name of LLM model
        provider: LLM provider ("openai", "openrouter")
        temperature: Model temperature
        parse_json: Whether to parse JSON responses
        rate_limit_config: Rate limiting configuration

    Returns:
        Configured AsyncLLMClient instance
    """
```

## Error Classes

```python
class PipelineError(Exception):
    """Base exception for pipeline errors."""

class DatasetLoadError(PipelineError):
    """Error loading dataset."""

class DocumentProcessingError(PipelineError):
    """Error processing document."""

class LLMInvocationError(PipelineError):
    """Error invoking LLM."""

class RateLimitExceededError(PipelineError):
    """Rate limit exceeded error."""

class ConfigurationError(PipelineError):
    """Invalid configuration error."""
```

## Type Definitions

```python
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

# Common type aliases
QAPair = Dict[str, Any]
Document = Any  # LangChain Document type
TokenStats = Dict[str, int]
TimingStats = Dict[str, float]
EvaluationResult = Dict[str, Any]
PipelineResult = Dict[str, Any]
```