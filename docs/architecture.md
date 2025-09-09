# Architecture Documentation

## Overview

FinMapReduce uses a modular, composition-based architecture with abstract base classes and the factory pattern to provide a flexible and extensible document question answering system.

## Core Design Principles

1. **Composition over Inheritance**: Pipelines compose dataset loaders and formatters
2. **Strategy Pattern**: Different strategies for datasets, output formats, and truncation
3. **Factory Pattern**: Centralized pipeline creation with clean configuration
4. **Async-First**: Native async/await support throughout the system
5. **Rate Limiting**: Built-in token bucket rate limiting for API calls

## Architecture Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   BasePipeline  │    │ DatasetLoader   │    │ OutputFormatter │
│   (Abstract)    │    │   (Abstract)    │    │   (Abstract)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                       ▲                       ▲
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│MapReducePipeline│    │FinanceBenchLoader│   │  JSONFormatter  │
│TruncationPipeline│   │   FinQALoader   │    │ HybridFormatter │
└─────────────────┘    │  WebappLoader   │    │PlainTextFormatter│
                       └─────────────────┘    └─────────────────┘
```

## Core Components

### Base Classes

#### BasePipeline (`src/core/base_pipeline.py`)

Abstract foundation for all pipeline types providing:

- **Async processing workflow**: Template method pattern for pipeline execution
- **Document batch loading**: ThreadPoolExecutor for parallel document loading
- **Global semaphore management**: Rate limiting across all pipeline operations
- **Question preprocessing**: Optional question improvement capabilities
- **Comprehensive error handling**: Graceful failure handling and recovery
- **Result compilation**: Abstract method for pipeline-specific statistics

**Key Methods:**
- `process_dataset_async()`: Main async processing entry point
- `process_single_qa_async()`: Abstract method for single QA processing
- `compile_statistics()`: Abstract method for pipeline-specific stats
- `_batch_load_documents_async()`: Parallel document loading with caching

**Process Flow:**
1. Load and validate dataset
2. Batch load unique documents in parallel
3. Process QA pairs concurrently with semaphore management
4. Compile statistics and persist results
5. Optional LLM judge evaluation

#### DatasetLoader (`src/loaders/dataset_loader.py`)

Abstract base for dataset-specific operations:

- **Data loading**: From various formats (JSONL, JSON)
- **Document processing**: Chunking and full document loading
- **Path resolution**: Dataset-specific file path handling
- **Metadata extraction**: Dataset configuration and statistics

**Key Methods:**
- `load_data()`: Load QA pairs from dataset files
- `load_document_chunks()`: Chunk documents for MapReduce processing
- `load_full_document()`: Load complete documents for Truncation
- `get_document_identifier()`: Generate unique document identifiers
- `get_results_directory()`: Dataset-specific output directory

#### OutputFormatter (`src/formatters/output_formatter.py`)

Abstract base for output format handling:

- **LLM interaction patterns**: Format-specific prompting strategies
- **Result preprocessing**: Filtering and transformation of map results
- **Response parsing**: Format-specific answer extraction
- **Configuration management**: Format-specific parameters

**Key Methods:**
- `ainvoke_llm_map()`: Async map phase LLM calls
- `ainvoke_llm_reduce()`: Async reduce phase LLM calls
- `preprocess_map_results()`: Filter/process map results
- `parse_final_result()`: Parse final answers from LLM responses

### Concrete Implementations

#### MapReducePipeline (`src/core/mapreduce_pipeline.py`)

Production MapReduce implementation featuring:

- **Parallel map phase processing**: Async semaphore management for chunks
- **Score-based filtering**: Relevance filtering for map results
- **Token usage tracking**: Comprehensive token consumption monitoring
- **Timing statistics**: Detailed performance metrics per phase
- **Format-specific LLM configuration**: Separate LLM instances for hybrid format

**Process Flow:**
1. **Document Loading**: Parallel batch loading of all unique documents
2. **Map Phase**: Concurrent processing of document chunks with semaphore
3. **Filtering**: Score-based filtering of map results (format-dependent)
4. **Reduce Phase**: Synthesis of filtered results into final answer
5. **Evaluation**: LLM judge assessment with detailed categories

#### TruncationPipeline (`src/core/truncation_pipeline.py`)

Single-pass processing with document truncation:

- **Strategy-based truncation**: Configurable truncation strategies
- **Full document processing**: Optimized for single-pass operations
- **Context window management**: Intelligent fitting to model limits
- **Retention statistics**: Tracking of document truncation rates

**Truncation Strategies:**
- **Start**: Preserves document beginning (good for executive summaries)
- **End**: Preserves document end (good for conclusions)
- **Smart**: Intelligent section preservation (future feature)

### Dataset Loaders

#### FinanceBenchLoader (`src/loaders/financebench_loader.py`)
- **JSONL format parsing**: FinanceBench-specific data structure handling
- **PDF document processing**: Configurable PDF parsers (marker, PyPDF, etc.)
- **Evidence extraction**: Financial document evidence formatting
- **Path resolution**: FinanceBench dataset file organization

#### FinQALoader (`src/loaders/finqa_loader.py`)
- **JSON format with markdown**: Pre-processed document references
- **Table and numerical data**: Structured financial data handling
- **Question formatting**: FinQA-specific question and evidence structure

#### WebappDatasetLoader (`src/loaders/webapp_loader.py`)
- **Real-time file upload**: Dynamic document processing
- **Multi-format support**: PDF, TXT, MD file handling
- **Security validation**: File size and type checking
- **Temporary management**: Secure temporary file handling

### Output Formatters

#### JSONFormatter (`src/formatters/json_formatter.py`)
- **Structured JSON I/O**: Complete JSON processing pipeline
- **XML formatting**: Reduce phase XML structure for better parsing
- **Robust JSON parsing**: Multiple fallback strategies for malformed JSON

#### HybridFormatter (`src/formatters/hybrid_formatter.py`)
- **Text-based map phase**: Human-readable map outputs with score extraction
- **JSON-based reduce phase**: Structured reduce output for parsing
- **Score-based filtering**: Configurable relevance thresholds
- **Dual LLM instances**: Separate map and reduce LLM configurations

#### PlainTextFormatter (`src/formatters/plain_text_formatter.py`)
- **Plain text processing**: Text-based processing throughout pipeline
- **Score extraction**: Pattern-based relevance score parsing
- **String concatenation**: Simple text aggregation for reduce input

## Async LLM Client System

### AsyncLLMClient (`src/llm/async_llm_client.py`)

Comprehensive async LLM wrapper providing:

#### Rate Limiting
- **Token bucket algorithm**: Dual limits for requests and tokens
- **Configurable parameters**: Burst sizes, refill rates, monitoring
- **Statistics tracking**: Real-time rate limiting metrics

#### Response Processing
- **JSONResponseProcessor**: Robust JSON extraction with multiple fallback strategies
- **RawResponseProcessor**: Direct response handling for text formats
- **Multiple parsing strategies**: Direct JSON, code blocks, pattern matching

#### Error Handling
- **Exponential backoff**: Dynamic retry delays with jitter
- **Configurable retry conditions**: Customizable error handling
- **Timeout management**: Request timeout handling and recovery

#### Provider Support
- **Multi-provider**: OpenAI and OpenRouter integration
- **Flexible API keys**: Environment-based key management
- **Model configuration**: Provider-specific model settings

### Rate Limiting Configuration

```python
@dataclass
class RateLimitConfig:
    requests_per_minute: int = 5000
    tokens_per_minute: int = 4000000
    request_burst_size: int = 500
    token_burst_size: Optional[int] = None
```

## Factory System

### PipelineFactory (`src/core/factory.py`)

Centralized pipeline creation with:

- **Component validation**: Dataset and format compatibility checking
- **Configuration composition**: Automatic component wiring and configuration
- **Pipeline introspection**: Information about available pipelines and formats
- **Extensible registration**: Dynamic registration of new components

**Usage Example:**
```python
# Create MapReduce pipeline
pipeline = PipelineFactory.create_pipeline(
    dataset='financebench',
    format_type='hybrid',
    llm=llm_instance,
    prompts_dict=prompts,
    chunk_size=32768,
    chunk_overlap=4096
)

# Create Truncation pipeline
pipeline = PipelineFactory.create_truncation_pipeline(
    dataset='financebench',
    strategy='start',
    context_window=128000,
    llm=llm_instance,
    prompts_dict=prompts
)
```

## Pipeline Execution Flow

### MapReduce Pipeline Process

1. **Initialization**
   - Load dataset and validate format
   - Initialize LLM client with rate limiting
   - Load prompt configuration

2. **Document Loading Phase**
   - Extract unique document identifiers
   - Batch load documents in parallel using ThreadPoolExecutor
   - Cache successful loads, log failures

3. **Map Phase**
   - Chunk documents according to configuration
   - Process chunks concurrently with semaphore management
   - Extract relevance scores, summaries, and evidence

4. **Filtering Phase** (format-dependent)
   - Apply score-based filtering for hybrid/plain_text formats
   - Aggregate results for reduce phase

5. **Reduce Phase**
   - Format map results for reduce LLM
   - Synthesize final answer with reasoning and evidence
   - Parse and validate final response

6. **Evaluation Phase** (optional)
   - LLM judge assessment with structured criteria
   - Batch evaluation for efficiency

### Truncation Pipeline Process

1. **Initialization**
   - Same as MapReduce pipeline

2. **Document Loading**
   - Load full documents without chunking
   - Apply truncation strategy to fit context window

3. **Processing**
   - Single LLM call with truncated document
   - Direct answer extraction and formatting

4. **Evaluation**
   - Standard LLM judge assessment

## Performance Optimization

### Concurrency Management
- **Global semaphore**: Controls total concurrent requests across pipeline
- **Async map phase**: Parallel chunk processing with `asyncio.gather()`
- **Batch document loading**: I/O-bound operations with ThreadPoolExecutor

### Caching Strategy
- **Document cache**: Pickle-based persistent caching of parsed documents
- **Marker cache**: Specialized caching for marker parser results
- **Pipeline cache**: Webapp reuse of configured pipeline instances

### Memory Management
- **Chunk streaming**: Process chunks without loading entire documents
- **Result streaming**: Stream results to disk during processing
- **Cache cleanup**: Automatic cleanup of temporary files and caches

## Error Handling and Recovery

### Pipeline Level
- **Graceful degradation**: Continue processing with partial failures
- **Detailed logging**: Comprehensive error tracking and reporting
- **Retry mechanisms**: Configurable retry policies for transient failures

### LLM Client Level
- **Rate limit handling**: Automatic backoff and retry for rate limits
- **Timeout recovery**: Configurable timeouts with retry logic
- **Provider failover**: Future support for multiple LLM providers

### Data Processing Level
- **PDF parser fallbacks**: Multiple parsers with automatic fallback
- **JSON parsing robustness**: Multiple JSON parsing strategies
- **File system resilience**: Proper handling of file system errors

## Extensibility

### Adding New Datasets
1. Create class inheriting from `DatasetLoader`
2. Implement required abstract methods
3. Register with `PipelineFactory.register_dataset()`

### Adding New Output Formats
1. Create class inheriting from `OutputFormatter`
2. Implement LLM interaction methods
3. Update factory `get_available_formats()`

### Adding New Pipeline Types
1. Create class inheriting from `BasePipeline`
2. Implement processing workflow
3. Register with factory system

## Testing Strategy

### Unit Testing
- Individual component testing with mocks
- Pipeline workflow testing with synthetic data
- LLM client testing with response mocking

### Integration Testing
- End-to-end pipeline testing with sample datasets
- Multi-component interaction testing
- Performance testing with realistic workloads

### Load Testing
- Rate limiting validation under load
- Memory usage profiling with large documents
- Concurrent processing stress testing