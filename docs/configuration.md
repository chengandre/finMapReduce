# Configuration Guide

This guide covers all configuration options for FinMapReduce, including environment variables, prompt configurations, and processing parameters.

## Environment Variables

### Required Configuration

These environment variables must be set for the system to function:

```bash
# API Keys (at least one required)
OPENAI_API_KEY=your_openai_key_here
OPENROUTER_API_KEY=your_openrouter_key_here  # Optional
SELF_OPENAI_API_KEY=your_alternative_key     # Optional
```

### Optional Configuration

#### Server Settings

```bash
# Web application server configuration
HOST=0.0.0.0                    # Host to bind server to
PORT=8000                       # Port to run server on
DEBUG=false                     # Enable debug mode
LOG_LEVEL=info                  # Logging level (debug, info, warning, error)
```

#### Default Processing Settings

```bash
# Default LLM Configuration
DEFAULT_MODEL=gpt-4o-mini       # Default model name
DEFAULT_PROVIDER=openai         # Default provider (openai, openrouter)
DEFAULT_TEMPERATURE=0.0         # Default temperature (0.0-2.0)

# Default Pipeline Configuration
DEFAULT_CHUNK_SIZE=32768        # Default chunk size in characters
DEFAULT_CHUNK_OVERLAP=4096      # Default chunk overlap in characters
DEFAULT_FORMAT_TYPE=hybrid      # Default output format (json, hybrid, plain_text)
DEFAULT_PDF_PARSER=marker       # Default PDF parser (marker, pypdf, pymu, etc.)
DEFAULT_MAX_CONCURRENT_CHUNKS=50 # Maximum concurrent chunk processing
```

#### File Upload Settings

```bash
# File upload limits and handling
MAX_FILE_SIZE=52428800          # Maximum file size (50MB default)
TEMP_DIR=/tmp/webapp_uploads    # Temporary file directory
UPLOAD_TIMEOUT=300              # Upload timeout in seconds
```

#### API Configuration

```bash
# API metadata and settings
API_TITLE=MapReduce QA WebApp   # API title in OpenAPI docs
API_VERSION=1.0.0               # API version
API_DESCRIPTION=FinMapReduce Question Answering System
```

#### Rate Limiting Configuration

```bash
# Default rate limiting settings
DEFAULT_REQUESTS_PER_MINUTE=5000    # Default requests per minute
DEFAULT_TOKENS_PER_MINUTE=4000000   # Default tokens per minute
DEFAULT_REQUEST_BURST_SIZE=500      # Default request burst size
DEFAULT_TOKEN_BURST_SIZE=50000      # Default token burst size
```

### Environment File Setup

Create a `.env` file in the project root:

```bash
# Copy example environment file
cp .env.example .env

# Edit with your configuration
nano .env
```

Example `.env` file:

```bash
# Required API Keys
OPENAI_API_KEY=sk-your-openai-key-here
OPENROUTER_API_KEY=sk-or-your-openrouter-key-here
SELF_OPENAI_API_KEY=sk-your-alternative-key-here

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false
LOG_LEVEL=info

# Default Processing Settings
DEFAULT_MODEL=gpt-4o-mini
DEFAULT_PROVIDER=openai
DEFAULT_TEMPERATURE=0.0
DEFAULT_CHUNK_SIZE=32768
DEFAULT_CHUNK_OVERLAP=4096
DEFAULT_FORMAT_TYPE=hybrid
DEFAULT_PDF_PARSER=marker
DEFAULT_MAX_CONCURRENT_CHUNKS=50

# File Upload Settings
MAX_FILE_SIZE=52428800  # 50MB
TEMP_DIR=/tmp/webapp_uploads

# API Settings
API_TITLE=MapReduce QA WebApp
API_VERSION=1.0.0
```

## Prompt Configuration

### Prompt Configuration File

Prompts are configured in `config/prompts/prompt_config.yml`:

```yaml
prompt_sets:
  default:
    map_prompt: "config/prompts/map_prompt_hybrid.yml"
    reduce_prompt: "config/prompts/reduce_prompt_hybrid.yml"
    judge_prompt: "config/prompts/judge_prompt.yml"

  baseline:
    map_prompt: "config/prompts/map_prompt_baseline.yml"
    reduce_prompt: "config/prompts/reduce_prompt_baseline.yml"
    judge_prompt: "config/prompts/judge_prompt_baseline.yml"

  standard:
    map_prompt: "config/prompts/map_prompt.yml"
    reduce_prompt: "config/prompts/reduce_prompt.yml"
    judge_prompt: "config/prompts/judge_prompt.yml"

  hybrid:
    map_prompt: "config/prompts/map_prompt_hybrid.yml"
    reduce_prompt: "config/prompts/reduce_prompt_hybrid.yml"
    judge_prompt: "config/prompts/judge_prompt.yml"
    question_improvement_prompt: "config/prompts/question_improvement_prompt.yml"

  direct:
    map_prompt: "config/prompts/direct_prompt.yml"
    reduce_prompt: "config/prompts/direct_prompt.yml"
    judge_prompt: "config/prompts/judge_prompt.yml"

  finqa:
    map_prompt: "config/prompts/map_prompt_finqa.yml"
    reduce_prompt: "config/prompts/reduce_prompt_finqa.yml"
    judge_prompt: "config/prompts/judge_prompt.yml"

# Default prompt set if none specified
default_set: "default"
```

### Available Prompt Sets

#### Default Set
- **Purpose**: General-purpose prompts for most use cases
- **Format**: Hybrid format optimized for balanced performance
- **Use Cases**: Standard financial document analysis

#### Baseline Set
- **Purpose**: Simple, minimal prompts for comparison studies
- **Format**: Basic text prompts with minimal formatting
- **Use Cases**: Performance baseline establishment

#### Standard Set
- **Purpose**: Comprehensive prompts with detailed instructions
- **Format**: JSON-structured prompts for consistency
- **Use Cases**: High-accuracy requirements

#### Hybrid Set
- **Purpose**: Text map phase with JSON reduce phase
- **Format**: Mixed format for optimal performance/parsability balance
- **Use Cases**: Production deployments, balanced accuracy/speed

#### Direct Set
- **Purpose**: Single-pass processing without map-reduce
- **Format**: Direct question-answering prompts
- **Use Cases**: Simple queries, fast processing

#### FinQA Set
- **Purpose**: Specialized prompts for FinQA dataset
- **Format**: Numerical reasoning optimized
- **Use Cases**: Quantitative financial analysis

### Custom Prompt Configuration

#### Creating Custom Prompt Sets

1. **Add to prompt_config.yml**:
```yaml
prompt_sets:
  custom:
    map_prompt: "config/prompts/custom_map_prompt.yml"
    reduce_prompt: "config/prompts/custom_reduce_prompt.yml"
    judge_prompt: "config/prompts/judge_prompt.yml"
```

2. **Create prompt files**:
```yaml
# config/prompts/custom_map_prompt.yml
map_prompt: |
  You are analyzing financial documents to answer questions.

  Document chunk: {chunk}
  Question: {question}

  Your custom instructions here...

  Provide your analysis in the following format:
  Summary: [brief summary]
  Relevance Score: [0-10]
  Evidence: [relevant evidence]
```

#### Prompt Template Variables

Available variables in prompt templates:

**Map Phase Variables:**
- `{chunk}`: Document chunk content
- `{question}`: User question
- `{document_name}`: Name of source document
- `{chunk_index}`: Index of current chunk

**Reduce Phase Variables:**
- `{question}`: User question
- `{map_results}`: Formatted results from map phase
- `{document_name}`: Name of source document
- `{total_chunks}`: Total number of chunks processed

**Judge Phase Variables:**
- `{question}`: Original question
- `{expected_answer}`: Expected/reference answer
- `{llm_answer}`: Generated answer
- `{llm_reasoning}`: Generated reasoning
- `{evidence}`: Generated evidence

## Command Line Parameters

### Common Parameters

```bash
# Dataset and approach configuration
--dataset DATASET               # Dataset type (financebench, finqa)
--approach APPROACH             # Pipeline approach (mapreduce, truncation)
--data-path DATA_PATH           # Path to dataset file
--model_name MODEL_NAME         # LLM model name
--provider PROVIDER             # LLM provider (openai, openrouter)
--temperature TEMPERATURE       # Model temperature (0.0-2.0)
--num_samples NUM_SAMPLES       # Number of samples to process
--prompt PROMPT_SET             # Prompt set to use (default, hybrid, etc.)
```

### MapReduce-Specific Parameters

```bash
# Chunking configuration
--chunk-size CHUNK_SIZE         # Document chunk size (default: 32768)
--chunk-overlap CHUNK_OVERLAP   # Chunk overlap (default: 4096)
--format_type FORMAT_TYPE       # Output format (json, hybrid, plain_text)

# Processing configuration
--max_total_requests MAX_REQ    # Maximum concurrent requests (default: 20)
--score_threshold THRESHOLD     # Score filtering threshold (default: 5)

# PDF processing
--pdf_parser PARSER             # PDF parser (marker, pypdf, pymu, etc.)
```

### Truncation-Specific Parameters

```bash
# Truncation strategy
--strategy STRATEGY             # Truncation strategy (start, end, smart)
--context_window WINDOW         # Maximum context size (default: 128000)
--buffer BUFFER                 # Response buffer size (default: 2000)
--max-document-tokens MAX_TOKENS # Override for document token limit
```

### Rate Limiting Parameters

```bash
# Rate limiting configuration
--requests-per-minute RPM       # Requests per minute limit
--tokens-per-minute TPM         # Tokens per minute limit
--request_burst_size BURST      # Request burst size
--token_burst_size TOKEN_BURST  # Token burst size
```

### FinQA-Specific Parameters

```bash
# FinQA dataset configuration
--json_path JSON_PATH           # Path to FinQA JSON file
--doc_dir DOC_DIR               # Directory containing FinQA documents
```

### Advanced Parameters

```bash
# API key configuration
--key KEY_TYPE                  # API key selector (self, default)
--api_key API_KEY               # Direct API key specification

# Evaluation configuration
--skip_evaluation               # Skip LLM judge evaluation
--judge_model JUDGE_MODEL       # Separate model for evaluation

# Output configuration
--output_dir OUTPUT_DIR         # Custom output directory
--save_intermediate             # Save intermediate results
--verbose                       # Verbose logging
```

## Pipeline Configuration

### MapReduce Pipeline Settings

```python
# Example configuration
mapreduce_config = {
    "chunk_size": 32768,           # Characters per chunk
    "chunk_overlap": 4096,         # Overlap between chunks
    "format_type": "hybrid",       # Output format
    "score_threshold": 5,          # Minimum relevance score
    "max_total_requests": 20,      # Concurrent request limit
    "pdf_parser": "marker"         # PDF processing method
}
```

### Truncation Pipeline Settings

```python
# Example configuration
truncation_config = {
    "strategy": "start",           # Truncation strategy
    "context_window": 128000,      # Max context tokens
    "buffer": 2000,                # Buffer for response
    "max_total_requests": 20       # Concurrent request limit
}
```

### LLM Client Configuration

```python
# Rate limiting configuration
rate_limit_config = {
    "requests_per_minute": 5000,
    "tokens_per_minute": 4000000,
    "request_burst_size": 500,
    "token_burst_size": 50000
}

# LLM client configuration
llm_config = {
    "model_name": "gpt-4o-mini",
    "provider": "openai",
    "temperature": 0.0,
    "parse_json": True,
    "rate_limit_config": rate_limit_config
}
```

## Processing Parameters

### Chunk Size Guidelines

Choose chunk sizes based on your use case:

```bash
# Small chunks (fast processing, may miss context)
--chunk-size 8192 --chunk-overlap 1024

# Medium chunks (balanced performance)
--chunk-size 16384 --chunk-overlap 2048

# Large chunks (comprehensive context, slower)
--chunk-size 32768 --chunk-overlap 4096

# Extra large chunks (maximum context)
--chunk-size 65536 --chunk-overlap 8192
```

### Concurrency Guidelines

Adjust concurrency based on your API limits and performance needs:

```bash
# Conservative (for rate-limited APIs)
--max_total_requests 5

# Moderate (balanced performance)
--max_total_requests 20

# Aggressive (high-performance APIs)
--max_total_requests 50
```

### Score Threshold Guidelines

Configure score filtering for hybrid/plain_text formats:

```bash
# Permissive (includes more results)
--score_threshold 1

# Balanced (moderate filtering)
--score_threshold 5

# Strict (only highly relevant results)
--score_threshold 8
```

## PDF Parser Configuration

### Available PDF Parsers

```bash
# marker: Advanced PDF-to-markdown (recommended)
--pdf_parser marker

# pypdf: Fast Python-based parsing
--pdf_parser pypdf

# pymu: PyMuPDF with table extraction
--pdf_parser pymu

# pdfminer: Fallback text extraction
--pdf_parser pdfminer

# unstructured: Advanced layout analysis
--pdf_parser unstructured
```

### Parser-Specific Configuration

#### Marker Parser Settings

Marker parser configuration via environment variables:

```bash
# Marker cache directory
MARKER_CACHE_DIR=cache/marker

# Marker processing options
MARKER_TIMEOUT=300              # Processing timeout
MARKER_MAX_PAGES=500            # Maximum pages to process
MARKER_DPI=150                  # Image resolution
```

#### PyPDF Parser Settings

```bash
# PyPDF configuration
PYPDF_EXTRACT_IMAGES=false      # Extract images from PDFs
PYPDF_PASSWORD=""               # Default password for encrypted PDFs
```

### Parser Selection Guidelines

Choose parsers based on document characteristics:

- **marker**: Best for complex layouts, tables, mathematical content
- **pypdf**: Good for simple text documents, fastest processing
- **pymu**: Good for documents with tables and structured content
- **pdfminer**: Reliable fallback for problematic PDFs
- **unstructured**: Best for mixed content types

## Web Application Configuration

### Backend Configuration

Configuration for the FastAPI backend:

```python
# webapp/backend/config.py
class Config:
    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # API settings
    API_TITLE: str = os.getenv("API_TITLE", "MapReduce QA WebApp")
    API_VERSION: str = os.getenv("API_VERSION", "1.0.0")

    # File upload settings
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", 52428800))  # 50MB
    TEMP_DIR: str = os.getenv("TEMP_DIR", "/tmp/webapp_uploads")

    # Default processing settings
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
    DEFAULT_PROVIDER: str = os.getenv("DEFAULT_PROVIDER", "openai")
    DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", 0.0))
```

### Frontend Configuration

Configuration for the web interface is handled through the backend API and doesn't require separate configuration files.

## Logging Configuration

### Log Levels

Configure logging verbosity:

```bash
# Environment variable
LOG_LEVEL=debug    # debug, info, warning, error, critical

# Command line
python main_async.py --log-level debug
```

### Log Output

Configure log output destination:

```bash
# Log to file
LOG_FILE=logs/finmapreduce.log

# Log format
LOG_FORMAT="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### Docker Logging

Configure Docker container logging:

```yaml
# docker-compose.yml
services:
  webapp:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## Performance Tuning

### Memory Optimization

```bash
# Reduce memory usage
DEFAULT_CHUNK_SIZE=16384
DEFAULT_MAX_CONCURRENT_CHUNKS=10

# Python memory settings
PYTHONMALLOC=malloc
```

### CPU Optimization

```bash
# Optimize for CPU usage
DEFAULT_MAX_CONCURRENT_CHUNKS=25  # Based on CPU cores
WORKER_PROCESSES=4                # For production deployments
```

### Cache Configuration

```bash
# Cache directories
DOCUMENT_CACHE_DIR=cache/document_cache
MARKER_CACHE_DIR=cache/marker
PDF_CACHE_DIR=cache/pdf_cache

# Cache limits
MAX_CACHE_SIZE=10GB
CACHE_TTL=86400  # 24 hours
```

## Security Configuration

### API Key Security

```bash
# Use separate keys for different environments
OPENAI_API_KEY_DEV=sk-dev-key
OPENAI_API_KEY_PROD=sk-prod-key

# Key rotation
OPENAI_API_KEY_BACKUP=sk-backup-key
```

### File Upload Security

```bash
# File type restrictions
ALLOWED_FILE_TYPES=pdf,txt,md,docx
BLOCKED_FILE_TYPES=exe,bat,sh,py

# Scan uploaded files
VIRUS_SCAN_ENABLED=true
SCAN_TIMEOUT=30
```

### CORS Configuration

```python
# webapp/backend/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## Troubleshooting Configuration

### Common Configuration Issues

1. **API Key Not Found**:
   ```bash
   # Check environment variable
   echo $OPENAI_API_KEY

   # Verify .env file loading
   python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
   ```

2. **Invalid Model Names**:
   ```bash
   # List available models
   python -c "from openai import OpenAI; print(OpenAI().models.list())"
   ```

3. **Rate Limiting Issues**:
   ```bash
   # Test rate limits
   python -c "
   from src.llm.async_llm_client import AsyncLLMClient
   client = AsyncLLMClient('gpt-4o-mini')
   print(client.rate_limiter.get_stats())
   "
   ```

### Configuration Validation

Use the built-in configuration validation:

```python
# Validate configuration
python -c "
from webapp.backend.config import Config
config = Config()
print('Configuration valid:', config.validate())
"
```