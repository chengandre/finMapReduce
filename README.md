# FinMapReduce: AI Question Answering for Long Financial Documents

FinMapReduce is a research-driven system for answering questions about long, complex financial documents like SEC 10-K and 10-Q filings. Using a **MapReduce paradigm**, it processes documents that exceed LLM context limits by analyzing chunks in parallel and synthesizing results.

**Key Innovation**: Unlike RAG systems, FinMapReduce guarantees full document coverage through systematic parallel processing, making it ideal for comprehensive financial analysis.

## Quick Start

### 🚀 Web Interface (Recommended)

**Docker (Easiest)**:
```bash
git clone https://github.com/chengandre/finMapReduce.git
cd finMapReduce
cp .env.example .env  # Add your API keys
docker compose up -d
open http://localhost:8000
```

**Local Development**:
```bash
cd webapp/backend && python main.py
open http://localhost:8000
```

### 🖥️ Command Line

```bash
# FinanceBench with MapReduce
python main_async.py \
  --dataset financebench \
  --approach mapreduce \
  --format_type hybrid \
  --data-path data/financebench_open_source.jsonl \
  --model_name gpt-4o-mini \
  --num_samples 10

# Quick processing with truncation
python main_async.py \
  --dataset financebench \
  --approach truncation \
  --strategy start \
  --data-path data/financebench_open_source.jsonl \
  --model_name gpt-4o-mini \
  --num_samples 5
```

### 🐍 Python API

```python
import asyncio
from src.core.factory import PipelineFactory
from src.llm.async_llm_client import create_async_rate_limited_llm
from src.llm.utils import load_prompt_set

async def analyze_document():
    llm = create_async_rate_limited_llm("gpt-4o-mini")
    prompts = load_prompt_set("hybrid")

    pipeline = PipelineFactory.create_pipeline(
        dataset='financebench',
        format_type='hybrid',
        llm=llm,
        prompts_dict=prompts
    )

    results = await pipeline.process_dataset_async(
        data_path="your_data.jsonl",
        model_name="gpt-4o-mini",
        num_samples=10
    )
    return results

results = asyncio.run(analyze_document())
```

## Installation

### Prerequisites
- Python 3.10.18+
- API keys: OpenAI or OpenRouter

### Setup
```bash
git clone https://github.com/chengandre/finMapReduce.git
cd finMapReduce
pip install -r requirements.txt
cp .env.example .env  # Configure your API keys
```

### Environment Configuration
```bash
# Required (at least one)
OPENAI_API_KEY=your_openai_key_here
OPENROUTER_API_KEY=your_openrouter_key_here

# Optional optimizations
DEFAULT_MODEL=gpt-4o-mini
DEFAULT_CHUNK_SIZE=32768
DEFAULT_FORMAT_TYPE=hybrid
```

## How It Works

### 🗺️ MapReduce Pipeline
1. **Document Loading**: PDF processed with advanced parsing (marker, PyPDF)
2. **Map Phase**: Document chunks analyzed in parallel for relevance and evidence
3. **Reduce Phase**: Results synthesized into comprehensive answer with reasoning
4. **Evaluation**: Optional LLM judge assessment

### ⚡ Truncation Pipeline
1. **Document Loading**: Full document loaded and truncated to fit context
2. **Processing**: Single LLM call with intelligent truncation strategies
3. **Strategies**: `start` (keep beginning), `end` (keep conclusion)

### 🎯 Key Features
- **Full Document Coverage**: Unlike RAG, processes entire document systematically
- **Multiple Datasets**: FinanceBench, FinQA, custom uploads
- **Advanced PDF Processing**: Multiple parsers including marker CLI
- **Rate Limiting**: Built-in token bucket algorithm
- **Async Processing**: High-performance concurrent execution
- **Web Interface**: Drag-and-drop file upload with real-time preview

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   BasePipeline  │    │ DatasetLoader   │    │ OutputFormatter │
│   (Abstract)    │    │   (Abstract)    │    │   (Abstract)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                       ▲                       ▲
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│MapReducePipeline│    │FinanceBenchLoader│   │ HybridFormatter │
│TruncationPipeline│   │   FinQALoader   │    │  JSONFormatter  │
└─────────────────┘    │  WebappLoader   │    │PlainTextFormatter│
                       └─────────────────┘    └─────────────────┘
```

**Design Principles**: Composition over inheritance, factory pattern, async-first architecture, extensive rate limiting and error handling.

## Project Structure

```
finMapReduce/
├── src/                          # Core source code
│   ├── core/                     # Pipeline architecture
│   ├── loaders/                  # Dataset loaders (FinanceBench, FinQA)
│   ├── formatters/               # Output formatters (JSON, Hybrid, Text)
│   ├── llm/                      # LLM integration with rate limiting
│   ├── evaluation/               # LLM judge evaluation system
│   └── utils/                    # Document processing utilities
├── webapp/                       # Web application
│   ├── backend/                  # FastAPI server
│   └── frontend/                 # Static HTML/CSS/JavaScript
├── config/prompts/               # YAML prompt configurations
├── results/                      # Processing output
├── cache/                        # Document and parser cache
└── docs/                         # Comprehensive documentation
```

## Use Cases & Performance

### 📊 FinanceBench Results
- **Accuracy**: 85%+ on financial QA benchmarks
- **Coverage**: Processes 100+ page 10-K filings completely
- **Speed**: ~30 seconds per question (MapReduce), ~10 seconds (Truncation)
- **Scalability**: Handles 50+ concurrent document chunks

### 🎯 Ideal For
- **Financial Analysis**: Revenue trends, performance metrics, risk assessment
- **Compliance**: Regulatory requirement verification, policy analysis
- **Research**: Academic studies requiring comprehensive document coverage
- **Due Diligence**: M&A analysis, investment research

### 🚫 Not Ideal For
- **Simple Fact Lookup**: Use RAG for basic question answering
- **Real-time Applications**: Processing takes 10-60 seconds per document
- **Small Documents**: Overhead not justified for short texts

## Comprehensive Documentation

| Topic | Description | Link |
|-------|-------------|------|
| **🏗️ Architecture** | Technical architecture, design patterns, components | [docs/architecture.md](docs/architecture.md) |
| **📚 API Reference** | Complete class references, method signatures | [docs/api-reference.md](docs/api-reference.md) |
| **🚀 Deployment** | Docker, production, scaling, monitoring | [docs/deployment.md](docs/deployment.md) |
| **⚙️ Configuration** | Environment variables, prompts, parameters | [docs/configuration.md](docs/configuration.md) |
| **💡 Examples** | Usage examples, parameter tuning, integrations | [docs/examples.md](docs/examples.md) |
| **🔧 Troubleshooting** | Common issues, debugging, performance optimization | [docs/troubleshooting.md](docs/troubleshooting.md) |
| **👩‍💻 Development** | Adding datasets, formatters, extending pipelines | [docs/development.md](docs/development.md) |
| **🌐 Web Application** | Complete webapp documentation, API endpoints | [docs/web-application.md](docs/web-application.md) |

## Common Commands

```bash
# Process different datasets
python main_async.py --dataset financebench --approach mapreduce --num_samples 10
python main_async.py --dataset finqa --approach truncation --num_samples 5

# Customize processing
python main_async.py --model_name gpt-4o --chunk-size 65536 --score_threshold 7
python main_async.py --provider openrouter --model_name deepseek/deepseek-r1-0528:free

# Run evaluation
python main_async.py --dataset financebench --num_samples 50 # Includes LLM judge

# Docker deployment
docker compose up -d                    # Start web application
docker compose logs -f                  # View logs
docker compose down -v                  # Stop and cleanup
```

## Results Format

```json
{
  "configuration": {
    "dataset": "financebench_hybrid",
    "model_name": "gpt-4o-mini",
    "approach": "mapreduce"
  },
  "execution_time": "2025-01-15T10:30:00",
  "time_taken": 45.2,
  "num_samples": 10,
  "token_usage_summary": {
    "total_input_tokens": 125000,
    "total_output_tokens": 8500
  },
  "evaluations": {
    "gpt-4o-mini": {
      "accuracy": 0.85,
      "judgment_distribution": {
        "correct": 8,
        "coherent": 1,
        "incorrect": 1
      }
    }
  }
}
```

## Contributing

We welcome contributions! See [docs/development.md](docs/development.md) for:
- Adding new datasets
- Creating custom output formatters
- Extending pipeline architectures
- Improving evaluation methods