# Usage Examples

This document provides comprehensive examples of using FinMapReduce across different scenarios and use cases.

## Command Line Usage

### Basic FinanceBench Processing

#### Hybrid MapReduce (Recommended)

```bash
# Standard hybrid approach with balanced performance
python main_async.py \
  --dataset financebench \
  --approach mapreduce \
  --format_type hybrid \
  --data-path data/financebench/financebench_open_source.jsonl \
  --model_name gpt-4o-mini \
  --num_samples 10
```

#### JSON MapReduce

```bash
# Structured JSON processing throughout
python main_async.py \
  --dataset financebench \
  --approach mapreduce \
  --format_type json \
  --data-path data/financebench/financebench_open_source.jsonl \
  --model_name gpt-4o-mini \
  --chunk-size 32768 \
  --chunk-overlap 4096 \
  --num_samples 10
```

#### Plain Text MapReduce

```bash
# Text-based processing for simpler prompts
python main_async.py \
  --dataset financebench \
  --approach mapreduce \
  --format_type plain_text \
  --data-path data/financebench/financebench_open_source.jsonl \
  --model_name gpt-4o-mini \
  --score_threshold 7 \
  --num_samples 10
```

### Truncation Pipeline Examples

#### Start Truncation Strategy

```bash
# Keep document beginning, good for executive summaries
python main_async.py \
  --dataset financebench \
  --approach truncation \
  --strategy start \
  --context_window 128000 \
  --buffer 2000 \
  --data-path data/financebench/financebench_open_source.jsonl \
  --model_name gpt-4o-mini \
  --num_samples 10
```

#### End Truncation Strategy

```bash
# Keep document end, good for conclusions and recent data
python main_async.py \
  --dataset financebench \
  --approach truncation \
  --strategy end \
  --context_window 128000 \
  --buffer 2000 \
  --data-path data/financebench/financebench_open_source.jsonl \
  --model_name gpt-4o-mini \
  --num_samples 10
```

### FinQA Processing Examples

#### FinQA MapReduce

```bash
# FinQA with MapReduce approach
python main_async.py \
  --dataset finqa \
  --approach mapreduce \
  --format_type hybrid \
  --data-path data/finqa/finqa_subset_test.json \
  --doc_dir ../edgartools_finqa \
  --model_name gpt-4o-mini \
  --prompt finqa \
  --num_samples 10
```

#### FinQA Truncation

```bash
# FinQA with truncation approach
python main_async.py \
  --dataset finqa \
  --approach truncation \
  --strategy start \
  --data-path data/finqa/finqa_subset_test.json \
  --doc_dir ../edgartools_finqa \
  --model_name gpt-4o-mini \
  --prompt finqa \
  --num_samples 5
```

### Advanced Command Line Options

#### Custom Rate Limiting

```bash
# Conservative rate limiting for free APIs
python main_async.py \
  --dataset financebench \
  --approach mapreduce \
  --format_type hybrid \
  --data-path data/financebench/financebench_open_source.jsonl \
  --model_name gpt-4o-mini \
  --requests-per-minute 100 \
  --tokens-per-minute 100000 \
  --request_burst_size 10 \
  --max_total_requests 5 \
  --num_samples 10
```

#### Custom Chunking Configuration

```bash
# Large chunks for comprehensive analysis
python main_async.py \
  --dataset financebench \
  --approach mapreduce \
  --format_type hybrid \
  --data-path data/financebench/financebench_open_source.jsonl \
  --model_name gpt-4o-mini \
  --chunk-size 65536 \
  --chunk-overlap 8192 \
  --score_threshold 3 \
  --num_samples 10
```

#### Alternative PDF Parsers

```bash
# Using PyPDF for faster processing
python main_async.py \
  --dataset financebench \
  --approach mapreduce \
  --format_type hybrid \
  --data-path data/financebench/financebench_open_source.jsonl \
  --model_name gpt-4o-mini \
  --pdf_parser pypdf \
  --num_samples 10

# Using marker for best quality (default)
python main_async.py \
  --dataset financebench \
  --approach mapreduce \
  --format_type hybrid \
  --data-path data/financebench/financebench_open_source.jsonl \
  --model_name gpt-4o-mini \
  --pdf_parser marker \
  --num_samples 10
```

#### Different LLM Providers

```bash
# Using OpenRouter
python main_async.py \
  --dataset financebench \
  --approach mapreduce \
  --format_type hybrid \
  --data-path data/financebench/financebench_open_source.jsonl \
  --model_name deepseek/deepseek-r1-0528:free \
  --provider openrouter \
  --num_samples 10

# Using alternative API key
python main_async.py \
  --dataset financebench \
  --approach mapreduce \
  --format_type hybrid \
  --data-path data/financebench/financebench_open_source.jsonl \
  --model_name gpt-4o-mini \
  --key self \
  --num_samples 10
```

## Python API Examples

### Basic MapReduce Processing

```python
import asyncio
from src.core.factory import PipelineFactory
from src.llm.async_llm_client import create_async_rate_limited_llm, RateLimitConfig
from src.llm.utils import load_prompt_set

async def basic_mapreduce_example():
    """Basic MapReduce processing example."""

    # Create LLM client with rate limiting
    llm = create_async_rate_limited_llm(
        model_name="gpt-4o-mini",
        provider="openai",
        temperature=0.0,
        parse_json=True,
        rate_limit_config=RateLimitConfig(
            requests_per_minute=1000,
            tokens_per_minute=2000000,
            request_burst_size=100
        )
    )

    # Load prompts
    prompts = load_prompt_set("hybrid")

    # Create pipeline
    pipeline = PipelineFactory.create_pipeline(
        dataset='financebench',
        format_type='hybrid',
        llm=llm,
        prompts_dict=prompts,
        chunk_size=32768,
        chunk_overlap=4096
    )

    # Process dataset
    results = await pipeline.process_dataset_async(
        data_path="data/financebench/financebench_open_source.jsonl",
        model_name="gpt-4o-mini",
        num_samples=10
    )

    # Print summary statistics
    print(f"Processed {results['num_samples']} samples")
    print(f"Total time: {results['time_taken']:.1f} seconds")
    print(f"Total tokens: {results['token_usage_summary']['total_input_tokens']:,}")

    return results

# Run the example
if __name__ == "__main__":
    results = asyncio.run(basic_mapreduce_example())
```

### Advanced MapReduce with Custom Configuration

```python
import asyncio
from src.core.factory import PipelineFactory
from src.llm.async_llm_client import create_async_rate_limited_llm, RateLimitConfig
from src.llm.utils import load_prompt_set
from src.evaluation.async_evaluation import AsyncLLMJudgeEvaluator

async def advanced_mapreduce_example():
    """Advanced MapReduce with custom configuration and evaluation."""

    # Create separate LLM clients for processing and evaluation
    process_llm = create_async_rate_limited_llm(
        model_name="gpt-4o-mini",
        provider="openai",
        temperature=0.0,
        parse_json=True,
        rate_limit_config=RateLimitConfig(
            requests_per_minute=2000,
            tokens_per_minute=4000000,
            request_burst_size=200
        )
    )

    judge_llm = create_async_rate_limited_llm(
        model_name="gpt-4o",
        provider="openai",
        temperature=0.0,
        parse_json=True
    )

    # Load custom prompt set
    prompts = load_prompt_set("hybrid")

    # Create pipeline with custom configuration
    pipeline = PipelineFactory.create_pipeline(
        dataset='financebench',
        format_type='hybrid',
        llm=process_llm,
        prompts_dict=prompts,
        chunk_size=65536,  # Larger chunks
        chunk_overlap=8192,
        max_total_requests=30  # Higher concurrency
    )

    # Process dataset with evaluation
    results = await pipeline.process_dataset_async(
        data_path="data/financebench/financebench_open_source.jsonl",
        model_name="gpt-4o-mini",
        num_samples=50,
        score_threshold=7,  # Higher threshold
        judge_llm=judge_llm
    )

    # Print detailed statistics
    print("=== Processing Results ===")
    print(f"Samples processed: {results['num_samples']}")
    print(f"Total processing time: {results['time_taken']:.1f} seconds")
    print(f"Document loading time: {results['document_loading_time']:.1f} seconds")

    print("\n=== Token Usage ===")
    token_summary = results['token_usage_summary']
    print(f"Total input tokens: {token_summary['total_input_tokens']:,}")
    print(f"Total output tokens: {token_summary['total_output_tokens']:,}")
    print(f"Cache read tokens: {token_summary.get('total_cache_read_tokens', 0):,}")

    print("\n=== Phase Breakdown ===")
    phase_totals = results.get('phase_token_totals', {})
    for phase, tokens in phase_totals.items():
        print(f"{phase}: {tokens['input_tokens']:,} in, {tokens['output_tokens']:,} out")

    # Print evaluation results
    if 'evaluations' in results:
        print("\n=== Evaluation Results ===")
        for model, eval_data in results['evaluations'].items():
            print(f"{model}: {eval_data['accuracy']:.2%} accuracy")
            print(f"Distribution: {eval_data['judgment_distribution']}")

    return results

# Run the advanced example
if __name__ == "__main__":
    results = asyncio.run(advanced_mapreduce_example())
```

### Truncation Pipeline Example

```python
import asyncio
from src.core.factory import PipelineFactory
from src.llm.async_llm_client import create_async_rate_limited_llm
from src.llm.utils import load_prompt_set

async def truncation_example():
    """Truncation pipeline example with multiple strategies."""

    # Create LLM client
    llm = create_async_rate_limited_llm(
        model_name="gpt-4o-mini",
        provider="openai",
        temperature=0.0
    )

    # Load prompts
    prompts = load_prompt_set("standard")

    # Test different truncation strategies
    strategies = ["start", "end"]
    results = {}

    for strategy in strategies:
        print(f"\n=== Testing {strategy} strategy ===")

        # Create truncation pipeline
        pipeline = PipelineFactory.create_truncation_pipeline(
            dataset='financebench',
            strategy=strategy,
            context_window=128000,
            buffer=2000,
            llm=llm,
            prompts_dict=prompts,
            max_total_requests=10
        )

        # Process subset
        strategy_results = await pipeline.process_dataset_async(
            data_path="data/financebench/financebench_open_source.jsonl",
            model_name="gpt-4o-mini",
            num_samples=5
        )

        results[strategy] = strategy_results

        # Print strategy-specific statistics
        print(f"Time taken: {strategy_results['time_taken']:.1f} seconds")
        print(f"Avg retention rate: {strategy_results.get('avg_retention_rate', 0):.2%}")
        print(f"Total tokens: {strategy_results['token_usage_summary']['total_input_tokens']:,}")

    return results

# Run truncation example
if __name__ == "__main__":
    results = asyncio.run(truncation_example())
```

### Custom Dataset Processing

```python
import asyncio
from src.core.factory import PipelineFactory
from src.loaders.webapp_loader import WebappDatasetLoader
from src.llm.async_llm_client import create_async_rate_limited_llm
from src.llm.utils import load_prompt_set

async def custom_document_example():
    """Process a custom document with webapp loader."""

    # Create LLM client
    llm = create_async_rate_limited_llm(
        model_name="gpt-4o-mini",
        provider="openai",
        temperature=0.0,
        parse_json=True
    )

    # Load prompts
    prompts = load_prompt_set("hybrid")

    # Create webapp loader for custom document
    custom_loader = WebappDatasetLoader(
        file_path="/path/to/your/document.pdf",
        question="What were the main financial highlights in 2023?"
    )

    # Create pipeline with custom loader
    pipeline = PipelineFactory.create_pipeline(
        dataset='webapp',  # This will use the provided loader
        format_type='hybrid',
        llm=llm,
        prompts_dict=prompts,
        chunk_size=32768,
        chunk_overlap=4096,
        dataset_loader=custom_loader  # Override default loader
    )

    # Process the custom document
    results = await pipeline.process_dataset_async(
        data_path="dummy_path",  # Not used with webapp loader
        model_name="gpt-4o-mini",
        num_samples=1
    )

    # Extract answer
    if results['qa_data']:
        qa_result = results['qa_data'][0]
        print("=== Custom Document Analysis ===")
        print(f"Question: {qa_result['question']}")
        print(f"Answer: {qa_result['llm_answer']}")
        print(f"Reasoning: {qa_result['llm_reasoning']}")
        print(f"Evidence: {qa_result['llm_evidence']}")

    return results

# Run custom document example
if __name__ == "__main__":
    results = asyncio.run(custom_document_example())
```

## Web Application Usage

### Basic File Upload and Processing

1. **Start the web application**:
```bash
# Local development
cd webapp/backend
python main.py

# Or with Docker
docker compose up -d
```

2. **Access the web interface**:
```
http://localhost:8000
```

3. **Upload and process a document**:
   - Drag and drop a PDF file into the upload area
   - Enter your question in the text field
   - Click "Preview Document" to see content preview
   - Adjust processing parameters if needed
   - Click "Get Answer" to process

### Web API Usage

#### Document Processing Endpoint

```python
import requests
import json

# Prepare file and data
files = {
    'file': ('document.pdf', open('document.pdf', 'rb'), 'application/pdf')
}

data = {
    'question': 'What was the revenue growth in 2023?',
    'model_name': 'gpt-4o-mini',
    'provider': 'openai',
    'pipeline_type': 'mapreduce',
    'format_type': 'hybrid',
    'temperature': 0.0,
    'chunk_size': 32768,
    'chunk_overlap': 4096,
    'score_threshold': 5
}

# Send request
response = requests.post(
    'http://localhost:8000/api/answer',
    files=files,
    data=data
)

# Process response
if response.status_code == 200:
    result = response.json()
    print("Answer:", result['answer'])
    print("Reasoning:", result['reasoning'])
    print("Evidence:", result['evidence'])
    print("Processing time:", result['timing_stats']['total_time'])
else:
    print("Error:", response.json()['detail'])
```

#### Document Preview Endpoint

```python
import requests

# Preview document before processing
files = {
    'file': ('document.pdf', open('document.pdf', 'rb'), 'application/pdf')
}

response = requests.post(
    'http://localhost:8000/api/preview',
    files=files
)

if response.status_code == 200:
    preview = response.json()
    print("File size:", preview['file_size'])
    print("File type:", preview['file_type'])
    print("Estimated tokens:", preview['estimated_tokens'])
    print("Content preview:", preview['content_preview'][:500] + "...")
```

#### Available Models Endpoint

```python
import requests

# Get available models and configurations
response = requests.get('http://localhost:8000/api/models')

if response.status_code == 200:
    config = response.json()
    print("Available models:", config['models'])
    print("Available formats:", config['formats'])
    print("Pipeline types:", config['pipeline_types'])
    print("Default parameters:", config['defaults'])
```

## Batch Processing Examples

### Processing Multiple Datasets

```bash
#!/bin/bash
# Batch process multiple datasets

datasets=(
    "data/financebench/sample1.jsonl"
    "data/financebench/sample2.jsonl"
    "data/financebench/sample3.jsonl"
)

models=("gpt-4o-mini" "gpt-4o")

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        echo "Processing $dataset with $model"

        python main_async.py \
            --dataset financebench \
            --approach mapreduce \
            --format_type hybrid \
            --data-path "$dataset" \
            --model_name "$model" \
            --num_samples 20 \
            --output_dir "results/batch_$(basename $dataset .jsonl)_$(echo $model | tr '/' '_')"

        echo "Completed $dataset with $model"
        sleep 60  # Rate limiting pause
    done
done
```

### Parameter Sweep Example

```python
import asyncio
import itertools
from src.core.factory import PipelineFactory
from src.llm.async_llm_client import create_async_rate_limited_llm
from src.llm.utils import load_prompt_set

async def parameter_sweep_example():
    """Run parameter sweep across different configurations."""

    # Define parameter ranges
    chunk_sizes = [16384, 32768, 65536]
    score_thresholds = [3, 5, 7]
    format_types = ['hybrid', 'json']

    # Create LLM client
    llm = create_async_rate_limited_llm(
        model_name="gpt-4o-mini",
        provider="openai",
        temperature=0.0,
        parse_json=True
    )

    prompts = load_prompt_set("hybrid")

    results = {}

    # Iterate through all combinations
    for chunk_size, threshold, format_type in itertools.product(
        chunk_sizes, score_thresholds, format_types
    ):
        config_name = f"chunk{chunk_size}_thresh{threshold}_{format_type}"
        print(f"\n=== Running configuration: {config_name} ===")

        # Create pipeline with current configuration
        pipeline = PipelineFactory.create_pipeline(
            dataset='financebench',
            format_type=format_type,
            llm=llm,
            prompts_dict=prompts,
            chunk_size=chunk_size,
            chunk_overlap=chunk_size // 8,  # 12.5% overlap
            max_total_requests=10
        )

        # Process small sample
        config_results = await pipeline.process_dataset_async(
            data_path="data/financebench/financebench_open_source.jsonl",
            model_name="gpt-4o-mini",
            num_samples=5,
            score_threshold=threshold
        )

        # Store key metrics
        results[config_name] = {
            'accuracy': config_results.get('evaluations', {}).get('gpt-4o-mini', {}).get('accuracy', 0),
            'time_taken': config_results['time_taken'],
            'total_tokens': config_results['token_usage_summary']['total_input_tokens'],
            'avg_chunks_per_qa': config_results.get('avg_chunks_per_qa', 0)
        }

        print(f"Accuracy: {results[config_name]['accuracy']:.2%}")
        print(f"Time: {results[config_name]['time_taken']:.1f}s")
        print(f"Tokens: {results[config_name]['total_tokens']:,}")

    # Print summary comparison
    print("\n=== Parameter Sweep Results ===")
    print("Configuration | Accuracy | Time | Tokens | Chunks/QA")
    print("-" * 60)

    for config_name, metrics in results.items():
        print(f"{config_name:20} | {metrics['accuracy']:6.2%} | {metrics['time_taken']:4.1f}s | {metrics['total_tokens']:6,} | {metrics['avg_chunks_per_qa']:4.1f}")

    return results

# Run parameter sweep
if __name__ == "__main__":
    results = asyncio.run(parameter_sweep_example())
```

## Performance Testing Examples

### Load Testing

```python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from src.core.factory import PipelineFactory
from src.llm.async_llm_client import create_async_rate_limited_llm, RateLimitConfig
from src.llm.utils import load_prompt_set

async def load_test_example():
    """Load testing with concurrent pipeline instances."""

    # Create multiple LLM clients to simulate load
    llm_clients = []
    for i in range(5):  # 5 concurrent clients
        client = create_async_rate_limited_llm(
            model_name="gpt-4o-mini",
            provider="openai",
            temperature=0.0,
            parse_json=True,
            rate_limit_config=RateLimitConfig(
                requests_per_minute=1000,  # Distributed across clients
                tokens_per_minute=800000,
                request_burst_size=50
            )
        )
        llm_clients.append(client)

    prompts = load_prompt_set("hybrid")

    async def run_pipeline_instance(client_id, llm):
        """Run single pipeline instance."""
        print(f"Starting pipeline instance {client_id}")

        pipeline = PipelineFactory.create_pipeline(
            dataset='financebench',
            format_type='hybrid',
            llm=llm,
            prompts_dict=prompts,
            chunk_size=32768,
            chunk_overlap=4096,
            max_total_requests=5  # Lower per instance
        )

        start_time = time.time()
        results = await pipeline.process_dataset_async(
            data_path="data/financebench/financebench_open_source.jsonl",
            model_name="gpt-4o-mini",
            num_samples=3  # Small sample per instance
        )
        end_time = time.time()

        print(f"Pipeline instance {client_id} completed in {end_time - start_time:.1f}s")
        return results

    # Run all instances concurrently
    print("Starting load test with 5 concurrent pipeline instances")
    start_time = time.time()

    tasks = [
        run_pipeline_instance(i, client)
        for i, client in enumerate(llm_clients)
    ]

    all_results = await asyncio.gather(*tasks)

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate aggregate statistics
    total_samples = sum(r['num_samples'] for r in all_results)
    total_tokens = sum(r['token_usage_summary']['total_input_tokens'] for r in all_results)

    print(f"\n=== Load Test Results ===")
    print(f"Total time: {total_time:.1f}s")
    print(f"Total samples processed: {total_samples}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Samples per second: {total_samples / total_time:.2f}")
    print(f"Tokens per second: {total_tokens / total_time:,.0f}")

    return all_results

# Run load test
if __name__ == "__main__":
    results = asyncio.run(load_test_example())
```

### Memory Profiling Example

```python
import asyncio
import psutil
import os
from src.core.factory import PipelineFactory
from src.llm.async_llm_client import create_async_rate_limited_llm
from src.llm.utils import load_prompt_set

async def memory_profile_example():
    """Profile memory usage during processing."""

    process = psutil.Process(os.getpid())

    def get_memory_usage():
        """Get current memory usage in MB."""
        return process.memory_info().rss / 1024 / 1024

    print(f"Initial memory usage: {get_memory_usage():.1f} MB")

    # Create LLM client
    llm = create_async_rate_limited_llm(
        model_name="gpt-4o-mini",
        provider="openai",
        temperature=0.0,
        parse_json=True
    )

    print(f"Memory after LLM client creation: {get_memory_usage():.1f} MB")

    # Load prompts
    prompts = load_prompt_set("hybrid")
    print(f"Memory after loading prompts: {get_memory_usage():.1f} MB")

    # Test different chunk sizes for memory impact
    chunk_sizes = [8192, 16384, 32768, 65536]

    for chunk_size in chunk_sizes:
        print(f"\n=== Testing chunk size: {chunk_size} ===")

        # Create pipeline
        pipeline = PipelineFactory.create_pipeline(
            dataset='financebench',
            format_type='hybrid',
            llm=llm,
            prompts_dict=prompts,
            chunk_size=chunk_size,
            chunk_overlap=chunk_size // 8,
            max_total_requests=5
        )

        memory_before = get_memory_usage()
        print(f"Memory before processing: {memory_before:.1f} MB")

        # Process sample
        results = await pipeline.process_dataset_async(
            data_path="data/financebench/financebench_open_source.jsonl",
            model_name="gpt-4o-mini",
            num_samples=5
        )

        memory_after = get_memory_usage()
        memory_delta = memory_after - memory_before

        print(f"Memory after processing: {memory_after:.1f} MB")
        print(f"Memory delta: {memory_delta:+.1f} MB")
        print(f"Processing time: {results['time_taken']:.1f}s")
        print(f"Avg chunks per QA: {results.get('avg_chunks_per_qa', 0):.1f}")

        # Clear some variables to help with garbage collection
        del pipeline
        del results

# Run memory profiling
if __name__ == "__main__":
    asyncio.run(memory_profile_example())
```

## Integration Examples

### Jupyter Notebook Integration

```python
# Cell 1: Setup and imports
import asyncio
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, HTML

from src.core.factory import PipelineFactory
from src.llm.async_llm_client import create_async_rate_limited_llm
from src.llm.utils import load_prompt_set

# Cell 2: Create and run pipeline
async def notebook_analysis():
    """Analysis for Jupyter notebook."""

    llm = create_async_rate_limited_llm(
        model_name="gpt-4o-mini",
        provider="openai",
        temperature=0.0,
        parse_json=True
    )

    prompts = load_prompt_set("hybrid")

    pipeline = PipelineFactory.create_pipeline(
        dataset='financebench',
        format_type='hybrid',
        llm=llm,
        prompts_dict=prompts,
        chunk_size=32768,
        chunk_overlap=4096
    )

    results = await pipeline.process_dataset_async(
        data_path="data/financebench/financebench_open_source.jsonl",
        model_name="gpt-4o-mini",
        num_samples=20
    )

    return results

# Run the analysis
results = await notebook_analysis()

# Cell 3: Visualize results
def visualize_results(results):
    """Create visualizations of results."""

    # Extract timing data
    qa_data = results['qa_data']
    processing_times = [qa.get('processing_time', 0) for qa in qa_data]
    token_counts = [qa.get('token_stats', {}).get('total_input_tokens', 0) for qa in qa_data]

    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Processing time distribution
    ax1.hist(processing_times, bins=10, alpha=0.7)
    ax1.set_xlabel('Processing Time (seconds)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Processing Time Distribution')

    # Token usage
    ax2.scatter(token_counts, processing_times, alpha=0.7)
    ax2.set_xlabel('Input Tokens')
    ax2.set_ylabel('Processing Time (seconds)')
    ax2.set_title('Tokens vs Processing Time')

    plt.tight_layout()
    plt.show()

    # Summary table
    summary_df = pd.DataFrame({
        'Metric': ['Total Samples', 'Total Time', 'Avg Time/Sample', 'Total Tokens', 'Avg Tokens/Sample'],
        'Value': [
            results['num_samples'],
            f"{results['time_taken']:.1f}s",
            f"{results['time_taken']/results['num_samples']:.1f}s",
            f"{results['token_usage_summary']['total_input_tokens']:,}",
            f"{results['token_usage_summary']['total_input_tokens']/results['num_samples']:,.0f}"
        ]
    })

    display(HTML(summary_df.to_html(index=False)))

# Cell 4: Run visualization
visualize_results(results)
```

### FastAPI Integration

```python
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import asyncio
from typing import Optional

from src.core.factory import PipelineFactory
from src.loaders.webapp_loader import WebappDatasetLoader
from src.llm.async_llm_client import create_async_rate_limited_llm
from src.llm.utils import load_prompt_set

app = FastAPI(title="FinMapReduce API", version="1.0.0")

# Global pipeline cache
pipeline_cache = {}

class ProcessingRequest(BaseModel):
    question: str
    model_name: str = "gpt-4o-mini"
    format_type: str = "hybrid"
    chunk_size: int = 32768
    chunk_overlap: int = 4096
    temperature: float = 0.0

class ProcessingResponse(BaseModel):
    answer: str
    reasoning: str
    evidence: list
    processing_time: float
    token_usage: dict

@app.post("/process-document", response_model=ProcessingResponse)
async def process_document(
    file: UploadFile = File(...),
    question: str = Form(...),
    model_name: str = Form("gpt-4o-mini"),
    format_type: str = Form("hybrid"),
    chunk_size: int = Form(32768),
    chunk_overlap: int = Form(4096),
    temperature: float = Form(0.0)
):
    """Process uploaded document with MapReduce pipeline."""

    try:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Create pipeline key for caching
        pipeline_key = f"{model_name}_{format_type}_{chunk_size}_{chunk_overlap}_{temperature}"

        # Get or create pipeline
        if pipeline_key not in pipeline_cache:
            llm = create_async_rate_limited_llm(
                model_name=model_name,
                provider="openai",
                temperature=temperature,
                parse_json=True
            )

            prompts = load_prompt_set("hybrid")

            # Create custom loader
            loader = WebappDatasetLoader(temp_path, question)

            pipeline = PipelineFactory.create_pipeline(
                dataset='webapp',
                format_type=format_type,
                llm=llm,
                prompts_dict=prompts,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                dataset_loader=loader
            )

            pipeline_cache[pipeline_key] = pipeline
        else:
            # Update loader for cached pipeline
            pipeline = pipeline_cache[pipeline_key]
            pipeline.dataset_loader = WebappDatasetLoader(temp_path, question)

        # Process document
        results = await pipeline.process_dataset_async(
            data_path="dummy_path",
            model_name=model_name,
            num_samples=1
        )

        # Extract result
        if results['qa_data']:
            qa_result = results['qa_data'][0]
            return ProcessingResponse(
                answer=qa_result['llm_answer'],
                reasoning=qa_result['llm_reasoning'],
                evidence=qa_result['llm_evidence'],
                processing_time=results['time_taken'],
                token_usage=results['token_usage_summary']
            )
        else:
            raise HTTPException(status_code=500, detail="No results generated")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temporary file
        import os
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "cache_size": len(pipeline_cache)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

These examples demonstrate the flexibility and power of the FinMapReduce system across different use cases and integration scenarios.