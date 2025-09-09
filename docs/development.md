# Development Guide

This guide covers extending and contributing to the FinMapReduce system, including adding new datasets, output formats, and pipeline types.

## Development Setup

### Prerequisites

- Python 3.10.18+
- Git for version control
- Virtual environment (recommended)
- API keys for testing

### Local Development Environment

1. **Clone and Setup**:
```bash
git clone https://github.com/chengandre/finMapReduce.git
cd finMapReduce

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If exists
```

2. **Configuration for Development**:
```bash
# Copy environment file
cp .env.example .env.dev

# Edit with development settings
nano .env.dev
```

3. **Development Environment Variables**:
```bash
# .env.dev
OPENAI_API_KEY=your_dev_key
DEBUG=true
LOG_LEVEL=debug
DEFAULT_MODEL=gpt-4o-mini  # Use cheaper model for development
MAX_FILE_SIZE=10485760     # 10MB for development
```

## Project Structure

Understanding the project architecture is crucial for development:

```
src/
├── core/                     # Core pipeline architecture
│   ├── base_pipeline.py      # Abstract base class
│   ├── mapreduce_pipeline.py # MapReduce implementation
│   ├── truncation_pipeline.py# Truncation implementation
│   └── factory.py            # Factory pattern implementation
├── loaders/                  # Dataset loaders
├── formatters/               # Output formatters
├── llm/                      # LLM integration
├── evaluation/               # Evaluation system
└── utils/                    # Utilities
```

## Adding New Datasets

### Step 1: Create Dataset Loader

Create a new loader class inheriting from `DatasetLoader`:

```python
# src/loaders/my_dataset_loader.py
from typing import List, Dict, Any, Optional, Tuple
from src.loaders.dataset_loader import DatasetLoader
from src.utils.document_processing import load_document_chunk, load_full_document

class MyDatasetLoader(DatasetLoader):
    """Loader for MyDataset with custom document processing."""

    def __init__(self, custom_param: str = "default"):
        """Initialize with dataset-specific parameters."""
        self.custom_param = custom_param

    def load_data(self, data_path: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load QA data from dataset file.

        Args:
            data_path: Path to dataset file
            num_samples: Optional limit on number of samples

        Returns:
            List of QA pairs with required fields:
            - question: str
            - answer: str (expected answer)
            - doc_name: str (document identifier)
            - evidence: str (optional evidence text)
        """
        qa_data = []

        # Example: JSON file loading
        import json
        with open(data_path, 'r') as f:
            raw_data = json.load(f)

        for i, item in enumerate(raw_data):
            if num_samples and i >= num_samples:
                break

            qa_pair = {
                'question': item['question'],
                'answer': item.get('answer', ''),
                'doc_name': item['document_id'],
                'evidence': item.get('evidence', ''),
                'doc_path': item.get('document_path', ''),
                # Add any dataset-specific fields
                'custom_field': item.get('custom_field', '')
            }
            qa_data.append(qa_pair)

        return qa_data

    def load_document_chunks(self, qa_pair: Dict[str, Any], chunk_size: int,
                           chunk_overlap: int, pdf_parser: str = "marker") -> Tuple[List[Any], int]:
        """Load and chunk document for MapReduce processing."""

        # Get document path from QA pair
        doc_path = self._get_document_path(qa_pair)

        # Use utility function for chunking
        chunks, token_count = load_document_chunk(
            doc_path=doc_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            method=pdf_parser
        )

        return chunks, token_count

    def load_full_document(self, qa_pair: Dict[str, Any],
                         pdf_parser: str = "marker") -> Tuple[str, int]:
        """Load complete document for Truncation processing."""

        doc_path = self._get_document_path(qa_pair)

        # Use utility function for full document loading
        document_text, token_count = load_full_document(
            doc_path=doc_path,
            method=pdf_parser
        )

        return document_text, token_count

    def get_document_identifier(self, qa_pair: Dict[str, Any]) -> str:
        """Generate unique identifier for document."""
        return qa_pair['doc_name']

    def get_results_directory(self) -> str:
        """Get directory for saving results."""
        return "results/mydataset_results"

    def _get_document_path(self, qa_pair: Dict[str, Any]) -> str:
        """Convert QA pair to document path."""
        if 'doc_path' in qa_pair and qa_pair['doc_path']:
            return qa_pair['doc_path']

        # Example: Construct path from doc_name
        base_dir = "/path/to/mydataset/documents"
        return f"{base_dir}/{qa_pair['doc_name']}.pdf"
```

### Step 2: Register Dataset with Factory

```python
# In src/core/factory.py or in your module
from src.loaders.my_dataset_loader import MyDatasetLoader

class PipelineFactory:
    # ... existing code ...

    @classmethod
    def create_pipeline(cls, dataset: str, format_type: str, **kwargs):
        """Create pipeline with dataset loader."""

        # Add your dataset to the loader mapping
        dataset_loaders = {
            'financebench': FinanceBenchLoader,
            'finqa': lambda doc_dir=None: FinQALoader(doc_dir),
            'webapp': WebappDatasetLoader,
            'mydataset': MyDatasetLoader,  # Add your dataset
        }

        # ... rest of factory logic ...
```

### Step 3: Test Your Dataset Loader

```python
# test_mydataset_loader.py
import pytest
from src.loaders.my_dataset_loader import MyDatasetLoader

def test_mydataset_loader():
    """Test MyDataset loader functionality."""

    loader = MyDatasetLoader()

    # Test data loading
    qa_data = loader.load_data("test_data/sample_mydataset.json", num_samples=2)
    assert len(qa_data) == 2
    assert 'question' in qa_data[0]
    assert 'answer' in qa_data[0]
    assert 'doc_name' in qa_data[0]

    # Test document identifier generation
    doc_id = loader.get_document_identifier(qa_data[0])
    assert isinstance(doc_id, str)
    assert len(doc_id) > 0

    # Test results directory
    results_dir = loader.get_results_directory()
    assert "mydataset" in results_dir.lower()
```

### Step 4: Update Command Line Interface

```python
# In main_async.py, add support for new dataset
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=[
        'financebench', 'finqa', 'webapp', 'mydataset'  # Add your dataset
    ], required=True)
    # ... other arguments ...

    return parser.parse_args()
```

## Adding New Output Formats

### Step 1: Create Output Formatter

```python
# src/formatters/my_formatter.py
from typing import Dict, Any, List, Tuple
from src.formatters.output_formatter import OutputFormatter

class MyFormatter(OutputFormatter):
    """Custom output formatter with specific processing logic."""

    def __init__(self, llm, prompts_dict: Dict[str, Any], custom_config: Dict = None):
        """Initialize formatter with custom configuration."""
        super().__init__(llm, prompts_dict)
        self.custom_config = custom_config or {}

    async def ainvoke_llm_map(self, chunk: Any, question: str, **kwargs) -> Dict[str, Any]:
        """Invoke LLM for map phase processing.

        Args:
            chunk: Document chunk to process
            question: User question
            **kwargs: Additional parameters

        Returns:
            Dict with map phase results including:
            - summary: str
            - relevance_score: int (0-10)
            - evidence: List[str]
            - custom_field: Any custom processing result
        """

        # Format prompt with custom template
        map_prompt = self.prompts_dict.get('map_prompt', '')
        formatted_prompt = map_prompt.format(
            chunk=chunk.page_content,
            question=question,
            custom_instruction=self.custom_config.get('map_instruction', '')
        )

        # Invoke LLM
        response = await self.llm.ainvoke(formatted_prompt, **kwargs)

        # Parse response according to your format
        map_result = self._parse_map_response(response)

        # Add custom processing
        map_result['custom_field'] = self._custom_map_processing(chunk, question)

        return map_result

    async def ainvoke_llm_reduce(self, formatted_results: Any, question: str, **kwargs) -> Any:
        """Invoke LLM for reduce phase processing."""

        # Format reduce prompt
        reduce_prompt = self.prompts_dict.get('reduce_prompt', '')
        formatted_prompt = reduce_prompt.format(
            question=question,
            map_results=formatted_results,
            custom_instruction=self.custom_config.get('reduce_instruction', '')
        )

        # Invoke LLM
        response = await self.llm.ainvoke(formatted_prompt, **kwargs)

        # Apply custom post-processing
        processed_response = self._custom_reduce_processing(response)

        return processed_response

    def preprocess_map_results(self, map_results: List[Dict[str, Any]], **kwargs) -> Any:
        """Preprocess map results for reduce phase.

        Args:
            map_results: List of map phase results
            **kwargs: Additional parameters (score_threshold, etc.)

        Returns:
            Formatted results for reduce phase
        """

        # Apply custom filtering
        filtered_results = self._custom_filter_results(map_results, **kwargs)

        # Format for reduce phase
        formatted_results = self._format_for_reduce(filtered_results)

        return formatted_results

    def parse_final_result(self, reduce_result: Any) -> Tuple[str, str, List[str]]:
        """Parse final result into answer, reasoning, evidence.

        Returns:
            Tuple of (answer, reasoning, evidence_list)
        """

        # Custom parsing logic
        answer = self._extract_answer(reduce_result)
        reasoning = self._extract_reasoning(reduce_result)
        evidence = self._extract_evidence(reduce_result)

        return answer, reasoning, evidence

    def _parse_map_response(self, response: str) -> Dict[str, Any]:
        """Parse map phase LLM response."""
        # Implement your parsing logic
        # This could be JSON parsing, regex extraction, etc.

        # Example: Pattern-based extraction
        import re

        summary_match = re.search(r'Summary: (.+)', response)
        score_match = re.search(r'Score: (\d+)', response)
        evidence_match = re.search(r'Evidence: (.+)', response, re.DOTALL)

        return {
            'summary': summary_match.group(1) if summary_match else '',
            'relevance_score': int(score_match.group(1)) if score_match else 0,
            'evidence': [evidence_match.group(1)] if evidence_match else [],
            'raw_response': response
        }

    def _custom_map_processing(self, chunk: Any, question: str) -> Any:
        """Custom processing for map phase."""
        # Add your custom logic here
        return {'processed': True, 'chunk_length': len(chunk.page_content)}

    def _custom_filter_results(self, results: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """Apply custom filtering logic."""
        threshold = kwargs.get('score_threshold', 5)

        # Custom filtering logic
        filtered = [
            result for result in results
            if result.get('relevance_score', 0) >= threshold
        ]

        # Sort by custom criteria
        filtered.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

        return filtered

    def _format_for_reduce(self, results: List[Dict[str, Any]]) -> str:
        """Format filtered results for reduce phase."""
        formatted_parts = []

        for i, result in enumerate(results, 1):
            part = f"Result {i}:\n"
            part += f"Summary: {result.get('summary', '')}\n"
            part += f"Evidence: {'; '.join(result.get('evidence', []))}\n"
            part += f"Score: {result.get('relevance_score', 0)}\n\n"
            formatted_parts.append(part)

        return "\n".join(formatted_parts)

    def _custom_reduce_processing(self, response: str) -> str:
        """Apply custom post-processing to reduce response."""
        # Add any custom logic here
        return response.strip()

    def _extract_answer(self, response: str) -> str:
        """Extract answer from reduce response."""
        # Implement extraction logic
        import re
        answer_match = re.search(r'Answer: (.+?)(?:\n|$)', response, re.DOTALL)
        return answer_match.group(1).strip() if answer_match else response[:200]

    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning from reduce response."""
        import re
        reasoning_match = re.search(r'Reasoning: (.+?)(?:\n(?:Answer|Evidence):|$)', response, re.DOTALL)
        return reasoning_match.group(1).strip() if reasoning_match else ''

    def _extract_evidence(self, response: str) -> List[str]:
        """Extract evidence from reduce response."""
        import re
        evidence_match = re.search(r'Evidence: (.+)', response, re.DOTALL)
        if evidence_match:
            evidence_text = evidence_match.group(1).strip()
            # Split by common delimiters
            evidence_list = [e.strip() for e in re.split(r'[;|\n]', evidence_text) if e.strip()]
            return evidence_list
        return []
```

### Step 2: Register Formatter with Factory

```python
# In src/core/factory.py
from src.formatters.my_formatter import MyFormatter

class PipelineFactory:
    @classmethod
    def get_available_formats(cls) -> List[str]:
        """Get list of available output formats."""
        return ['json', 'hybrid', 'plain_text', 'my_format']  # Add your format

    @classmethod
    def create_pipeline(cls, dataset: str, format_type: str, **kwargs):
        # Add formatter mapping
        if format_type == 'my_format':
            formatter = MyFormatter(
                llm=llm,
                prompts_dict=prompts_dict,
                custom_config=kwargs.get('custom_config', {})
            )
        # ... existing format handling ...
```

### Step 3: Create Custom Prompts

```yaml
# config/prompts/my_format_map_prompt.yml
map_prompt: |
  You are analyzing a document chunk to answer a financial question.

  Document chunk: {chunk}
  Question: {question}

  {custom_instruction}

  Please provide your analysis in the following format:
  Summary: [Brief summary of relevant information]
  Score: [Relevance score from 0-10]
  Evidence: [Specific evidence from the text]
  Custom Analysis: [Your custom analysis here]

# config/prompts/my_format_reduce_prompt.yml
reduce_prompt: |
  Based on the following analysis results, provide a comprehensive answer.

  Question: {question}
  Analysis Results:
  {map_results}

  {custom_instruction}

  Please provide:
  Answer: [Direct answer to the question]
  Reasoning: [Explanation of your reasoning]
  Evidence: [Supporting evidence from the document]
```

### Step 4: Test Your Formatter

```python
# test_my_formatter.py
import pytest
import asyncio
from src.formatters.my_formatter import MyFormatter
from src.llm.async_llm_client import create_async_rate_limited_llm

@pytest.mark.asyncio
async def test_my_formatter():
    """Test custom formatter functionality."""

    # Create mock objects
    llm = create_async_rate_limited_llm("gpt-4o-mini")
    prompts = {
        'map_prompt': 'Test map prompt: {chunk} {question}',
        'reduce_prompt': 'Test reduce prompt: {question} {map_results}'
    }

    formatter = MyFormatter(llm, prompts, {'map_instruction': 'Be concise'})

    # Test map phase (mock chunk object)
    class MockChunk:
        def __init__(self, content):
            self.page_content = content

    chunk = MockChunk("Test document content about revenue growth.")
    map_result = await formatter.ainvoke_llm_map(chunk, "What is revenue growth?")

    assert 'summary' in map_result
    assert 'relevance_score' in map_result
    assert 'evidence' in map_result
    assert 'custom_field' in map_result

    # Test preprocessing
    map_results = [
        {'relevance_score': 8, 'summary': 'High relevance'},
        {'relevance_score': 3, 'summary': 'Low relevance'},
        {'relevance_score': 7, 'summary': 'Medium relevance'}
    ]

    filtered = formatter.preprocess_map_results(map_results, score_threshold=5)
    assert isinstance(filtered, str)

    # Test reduce phase
    reduce_result = await formatter.ainvoke_llm_reduce(filtered, "Test question?")

    # Test final parsing
    answer, reasoning, evidence = formatter.parse_final_result(reduce_result)
    assert isinstance(answer, str)
    assert isinstance(reasoning, str)
    assert isinstance(evidence, list)
```

## Adding New Pipeline Types

### Step 1: Create Pipeline Class

```python
# src/core/my_pipeline.py
from typing import Dict, Any, List, Optional
from src.core.base_pipeline import BasePipeline

class MyPipeline(BasePipeline):
    """Custom pipeline with specialized processing logic."""

    def __init__(self, dataset_loader, custom_formatter, llm, prompts_dict: Dict[str, Any],
                 custom_param: int = 100, **kwargs):
        """Initialize custom pipeline."""
        super().__init__(dataset_loader, llm, prompts_dict, **kwargs)
        self.custom_formatter = custom_formatter
        self.custom_param = custom_param

    async def process_single_qa_async(self, qa_pair: Dict[str, Any],
                                    document_cache: Optional[Dict] = None) -> Dict[str, Any]:
        """Process single QA pair using custom approach.

        This is the core method that defines your pipeline's behavior.
        """

        # Record start time
        import time
        start_time = time.time()

        try:
            # Step 1: Load document
            doc_identifier = self.dataset_loader.get_document_identifier(qa_pair)

            if document_cache and doc_identifier in document_cache:
                # Use cached document
                document_data, original_tokens = document_cache[doc_identifier]
            else:
                # Load document using your custom logic
                document_data, original_tokens = await self._load_document_custom(qa_pair)

            # Step 2: Custom processing phases
            phase1_result = await self._custom_phase1(document_data, qa_pair['question'])
            phase2_result = await self._custom_phase2(phase1_result, qa_pair['question'])
            final_result = await self._custom_final_phase(phase2_result, qa_pair['question'])

            # Step 3: Parse results
            answer, reasoning, evidence = self.custom_formatter.parse_final_result(final_result)

            # Step 4: Compile results
            processing_time = time.time() - start_time

            result = {
                'question': qa_pair['question'],
                'answer': qa_pair.get('answer', ''),
                'llm_answer': answer,
                'llm_reasoning': reasoning,
                'llm_evidence': evidence,
                'doc_name': qa_pair.get('doc_name', ''),
                'processing_time': processing_time,
                'token_stats': self._get_token_stats(),
                'custom_metrics': self._get_custom_metrics(phase1_result, phase2_result)
            }

            return result

        except Exception as e:
            # Handle errors gracefully
            processing_time = time.time() - start_time
            return {
                'question': qa_pair['question'],
                'answer': qa_pair.get('answer', ''),
                'llm_answer': f'Error: {str(e)}',
                'llm_reasoning': 'Processing failed',
                'llm_evidence': [],
                'doc_name': qa_pair.get('doc_name', ''),
                'processing_time': processing_time,
                'error': str(e),
                'token_stats': {},
                'custom_metrics': {}
            }

    async def _load_document_custom(self, qa_pair: Dict[str, Any]) -> tuple:
        """Load document with custom logic."""

        # Example: Load document with custom chunking strategy
        chunks, token_count = self.dataset_loader.load_document_chunks(
            qa_pair,
            chunk_size=self.custom_param,  # Use custom parameter
            chunk_overlap=self.custom_param // 8
        )

        return chunks, token_count

    async def _custom_phase1(self, document_data, question: str) -> Dict[str, Any]:
        """First custom processing phase."""

        # Example: Parallel processing of chunks with custom logic
        results = []

        for chunk in document_data:
            # Apply custom processing to each chunk
            chunk_result = await self._process_chunk_custom(chunk, question)
            results.append(chunk_result)

        return {
            'phase': 'phase1',
            'results': results,
            'total_chunks': len(document_data)
        }

    async def _custom_phase2(self, phase1_result: Dict[str, Any], question: str) -> Dict[str, Any]:
        """Second custom processing phase."""

        # Example: Aggregate and filter results
        chunk_results = phase1_result['results']

        # Custom filtering logic
        high_quality_results = [
            result for result in chunk_results
            if result.get('quality_score', 0) > 0.7
        ]

        # Custom aggregation
        aggregated_content = self._aggregate_results(high_quality_results)

        return {
            'phase': 'phase2',
            'aggregated_content': aggregated_content,
            'filtered_count': len(high_quality_results),
            'original_count': len(chunk_results)
        }

    async def _custom_final_phase(self, phase2_result: Dict[str, Any], question: str) -> str:
        """Final processing phase."""

        # Use custom formatter for final synthesis
        final_prompt = self.prompts_dict.get('custom_final_prompt', '')
        formatted_prompt = final_prompt.format(
            question=question,
            content=phase2_result['aggregated_content'],
            metadata=phase2_result
        )

        final_response = await self.llm.ainvoke(formatted_prompt)

        return final_response

    async def _process_chunk_custom(self, chunk, question: str) -> Dict[str, Any]:
        """Custom chunk processing logic."""

        # Example: Custom analysis of chunk
        custom_prompt = self.prompts_dict.get('custom_chunk_prompt', '')
        formatted_prompt = custom_prompt.format(
            chunk=chunk.page_content,
            question=question
        )

        response = await self.llm.ainvoke(formatted_prompt)

        # Parse response and calculate custom metrics
        quality_score = self._calculate_quality_score(response)

        return {
            'chunk_response': response,
            'quality_score': quality_score,
            'chunk_length': len(chunk.page_content)
        }

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> str:
        """Aggregate chunk results with custom logic."""

        # Sort by quality score
        sorted_results = sorted(results, key=lambda x: x['quality_score'], reverse=True)

        # Take top results and combine
        top_results = sorted_results[:5]  # Top 5 results

        aggregated = "\n\n".join([
            result['chunk_response'] for result in top_results
        ])

        return aggregated

    def _calculate_quality_score(self, response: str) -> float:
        """Calculate custom quality score for response."""

        # Example: Simple heuristic based on response length and keywords
        score = 0.5  # Base score

        # Length bonus
        if len(response) > 100:
            score += 0.2

        # Keyword matching (example)
        financial_keywords = ['revenue', 'profit', 'growth', 'expenses', 'margin']
        keyword_count = sum(1 for keyword in financial_keywords if keyword in response.lower())
        score += keyword_count * 0.1

        return min(score, 1.0)  # Cap at 1.0

    def _get_token_stats(self) -> Dict[str, int]:
        """Get token usage statistics."""
        # Implement token tracking based on your LLM client
        if hasattr(self.llm, 'get_stats'):
            stats = self.llm.get_stats()
            return stats or {}
        return {}

    def _get_custom_metrics(self, phase1_result: Dict, phase2_result: Dict) -> Dict[str, Any]:
        """Get custom pipeline metrics."""
        return {
            'total_chunks_processed': phase1_result.get('total_chunks', 0),
            'high_quality_chunks': phase2_result.get('filtered_count', 0),
            'quality_retention_rate': (
                phase2_result.get('filtered_count', 0) /
                max(1, phase1_result.get('total_chunks', 1))
            ),
            'avg_quality_score': self._calculate_avg_quality(phase1_result.get('results', []))
        }

    def _calculate_avg_quality(self, results: List[Dict[str, Any]]) -> float:
        """Calculate average quality score."""
        if not results:
            return 0.0

        total_score = sum(result.get('quality_score', 0) for result in results)
        return total_score / len(results)

    def compile_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compile pipeline-specific statistics."""

        # Basic statistics
        total_samples = len(results)
        successful_samples = len([r for r in results if 'error' not in r])
        total_processing_time = sum(r.get('processing_time', 0) for r in results)

        # Custom statistics
        custom_metrics_list = [r.get('custom_metrics', {}) for r in results if 'custom_metrics' in r]

        avg_quality_retention = 0
        if custom_metrics_list:
            avg_quality_retention = sum(
                m.get('quality_retention_rate', 0) for m in custom_metrics_list
            ) / len(custom_metrics_list)

        return {
            'total_samples': total_samples,
            'successful_samples': successful_samples,
            'success_rate': successful_samples / max(1, total_samples),
            'total_processing_time': total_processing_time,
            'avg_processing_time': total_processing_time / max(1, total_samples),
            'avg_quality_retention_rate': avg_quality_retention,
            'custom_pipeline_type': 'MyPipeline'
        }
```

### Step 2: Register Pipeline with Factory

```python
# In src/core/factory.py
from src.core.my_pipeline import MyPipeline
from src.formatters.my_formatter import MyFormatter

class PipelineFactory:
    @classmethod
    def create_my_pipeline(cls, dataset: str, llm, prompts_dict: Dict[str, Any],
                          custom_param: int = 100, **kwargs):
        """Create custom pipeline type."""

        # Get dataset loader
        dataset_loader = cls._get_dataset_loader(dataset, **kwargs)

        # Create custom formatter
        custom_formatter = MyFormatter(llm, prompts_dict)

        # Create and return pipeline
        return MyPipeline(
            dataset_loader=dataset_loader,
            custom_formatter=custom_formatter,
            llm=llm,
            prompts_dict=prompts_dict,
            custom_param=custom_param,
            **kwargs
        )
```

### Step 3: Add Command Line Support

```python
# In main_async.py
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--approach', choices=[
        'mapreduce', 'truncation', 'my_pipeline'  # Add your pipeline
    ], required=True)

    # Add custom parameters
    parser.add_argument('--custom_param', type=int, default=100,
                       help='Custom parameter for my_pipeline')

    return parser.parse_args()

async def main():
    args = parse_arguments()

    # ... existing setup code ...

    if args.approach == 'my_pipeline':
        pipeline = PipelineFactory.create_my_pipeline(
            dataset=args.dataset,
            llm=llm,
            prompts_dict=prompts,
            custom_param=args.custom_param
        )
    # ... existing approach handling ...
```

## Testing Framework

### Unit Tests

```python
# tests/test_my_components.py
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from src.loaders.my_dataset_loader import MyDatasetLoader
from src.formatters.my_formatter import MyFormatter
from src.core.my_pipeline import MyPipeline

class TestMyDatasetLoader:
    def test_load_data(self):
        loader = MyDatasetLoader()
        # Test with mock data file
        # ... test implementation ...

    def test_document_identifier(self):
        loader = MyDatasetLoader()
        qa_pair = {'doc_name': 'test_doc', 'question': 'test?'}
        doc_id = loader.get_document_identifier(qa_pair)
        assert doc_id == 'test_doc'

class TestMyFormatter:
    @pytest.mark.asyncio
    async def test_map_phase(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = "Summary: Test summary\nScore: 8\nEvidence: Test evidence"

        formatter = MyFormatter(mock_llm, {'map_prompt': '{chunk} {question}'})

        chunk = Mock()
        chunk.page_content = "Test content"

        result = await formatter.ainvoke_llm_map(chunk, "Test question?")

        assert 'summary' in result
        assert 'relevance_score' in result
        assert result['relevance_score'] == 8

class TestMyPipeline:
    @pytest.mark.asyncio
    async def test_single_qa_processing(self):
        # Mock dependencies
        mock_loader = Mock(spec=MyDatasetLoader)
        mock_formatter = Mock(spec=MyFormatter)
        mock_llm = AsyncMock()

        # Setup mocks
        mock_loader.get_document_identifier.return_value = "test_doc"
        mock_loader.load_document_chunks.return_value = ([Mock()], 1000)
        mock_formatter.parse_final_result.return_value = ("answer", "reasoning", ["evidence"])

        pipeline = MyPipeline(
            dataset_loader=mock_loader,
            custom_formatter=mock_formatter,
            llm=mock_llm,
            prompts_dict={'custom_chunk_prompt': '{chunk} {question}'}
        )

        qa_pair = {
            'question': 'Test question?',
            'answer': 'Expected answer',
            'doc_name': 'test_doc'
        }

        result = await pipeline.process_single_qa_async(qa_pair)

        assert 'llm_answer' in result
        assert 'processing_time' in result
        assert 'custom_metrics' in result
```

### Integration Tests

```python
# tests/test_integration.py
import pytest
import asyncio
from src.core.factory import PipelineFactory
from src.llm.async_llm_client import create_async_rate_limited_llm

@pytest.mark.asyncio
@pytest.mark.integration
async def test_end_to_end_my_pipeline():
    """Test complete pipeline with real components."""

    # Create real LLM client (with test API key)
    llm = create_async_rate_limited_llm("gpt-4o-mini", temperature=0.0)

    # Load test prompts
    prompts = {
        'custom_chunk_prompt': 'Analyze: {chunk}\nQuestion: {question}',
        'custom_final_prompt': 'Final answer for: {question}\nContent: {content}'
    }

    # Create pipeline
    pipeline = PipelineFactory.create_my_pipeline(
        dataset='mydataset',
        llm=llm,
        prompts_dict=prompts,
        custom_param=50
    )

    # Process test data
    results = await pipeline.process_dataset_async(
        data_path='tests/data/sample_mydataset.json',
        model_name='gpt-4o-mini',
        num_samples=2
    )

    # Verify results
    assert results['num_samples'] == 2
    assert 'time_taken' in results
    assert len(results['qa_data']) == 2

    # Check custom metrics
    for qa_result in results['qa_data']:
        assert 'custom_metrics' in qa_result
        assert 'quality_retention_rate' in qa_result['custom_metrics']
```

## Contributing Guidelines

### Code Style

1. **Follow PEP 8**: Use consistent Python code style
2. **Type Hints**: Add type hints to all public methods
3. **Docstrings**: Document all classes and public methods
4. **Error Handling**: Implement proper error handling and logging

### Example Code Style:

```python
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ExampleClass:
    """Example class demonstrating code style."""

    def __init__(self, param: str, optional_param: Optional[int] = None) -> None:
        """Initialize example class.

        Args:
            param: Required string parameter
            optional_param: Optional integer parameter
        """
        self.param = param
        self.optional_param = optional_param or 42

    async def async_method(self, data: Dict[str, Any]) -> Tuple[str, int]:
        """Example async method.

        Args:
            data: Input data dictionary

        Returns:
            Tuple of processed string and count

        Raises:
            ValueError: If data is invalid
        """
        try:
            if not data:
                raise ValueError("Data cannot be empty")

            # Process data
            result = self._process_data(data)
            count = len(result)

            logger.info(f"Processed {count} items")
            return result, count

        except Exception as e:
            logger.error(f"Error in async_method: {e}")
            raise

    def _process_data(self, data: Dict[str, Any]) -> str:
        """Private method for data processing."""
        return str(data)
```

### Submission Process

1. **Fork Repository**: Create your own fork
2. **Feature Branch**: Create feature branch from `main`
3. **Implement Changes**: Add your dataset/formatter/pipeline
4. **Add Tests**: Include comprehensive tests
5. **Documentation**: Update relevant documentation
6. **Pull Request**: Submit PR with clear description

### PR Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] New dataset loader
- [ ] New output formatter
- [ ] New pipeline type
- [ ] Bug fix
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Documentation
- [ ] Code is documented
- [ ] README updated if needed
- [ ] Examples added if applicable

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] No hardcoded credentials or paths
- [ ] Error handling implemented
```

## Debugging and Development Tools

### Debug Configuration

```python
# debug_config.py
import logging
import sys

def setup_debug_logging():
    """Setup comprehensive debug logging."""

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # Create file handler
    file_handler = logging.FileHandler('debug.log')
    file_handler.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# Usage
if __name__ == "__main__":
    logger = setup_debug_logging()
    logger.debug("Debug logging enabled")
```

### Performance Profiling

```python
# profile_tools.py
import cProfile
import pstats
import io
from functools import wraps

def profile_function(func):
    """Decorator to profile function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()

        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()

            # Print stats
            s = io.StringIO()
            stats = pstats.Stats(profiler, stream=s)
            stats.sort_stats('tottime')
            stats.print_stats(10)  # Top 10 functions

            print(f"\nProfile for {func.__name__}:")
            print(s.getvalue())

        return result
    return wrapper

# Usage
@profile_function
def my_function():
    # Your function code here
    pass
```

This development guide provides comprehensive information for extending the FinMapReduce system. Follow these patterns and guidelines to maintain consistency and quality in your contributions.