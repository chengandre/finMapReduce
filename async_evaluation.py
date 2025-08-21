import asyncio
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Type
from tqdm.asyncio import tqdm as atqdm


class EvaluationFormatter(ABC):
    """
    Abstract base class for formatting evaluation items.

    Each formatter knows how to format QA data for a specific evaluation style.
    """

    @abstractmethod
    def format_item(self, sample: Dict[str, Any], item_number: int) -> str:
        """
        Format a single evaluation item.

        Args:
            sample: Dictionary containing question, answers, and other data
            item_number: Item number in the batch

        Returns:
            Formatted string for the evaluation item
        """
        pass

    @abstractmethod
    def extract_sample_data(self, qa_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant data from a QA item for evaluation.

        Args:
            qa_item: Original QA data item

        Returns:
            Dictionary with data needed for evaluation
        """
        pass


class StandardEvaluationFormatter(EvaluationFormatter):
    """
    Formatter for standard evaluation with evidence and reasoning.

    Used by FinanceBench and FinQA pipelines.
    """

    def extract_sample_data(self, qa_item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data for standard evaluation format."""
        return {
            "llm_evidence": qa_item.get("llm_evidence", []),
            "llm_reasoning": qa_item.get("llm_reasoning", "No reasoning provided"),
            "llm_answer": qa_item.get("llm_answer", "No answer provided"),
            "golden_answer": qa_item["answer"],
            "question": qa_item["question"]
        }

    def format_item(self, sample: Dict[str, Any], item_number: int) -> str:
        """Format item with evidence and reasoning."""
        return (
            f"  <item>\n"
            f"    <item_number>{item_number}</item_number>\n"
            f"    <query>\n"
            f"      {sample['question']}\n"
            f"    </query>\n"
            f"    <llm_evidence>\n"
            f"      {sample['llm_evidence']}\n"
            f"    </llm_evidence>\n"
            f"    <llm_reasoning>\n"
            f"      {sample['llm_reasoning']}\n"
            f"    </llm_reasoning>\n"
            f"    <answers_to_compare>\n"
            f"      <llm_answer>\n"
            f"        {sample['llm_answer']}\n"
            f"      </llm_answer>\n"
            f"      <golden_answer>\n"
            f"        {sample['golden_answer']}\n"
            f"      </golden_answer>\n"
            f"    </answers_to_compare>\n"
            f"  </item>"
        )


class LastYearEvaluationFormatter(EvaluationFormatter):
    """
    Formatter for last year's evaluation format.

    Simpler format with just llm_answer and golden_answer.
    """

    def extract_sample_data(self, qa_item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data for last year evaluation format."""
        return {
            "llm_answer": qa_item["llm_answer"],
            "golden_answer": qa_item["answer"],
            "question": qa_item["question"]
        }

    def format_item(self, sample: Dict[str, Any], item_number: int) -> str:
        """Format item with just answers."""
        return (
            f"  <item>\n"
            f"    <item_number>{item_number}</item_number>\n"
            f"    <query>\n"
            f"      {sample['question']}\n"
            f"    </query>\n"
            f"    <answers_to_compare>\n"
            f"      <llm_answer>\n"
            f"        {sample['llm_answer']}\n"
            f"      </llm_answer>\n"
            f"      <golden_answer>\n"
            f"        {sample['golden_answer']}\n"
            f"      </golden_answer>\n"
            f"    </answers_to_compare>\n"
            f"  </item>"
        )


class SimpleAnswerEvaluationFormatter(EvaluationFormatter):
    """
    Example of a simple formatter for just comparing answers.

    This demonstrates how easy it is to add new evaluation formats.
    """

    def extract_sample_data(self, qa_item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract just the answers and question."""
        return {
            "model_answer": qa_item.get("llm_answer", qa_item.get("evaluation_answer", "No answer")),
            "golden_answer": qa_item["answer"],
            "question": qa_item["question"]
        }

    def format_item(self, sample: Dict[str, Any], item_number: int) -> str:
        """Simple format with just the answers."""
        return (
            f"  <item>\n"
            f"    <item_number>{item_number}</item_number>\n"
            f"    <question>{sample['question']}</question>\n"
            f"    <model_answer>{sample['model_answer']}</model_answer>\n"
            f"    <golden_answer>{sample['golden_answer']}</golden_answer>\n"
            f"  </item>"
        )


class DetailedEvaluationFormatter(EvaluationFormatter):
    """
    Example of a detailed formatter that includes metadata.

    Could be used for more comprehensive evaluation including sources, confidence, etc.
    """

    def extract_sample_data(self, qa_item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive data including metadata."""
        return {
            "question": qa_item["question"],
            "llm_answer": qa_item.get("llm_answer", "No answer provided"),
            "golden_answer": qa_item["answer"],
            "llm_evidence": qa_item.get("llm_evidence", []),
            "llm_reasoning": qa_item.get("llm_reasoning", ""),
            "question_type": qa_item.get("question_type", "unknown"),
            "doc_name": qa_item.get("doc_name", "unknown"),
            "confidence": qa_item.get("confidence", "N/A")
        }

    def format_item(self, sample: Dict[str, Any], item_number: int) -> str:
        """Detailed format with metadata."""
        return (
            f"  <item>\n"
            f"    <item_number>{item_number}</item_number>\n"
            f"    <metadata>\n"
            f"      <question_type>{sample['question_type']}</question_type>\n"
            f"      <document>{sample['doc_name']}</document>\n"
            f"      <confidence>{sample['confidence']}</confidence>\n"
            f"    </metadata>\n"
            f"    <query>{sample['question']}</query>\n"
            f"    <llm_response>\n"
            f"      <answer>{sample['llm_answer']}</answer>\n"
            f"      <reasoning>{sample['llm_reasoning']}</reasoning>\n"
            f"      <evidence>{sample['llm_evidence']}</evidence>\n"
            f"    </llm_response>\n"
            f"    <golden_answer>{sample['golden_answer']}</golden_answer>\n"
            f"  </item>"
        )


class EvaluationFormatterFactory:
    """
    Factory for creating evaluation formatters.

    Automatically detects the appropriate formatter based on data structure.
    """

    # Registry of formatters
    _formatters: Dict[str, Type[EvaluationFormatter]] = {
        'standard': StandardEvaluationFormatter,
        'last_year': LastYearEvaluationFormatter,
        'simple': SimpleAnswerEvaluationFormatter,
        'detailed': DetailedEvaluationFormatter,
    }

    @classmethod
    def create_formatter(cls, qa_data: List[Dict[str, Any]],
                        formatter_type: Optional[str] = None) -> EvaluationFormatter:
        """
        Create appropriate formatter for the data.

        Args:
            qa_data: List of QA items to evaluate
            formatter_type: Explicit formatter type (optional)

        Returns:
            Appropriate EvaluationFormatter instance
        """
        if formatter_type:
            if formatter_type not in cls._formatters:
                raise ValueError(f"Unknown formatter type: {formatter_type}")
            return cls._formatters[formatter_type]()

        # Auto-detect based on data structure
        if not qa_data:
            return StandardEvaluationFormatter()  # Default

        sample = qa_data[0]

        # Check for standard format
        if 'llm_evidence' in sample or 'llm_reasoning' in sample:
            return StandardEvaluationFormatter()

        # Check for last year format
        if 'llm_answer' in sample:
            return LastYearEvaluationFormatter()


        # Default to simple format
        return SimpleAnswerEvaluationFormatter()

    @classmethod
    def register_formatter(cls, name: str, formatter_class: Type[EvaluationFormatter]):
        """Register a new formatter type."""
        if not issubclass(formatter_class, EvaluationFormatter):
            raise TypeError(f"{formatter_class} must inherit from EvaluationFormatter")
        cls._formatters[name] = formatter_class

    @classmethod
    def get_available_formatters(cls) -> List[str]:
        """Get list of available formatter types."""
        return list(cls._formatters.keys())


class AsyncLLMJudgeEvaluator:
    """Async version of LLM judge evaluator"""

    def __init__(self,
                 llm: Any,
                 judge_prompt: Any,
                 batch_size: int = 5,
                 max_workers: int = 20):
        self.llm = llm
        self.judge_prompt = judge_prompt
        self.batch_size = batch_size
        self.max_workers = max_workers

    async def evaluate_async(self,
                            qa_data: List[Dict[str, Any]],
                            formatter: Optional[Any] = None) -> Dict[str, Any]:
        """Async evaluation maintaining same output structure"""

        # Get appropriate formatter (reuse existing logic)
        if formatter is None:
            formatter = EvaluationFormatterFactory.create_formatter(qa_data)

        # Prepare batches (reuse existing method)
        batches = self._prepare_batches(qa_data, formatter)

        # Process batches asynchronously
        batch_results = await self._process_batches_async(batches, formatter)

        # Apply results back to qa_data (reuse existing method)
        self._apply_results_to_qa_data(qa_data, batch_results)

        # Calculate and return statistics (reuse existing method)
        return self._calculate_statistics(qa_data, batch_results)

    async def _process_single_batch_async(self,
                                         batch_data: tuple,
                                         formatter: Any) -> Dict[str, Any]:
        """Process a single batch asynchronously"""
        batch_idx, batch = batch_data

        # Format items for evaluation (reuse existing logic)
        context_parts = []
        for i, sample in enumerate(batch, 1):
            item_block = formatter.format_item(sample, i)
            context_parts.append(item_block)

        context = "<evaluation_items>\n" + "\n".join(context_parts) + "\n</evaluation_items>"

        try:
            # Use async invoke method
            judge_response = await self.llm.invoke(self.judge_prompt, context=context)

            # Extract token usage (reuse existing method)
            tokens = self._extract_token_usage_from_response(judge_response)

            # Parse response (reuse existing method)
            evaluation_data = self._parse_judge_response(judge_response)

            return {
                "batch_idx": batch_idx,
                "success": True,
                "evaluation_data": evaluation_data,
                "batch": batch,
                "tokens": tokens
            }
        except Exception as e:
            print(f"Error processing batch {batch_idx + 1}: {str(e)}")

            # Create fallback results (same as original)
            fallback = {
                "evaluation_results": [
                    {
                        "evaluation_number": i + 1,
                        "reasoning": f"Processing error: {str(e)}",
                        "judgement": "Error"
                    }
                    for i in range(len(batch))
                ]
            }

            return {
                "batch_idx": batch_idx,
                "success": False,
                "evaluation_data": fallback,
                "batch": batch,
                "error": str(e),
                "tokens": {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0}
            }

    async def _process_batches_async(self,
                                    batches: List[tuple],
                                    formatter: Any) -> List[Dict[str, Any]]:
        """Process all batches asynchronously with concurrency control"""

        # Process batches without semaphore (rate limiting handled in async_llm_client.py)
        # Create tasks
        tasks = [
            asyncio.create_task(self._process_single_batch_async(batch_data, formatter))
            for batch_data in batches
        ]

        # Process with progress bar
        batch_results = []
        try:
            for task in atqdm.as_completed(tasks, total=len(tasks), desc="Evaluating batches"):
                result = await task
                batch_results.append(result)

                if not result['success']:
                    print(f"Error in batch {result['batch_idx'] + 1}: {result.get('error', 'Unknown error')}")
        except ImportError:
            # Fallback if tqdm.asyncio is not available
            from tqdm import tqdm
            pbar = tqdm(total=len(tasks), desc="Evaluating batches")
            for task in asyncio.as_completed(tasks):
                result = await task
                batch_results.append(result)
                pbar.update(1)
                if not result['success']:
                    print(f"Error in batch {result['batch_idx'] + 1}: {result.get('error', 'Unknown error')}")
            pbar.close()

        # Sort by batch index to maintain order
        batch_results.sort(key=lambda x: x['batch_idx'])
        return batch_results

    def _prepare_batches(self, qa_data: List[Dict[str, Any]], formatter: EvaluationFormatter) -> List[tuple]:
        """Prepare batches for evaluation."""
        batches = []
        total_samples = len(qa_data)

        for i in range(0, total_samples, self.batch_size):
            batch = []
            end_idx = min(i + self.batch_size, total_samples)

            for j in range(i, end_idx):
                sample = formatter.extract_sample_data(qa_data[j])
                batch.append(sample)

            batches.append((i // self.batch_size, batch))

        return batches

    def _apply_results_to_qa_data(self, qa_data: List[Dict[str, Any]], batch_results: List[Dict[str, Any]]):
        """Apply evaluation results back to the original QA data."""
        for result in batch_results:
            batch_idx = result['batch_idx']
            evaluation_data = result['evaluation_data']

            # Add judgment results back to qa_data
            eval_results = evaluation_data.get("evaluation_results", [])
            for i, eval_result in enumerate(eval_results):
                qa_idx = batch_idx * self.batch_size + i
                if qa_idx < len(qa_data):
                    qa_data[qa_idx]["judgment"] = eval_result.get("judgement", "Error")
                    qa_data[qa_idx]["reasoning"] = eval_result.get("reasoning", "No reasoning provided")

    def _calculate_statistics(self, qa_data: List[Dict[str, Any]], batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate evaluation statistics."""
        # Count judgments
        judgment_counts = {
            "correct": 0,
            "coherent": 0,
            "deviated": 0,
            "incorrect": 0,
            "no_answer": 0,
            "error": 0
        }

        total_samples = 0

        for qa_item in qa_data:
            judgment = qa_item.get("judgment", "Error").lower()

            # Normalize judgment
            if judgment == "correct":
                judgment_counts["correct"] += 1
            elif judgment == "coherent":
                judgment_counts["coherent"] += 1
            elif judgment == "deviated":
                judgment_counts["deviated"] += 1
            elif judgment == "incorrect":
                judgment_counts["incorrect"] += 1
            elif judgment == "no answer" or judgment == "no_answer":
                judgment_counts["no_answer"] += 1
            else:
                judgment_counts["error"] += 1

            total_samples += 1

        # Calculate evaluation token totals
        evaluation_tokens = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0}
        for result in batch_results:
            tokens = result.get("tokens", {})
            evaluation_tokens["input_tokens"] += tokens.get("input_tokens", 0)
            evaluation_tokens["output_tokens"] += tokens.get("output_tokens", 0)
            evaluation_tokens["cache_read_tokens"] += tokens.get("cache_read_tokens", 0)

        # Collect all detailed judgments
        all_judgments = []
        for result in batch_results:
            all_judgments.append(result['evaluation_data'])

        # Prepare results
        results = {
            "correct": judgment_counts["correct"],
            "coherent": judgment_counts["coherent"],
            "deviated": judgment_counts["deviated"],
            "incorrect": judgment_counts["incorrect"],
            "no_answer": judgment_counts["no_answer"],
            "total": total_samples,
            "accuracy": judgment_counts["correct"] / total_samples if total_samples > 0 else 0,
            "evaluation_tokens": evaluation_tokens,
            "detailed_judgments": all_judgments
        }

        return results

    def _parse_judge_response(self, judge_response: Any) -> Dict[str, Any]:
        """Parse the judge's response to extract evaluation data."""
        if isinstance(judge_response, dict):
            json_result = judge_response.get('json', {})
            if isinstance(json_result, dict):
                return json_result
            else:
                return {}

        # Try to parse JSON from string response
        try:
            import json5
            result = json5.loads(str(judge_response))
            if isinstance(result, dict):
                return result
            else:
                return {}
        except:
            return {}

    def _extract_token_usage_from_response(self, response: Any) -> Dict[str, int]:
        """Extract token usage from LLM response."""
        tokens = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0}

        if isinstance(response, dict):
            raw_response = response.get('raw_response')
            if raw_response and hasattr(raw_response, 'usage_metadata') and raw_response.usage_metadata:
                tokens["input_tokens"] = raw_response.usage_metadata.get("input_tokens", 0)
                tokens["output_tokens"] = raw_response.usage_metadata.get("output_tokens", 0)
                input_token_details = raw_response.usage_metadata.get("input_token_details", {})
                tokens["cache_read_tokens"] = input_token_details.get("cache_read", 0)
        elif hasattr(response, 'usage_metadata') and response.usage_metadata:
            tokens["input_tokens"] = response.usage_metadata.get("input_tokens", 0)
            tokens["output_tokens"] = response.usage_metadata.get("output_tokens", 0)
            input_token_details = response.usage_metadata.get("input_token_details", {})
            tokens["cache_read_tokens"] = input_token_details.get("cache_read", 0)

        return tokens


# Convenience function to evaluate QA data using LLM judge (async version)
async def evaluate_with_llm_judge_async(llm: Any,
                                       qa_data: List[Dict[str, Any]],
                                       prompts_dict: Dict[str, Any],
                                       batch_size: int = 5,
                                       judge_prompt_key: str = 'judge_prompt',
                                       formatter_type: Optional[str] = None,
                                       max_workers: int = 20) -> Dict[str, Any]:
    """
    Convenience function to evaluate QA data using LLM judge (async version).

    This function maintains compatibility with existing code while using
    the new extensible evaluation system.

    Args:
        llm: Language model instance
        qa_data: List of QA items to evaluate
        prompts_dict: Dictionary containing prompts
        batch_size: Number of items per batch
        judge_prompt_key: Key for judge prompt in prompts_dict
        formatter_type: Specific formatter to use (auto-detected if None)
        max_workers: Maximum concurrent workers

    Returns:
        Evaluation results dictionary
    """
    # Get judge prompt
    judge_prompt = prompts_dict[judge_prompt_key]

    # Create evaluator
    evaluator = AsyncLLMJudgeEvaluator(
        llm=llm,
        judge_prompt=judge_prompt,
        batch_size=batch_size,
        max_workers=max_workers
    )

    # Get formatter
    if formatter_type:
        formatter = EvaluationFormatterFactory._formatters[formatter_type]()
    else:
        formatter = None  # Auto-detect

    # Evaluate
    return await evaluator.evaluate_async(qa_data, formatter)