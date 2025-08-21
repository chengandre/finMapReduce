import asyncio
from functools import partial
from typing import Any, Dict, List, Optional
from tqdm.asyncio import tqdm as atqdm

from evaluation import EvaluationFormatterFactory


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

    def _prepare_batches(self, qa_data, formatter):
        """Prepare batches for evaluation - reuse from sync version"""
        from evaluation import LLMJudgeEvaluator

        # Create a temporary evaluator instance to access the method
        temp_evaluator = LLMJudgeEvaluator(self.llm, self.judge_prompt, self.batch_size)
        return temp_evaluator._prepare_batches(qa_data, formatter)

    def _apply_results_to_qa_data(self, qa_data, batch_results):
        """Apply results back to qa_data - reuse from sync version"""
        from evaluation import LLMJudgeEvaluator

        # Create a temporary evaluator instance to access the method
        temp_evaluator = LLMJudgeEvaluator(self.llm, self.judge_prompt, self.batch_size)
        return temp_evaluator._apply_results_to_qa_data(qa_data, batch_results)

    def _calculate_statistics(self, qa_data, batch_results):
        """Calculate statistics - reuse from sync version"""
        from evaluation import LLMJudgeEvaluator

        # Create a temporary evaluator instance to access the method
        temp_evaluator = LLMJudgeEvaluator(self.llm, self.judge_prompt, self.batch_size)
        return temp_evaluator._calculate_statistics(qa_data, batch_results)

    def _parse_judge_response(self, judge_response):
        """Parse judge response - reuse from sync version"""
        from evaluation import LLMJudgeEvaluator

        # Create a temporary evaluator instance to access the method
        temp_evaluator = LLMJudgeEvaluator(self.llm, self.judge_prompt, self.batch_size)
        return temp_evaluator._parse_judge_response(judge_response)

    def _extract_token_usage_from_response(self, response):
        """Extract token usage - reuse from sync version"""
        from evaluation import LLMJudgeEvaluator

        # Create a temporary evaluator instance to access the method
        temp_evaluator = LLMJudgeEvaluator(self.llm, self.judge_prompt, self.batch_size)
        return temp_evaluator._extract_token_usage_from_response(response)


# Helper function for async evaluation
async def evaluate_with_llm_judge_async(llm: Any,
                                       qa_data: List[Dict[str, Any]],
                                       prompts_dict: Dict[str, Any],
                                       batch_size: int = 5,
                                       judge_prompt_key: str = 'judge_prompt',
                                       formatter_type: Optional[str] = None,
                                       max_workers: int = 20) -> Dict[str, Any]:
    """Async version of evaluation helper"""

    judge_prompt = prompts_dict[judge_prompt_key]

    evaluator = AsyncLLMJudgeEvaluator(
        llm=llm,
        judge_prompt=judge_prompt,
        batch_size=batch_size,
        max_workers=max_workers
    )

    if formatter_type:
        formatter = EvaluationFormatterFactory._formatters[formatter_type]()
    else:
        formatter = None

    return await evaluator.evaluate_async(qa_data, formatter)