import concurrent.futures
import datetime
import glob
import json
import os
import random
import statistics
import time

from langchain.prompts import load_prompt
from tqdm import tqdm

from utils import create_rate_limited_llm, RateLimitConfig
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

def load_all_samples():
    """Load all samples from all JSONL files in the results directory"""
    results_dir = '../financebench/results'
    jsonl_files = glob.glob(os.path.join(results_dir, '*.jsonl'))

    all_samples = []

    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r') as file:
            for line in file:
                try:
                    sample = json.loads(line)
                    # Check if the sample has all required fields
                    if all(key in sample for key in ['model_answer', 'gold_answer', 'question', 'label']):
                        all_samples.append(sample)
                except json.JSONDecodeError:
                    continue

    print(f"Loaded {len(all_samples)} samples from {len(jsonl_files)} files")
    return all_samples

def format_context(samples):
    """Format 5 samples into the required format for the judge prompt"""
    context_parts = []

    for i, sample in enumerate(samples, 1):
        test_block = (
            f"<test_{i}>\n"
            f"  <llm_answer>\n"
            f"  {sample['model_answer']}\n"
            f"  </llm_answer>\n"
            f"  <golden_answer>\n"
            f"  {sample['gold_answer']}\n"
            f"  </golden_answer>\n"
            f"  <query>\n"
            f"  {sample['question']}\n"
            f"  </query>\n"
            f"</test_{i}>"
        )
        context_parts.append(test_block)

    return "\n".join(context_parts)


def evaluate_judge_performance(judge_response_json, samples, raw_response=None):
    """Evaluate how well the LLM judge performed compared to the true labels"""
    # Validate the JSON structure
    if not isinstance(judge_response_json, dict) or "evaluation_results" not in judge_response_json:
        return {
            "success": False,
            "error": "Missing evaluation_results key in response",
            "raw_response": raw_response
        }

    # Extract judgments from the JSON response
    judgments = []
    evaluation_results = judge_response_json["evaluation_results"]

    if not isinstance(evaluation_results, list):
        return {
            "success": False,
            "error": "evaluation_results is not a list",
            "raw_response": raw_response
        }

    # Check if we have exactly 5 judgments
    if len(evaluation_results) != 5:
        return {
            "success": False,
            "error": f"Expected 5 judgments, got {len(evaluation_results)}",
            "raw_response": raw_response
        }

    # Extract and validate judgments
    valid_judgments = ["Correct", "Incorrect", "No answer"]
    for i, eval_result in enumerate(evaluation_results):
        if not isinstance(eval_result, dict) or "judgement" not in eval_result:
            return {
                "success": False,
                "error": f"Missing judgement key in evaluation result {i+1}",
                "raw_response": raw_response
            }

        judgment = eval_result.get("judgement")
        if judgment not in valid_judgments:
            return {
                "success": False,
                "error": f"Invalid judgment '{judgment}' in result {i+1}. Expected one of: {valid_judgments}",
                "raw_response": raw_response
            }

        judgments.append(judgment)

    # Map the true labels to the expected format
    true_labels = []
    for sample in samples:
        if sample['label'] == "Correct Answer":
            true_labels.append("Correct")
        elif sample['label'] == "Incorrect Answer":
            true_labels.append("Incorrect")
        elif sample['label'] == "Refusal":
            true_labels.append("No answer")

    # Compare and calculate accuracy
    correct = 0
    incorrect_judgments = []

    for i, (judgment, true_label) in enumerate(zip(judgments, true_labels)):
        is_correct = judgment == true_label
        correct += int(is_correct)
        if not is_correct:
            # Store sample index (1-based), judge's judgment, true label, the full sample, and the judge's full evaluation result
            judge_eval_result = evaluation_results[i] if i < len(evaluation_results) else {}
            incorrect_judgments.append((i+1, judgment, true_label, samples[i], judge_eval_result))

    # Calculate precision, recall, F1, and confusion matrix
    precision_recall_f1 = calculate_precision_recall_f1(true_labels, judgments)
    confusion_matrix_data = generate_confusion_matrix(true_labels, judgments)

    # Return the results as a dictionary
    accuracy = correct / len(samples) if samples else 0
    return {
        "success": True,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(samples),
        "incorrect_judgments": incorrect_judgments,
        "precision_recall_f1": precision_recall_f1,
        "confusion_matrix": confusion_matrix_data
    }

def analyze_results_by_label(judge_response_json, samples):
    """Analyze results by label type"""
    # Extract judgments from the JSON response
    judgments = []
    if "evaluation_results" in judge_response_json:
        for eval_result in judge_response_json["evaluation_results"]:
            judgment = eval_result.get("judgement")
            judgments.append(judgment)

    # Group by true label
    results_by_label = {
        "Correct Answer": {"total": 0, "correct": 0},
        "Incorrect Answer": {"total": 0, "correct": 0},
        "Refusal": {"total": 0, "correct": 0}
    }

    for i, sample in enumerate(samples):
        label = sample['label']
        results_by_label[label]["total"] += 1

        expected = "Correct" if label == "Correct Answer" else "Incorrect" if label == "Incorrect Answer" else "No answer"
        if i < len(judgments) and judgments[i] == expected:
            results_by_label[label]["correct"] += 1

    return results_by_label

def calculate_precision_recall_f1(y_true, y_pred, labels=None):
    """Calculate precision, recall, and F1 score for multi-class classification"""
    if labels is None:
        labels = ["Correct", "Incorrect", "No answer"]

    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    # Calculate macro averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0
    )

    # Calculate micro averages
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='micro', zero_division=0
    )

    # Create per-class results
    per_class_metrics = {}
    for i, label in enumerate(labels):
        # Handle both array and scalar cases safely
        try:
            if isinstance(precision, (list, tuple, np.ndarray)) and i < len(precision):
                prec_val = precision[i]
            else:
                prec_val = float(precision) if precision is not None else 0.0
        except:
            prec_val = 0.0

        try:
            if isinstance(recall, (list, tuple, np.ndarray)) and i < len(recall):
                rec_val = recall[i]
            else:
                rec_val = float(recall) if recall is not None else 0.0
        except:
            rec_val = 0.0

        try:
            if isinstance(f1, (list, tuple, np.ndarray)) and i < len(f1):
                f1_val = f1[i]
            else:
                f1_val = float(f1) if f1 is not None else 0.0
        except:
            f1_val = 0.0

        try:
            if isinstance(support, (list, tuple, np.ndarray)) and i < len(support):
                supp_val = support[i]
            else:
                supp_val = int(support) if support is not None else 0
        except:
            supp_val = 0

        per_class_metrics[label] = {
            "precision": float(prec_val),
            "recall": float(rec_val),
            "f1_score": float(f1_val),
            "support": int(supp_val)
        }

    return {
        "per_class": per_class_metrics,
        "macro_avg": {
            "precision": float(macro_precision),
            "recall": float(macro_recall),
            "f1_score": float(macro_f1)
        },
        "micro_avg": {
            "precision": float(micro_precision),
            "recall": float(micro_recall),
            "f1_score": float(micro_f1)
        }
    }

def generate_confusion_matrix(y_true, y_pred, labels=None):
    """Generate confusion matrix for the predictions"""
    if labels is None:
        labels = ["Correct", "Incorrect", "No answer"]

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Convert to list for JSON serialization
    cm_list = cm.tolist()

    # Create detailed confusion matrix with labels
    detailed_cm = {}
    for i, true_label in enumerate(labels):
        detailed_cm[f"True_{true_label}"] = {}
        for j, pred_label in enumerate(labels):
            detailed_cm[f"True_{true_label}"][f"Pred_{pred_label}"] = int(cm_list[i][j])

    return {
        "matrix": cm_list,
        "labels": labels,
        "detailed": detailed_cm
    }

def process_batch(batch_id, samples_batch, prompt_template, llm):
    """Process a batch of samples using the LLMClient wrapper"""
    batch_start_time = time.time()

    # Format samples into context
    context = format_context(samples_batch)

    try:
        prompt = prompt_template.format(context=context)
        response = llm.invoke(prompt)

        # Get both the parsed JSON and raw response from the wrapper
        judge_response_json = response.get('json', {})
        raw_response = response.get('raw_response')
        raw_content = raw_response.content if raw_response and hasattr(raw_response, 'content') else str(response)

        # Extract token usage information from usage_metadata
        token_usage = {}
        if raw_response and hasattr(raw_response, 'usage_metadata'):
            usage_metadata = raw_response.usage_metadata
            token_usage = {
                'input_tokens': usage_metadata.get('input_tokens', 0),
                'output_tokens': usage_metadata.get('output_tokens', 0),
                'total_tokens': usage_metadata.get('input_tokens', 0) + usage_metadata.get('output_tokens', 0)
            }

        # Evaluate the judge's performance with validation
        eval_results = evaluate_judge_performance(judge_response_json, samples_batch, raw_content)

        if eval_results["success"]:
            label_results = analyze_results_by_label(judge_response_json, samples_batch)
            eval_results["label_results"] = label_results

        # Add token usage and timing to results
        batch_end_time = time.time()
        batch_processing_time = batch_end_time - batch_start_time

        eval_results["token_usage"] = token_usage
        eval_results["batch_processing_time"] = batch_processing_time

        return eval_results

    except Exception as e:
        batch_end_time = time.time()
        batch_processing_time = batch_end_time - batch_start_time

        return {
            "success": False,
            "error": f"Error processing batch {batch_id}: {str(e)}",
            "raw_response": str(e),
            "batch_processing_time": batch_processing_time
        }

def main():
    # Load all samples
    all_samples = load_all_samples()

    if not all_samples:
        print("No samples found. Please check the data files.")
        return

    # Test mode - limit to 20 samples
    test_mode = False  # Set to False to process all samples
    max_samples = 5  # Number of samples to process in test mode

    if test_mode:
        # Select a balanced subset of samples
        random.seed(42)  # For reproducibility
        samples_by_label = {
            "Correct Answer": [],
            "Incorrect Answer": [],
            "Refusal": []
        }

        # Group samples by label
        for sample in all_samples:
            label = sample.get('label')
            if label in samples_by_label:
                samples_by_label[label].append(sample)

        # Select samples from each category
        selected_samples = []
        samples_per_label = max_samples // 3

        for label, samples in samples_by_label.items():
            if samples:
                selected = random.sample(samples, min(samples_per_label, len(samples)))
                selected_samples.extend(selected)

        # If we need more samples to reach max_samples
        if len(selected_samples) < max_samples:
            remaining_needed = max_samples - len(selected_samples)
            # Create a pool of remaining samples
            remaining_pool = []
            for label, samples in samples_by_label.items():
                for sample in samples:
                    if sample not in selected_samples:
                        remaining_pool.append(sample)

            if remaining_pool:
                additional = random.sample(remaining_pool, min(remaining_needed, len(remaining_pool)))
                selected_samples.extend(additional)

        # Shuffle the selected samples
        random.shuffle(selected_samples)
        all_samples = selected_samples[:max_samples]
        print(f"Test mode: Selected {len(all_samples)} samples for testing")

    # Load prompt template once
    prompt_template = load_prompt('prompts/judge_evaluation.yml')

    # Configure rate-limited LLM
    # rate_limiter_config = {
    #     "requests_per_minute": 20,
    #     "tokens_per_minute": 4000000,
    #     "request_burst_size": 5,
    # }

    rate_limiter_config = {
        "requests_per_minute": 30000,
        "tokens_per_minute": 150000000,
        "request_burst_size": 3000,
    }

    # model_name = "deepseek/deepseek-r1-0528:free"
    model_name = "gpt-5-nano"

    rate_config = RateLimitConfig(**rate_limiter_config)
    llm = create_rate_limited_llm(
        model_name=model_name,
        temperature=1.00,
        max_tokens=8192,
        provider="openai",
        api_key_env="elm",
        rate_limit_config=rate_config,
        parse_json=True
    )


    # Batch size for processing
    batch_size = 5  # Each batch contains 5 samples as per the prompt format

    # Split samples into batches
    batches = [all_samples[i:i+batch_size] for i in range(0, len(all_samples), batch_size)]

    # If the last batch has fewer than 5 samples, remove it or pad it
    if len(batches[-1]) < 5:
        batches.pop()

    print(f"Processing {len(batches)} batches with {len(batches) * batch_size} samples total")

    # Start overall timing
    overall_start_time = time.time()

    # Initialize results
    total_correct = 0
    total_samples = 0
    all_incorrect = []
    label_results_aggregate = {
        "Correct Answer": {"total": 0, "correct": 0},
        "Incorrect Answer": {"total": 0, "correct": 0},
        "Refusal": {"total": 0, "correct": 0}
    }
    api_errors = 0
    validation_errors = 0
    validation_error_responses = []

    # Initialize token usage tracking
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0

    # Initialize timing tracking
    total_processing_time = 0
    batch_processing_times = []

    # Initialize batch-level token usage tracking for median calculations
    batch_input_tokens = []
    batch_output_tokens = []
    batch_total_tokens = []

    # Initialize aggregated metrics for final calculation
    all_true_labels = []
    all_predicted_labels = []

    # Process batches in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=480) as executor:
        # Submit all tasks
        future_to_batch = {
            executor.submit(process_batch, i, batch, prompt_template, llm): (i, batch)
            for i, batch in enumerate(batches)
        }

        # Process results as they complete
        with tqdm(total=len(batches), desc="Processing batches") as pbar:
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_id, batch = future_to_batch[future]

                try:
                    result = future.result()

                    if result is None:
                        # Handle case where result is None
                        api_errors += 1
                        print(f"\nUnexpected None result in batch {batch_id}")
                        continue

                    if result["success"]:
                        # Update statistics
                        total_correct += result["correct"]
                        total_samples += result["total"]
                        all_incorrect.extend(result["incorrect_judgments"])

                        # Update label results
                        for label, data in result["label_results"].items():
                            label_results_aggregate[label]["total"] += data["total"]
                            label_results_aggregate[label]["correct"] += data["correct"]

                        # Update token usage
                        if "token_usage" in result:
                            token_usage = result["token_usage"]
                            batch_input = token_usage.get("input_tokens", 0)
                            batch_output = token_usage.get("output_tokens", 0)
                            batch_total = token_usage.get("total_tokens", 0)

                            total_input_tokens += batch_input
                            total_output_tokens += batch_output
                            total_tokens += batch_total

                            # Track batch-level token usage for median calculations
                            batch_input_tokens.append(batch_input)
                            batch_output_tokens.append(batch_output)
                            batch_total_tokens.append(batch_total)

                        # Update timing statistics
                        if "batch_processing_time" in result:
                            batch_time = result["batch_processing_time"]
                            batch_processing_times.append(batch_time)
                            total_processing_time += batch_time

                        # Collect labels for overall metrics calculation
                        if "precision_recall_f1" in result:
                            # Extract true and predicted labels from the batch for aggregation
                            for sample in batch:
                                true_label = sample['label']
                                if true_label == "Correct Answer":
                                    all_true_labels.append("Correct")
                                elif true_label == "Incorrect Answer":
                                    all_true_labels.append("Incorrect")
                                elif true_label == "Refusal":
                                    all_true_labels.append("No answer")

                            # Get judgments from confusion matrix data for predicted labels
                            if "confusion_matrix" in result and "matrix" in result["confusion_matrix"]:
                                # Extract judgments from the result - we need to reconstruct them from the batch processing
                                # This is a bit complex, but we can use the incorrect_judgments to help reconstruct
                                batch_predicted_labels = []
                                incorrect_judgments = result.get("incorrect_judgments", [])

                                # Reconstruct predicted labels for this batch
                                for i in range(result["total"]):
                                    # Check if this sample index appears in incorrect_judgments
                                    found_incorrect = False
                                    for idx, judgment, true_label, sample, judge_answer in incorrect_judgments:
                                        if idx == i + 1:  # incorrect_judgments uses 1-based indexing
                                            batch_predicted_labels.append(judgment)
                                            found_incorrect = True
                                            break

                                    if not found_incorrect:
                                        # This was a correct judgment, so predicted = true
                                        batch_true = batch[i]['label']
                                        if batch_true == "Correct Answer":
                                            batch_predicted_labels.append("Correct")
                                        elif batch_true == "Incorrect Answer":
                                            batch_predicted_labels.append("Incorrect")
                                        elif batch_true == "Refusal":
                                            batch_predicted_labels.append("No answer")

                                all_predicted_labels.extend(batch_predicted_labels)
                    else:
                        # Check if it's a validation error or API error
                        error_msg = result.get('error', 'Unknown error')
                        if 'raw_response' in result and result['raw_response']:
                            # This is a validation error - save the raw response
                            validation_errors += 1
                            print(f"\nValidation error in batch {batch_id}: {error_msg}")
                            validation_error_responses.append({
                                "batch_id": batch_id,
                                "samples": batch,
                                "error": error_msg,
                                "raw_response": result['raw_response']
                            })
                        else:
                            # This is an API error
                            api_errors += 1
                            print(f"\nAPI error in batch {batch_id}: {error_msg}")

                        # Still collect timing information for failed batches
                        if "batch_processing_time" in result:
                            batch_time = result["batch_processing_time"]
                            batch_processing_times.append(batch_time)
                            total_processing_time += batch_time

                except Exception as e:
                    api_errors += 1
                    print(f"\nUnexpected error in batch {batch_id}: {e}")

                # Update progress
                pbar.update(1)

                # Add a small delay to avoid rate limiting
                time.sleep(0.1)

    # Calculate overall processing time
    overall_end_time = time.time()
    overall_processing_time = overall_end_time - overall_start_time

    # Calculate overall precision, recall, F1, and confusion matrix if we have data
    overall_precision_recall_f1 = None
    overall_confusion_matrix = None

    if len(all_true_labels) > 0 and len(all_predicted_labels) > 0 and len(all_true_labels) == len(all_predicted_labels):
        overall_precision_recall_f1 = calculate_precision_recall_f1(all_true_labels, all_predicted_labels)
        overall_confusion_matrix = generate_confusion_matrix(all_true_labels, all_predicted_labels)

    # Print results
    print("\n===== Results =====")

    print(f"Model: {model_name}")

    # Calculate accuracy only on samples that were successfully evaluated
    accuracy = total_correct / total_samples if total_samples > 0 else 0

    # Calculate total samples that should have been processed
    total_expected_samples = len(batches) * batch_size

    # Calculate percentage of samples successfully processed
    processing_success_rate = total_samples / total_expected_samples if total_expected_samples > 0 else 0

    print(f"Processing success rate: {processing_success_rate:.2%} ({total_samples}/{total_expected_samples} samples)")
    print(f"Errors: {validation_errors} validation errors, {api_errors} API errors")
    print(f"Overall Accuracy: {accuracy:.2%} ({total_correct}/{total_samples}) of successfully processed samples")

    # Print token usage
    print(f"\nToken Usage:")
    print(f"Input tokens: {total_input_tokens:,}")
    print(f"Output tokens: {total_output_tokens:,}")
    print(f"Total tokens: {total_tokens:,}")
    if total_samples > 0:
        print(f"Average tokens per sample: {total_tokens / total_samples:.1f}")
        print(f"Average input tokens per sample: {total_input_tokens / total_samples:.1f}")
        print(f"Average output tokens per sample: {total_output_tokens / total_samples:.1f}")

    # Print batch-level token statistics
    if len(batch_input_tokens) > 0:
        print(f"Median input tokens per batch: {statistics.median(batch_input_tokens):.1f}")
        print(f"Median output tokens per batch: {statistics.median(batch_output_tokens):.1f}")
        print(f"Median total tokens per batch: {statistics.median(batch_total_tokens):.1f}")

    # Print timing statistics
    print(f"\nProcessing Time Statistics:")
    print(f"Overall processing time: {overall_processing_time:.2f} seconds")
    if len(batch_processing_times) > 0:
        avg_batch_time = sum(batch_processing_times) / len(batch_processing_times)
        median_batch_time = statistics.median(batch_processing_times)
        min_batch_time = min(batch_processing_times)
        max_batch_time = max(batch_processing_times)
        print(f"Average batch processing time: {avg_batch_time:.2f} seconds")
        print(f"Median batch processing time: {median_batch_time:.2f} seconds")
        print(f"Min batch processing time: {min_batch_time:.2f} seconds")
        print(f"Max batch processing time: {max_batch_time:.2f} seconds")
        if total_samples > 0:
            avg_sample_time = total_processing_time / total_samples
            print(f"Average processing time per sample: {avg_sample_time:.2f} seconds")

    # Print precision, recall, F1 scores
    if overall_precision_recall_f1:
        print(f"\nOverall Classification Metrics:")
        print(f"Macro-averaged Precision: {overall_precision_recall_f1['macro_avg']['precision']:.3f}")
        print(f"Macro-averaged Recall: {overall_precision_recall_f1['macro_avg']['recall']:.3f}")
        print(f"Macro-averaged F1-score: {overall_precision_recall_f1['macro_avg']['f1_score']:.3f}")
        print(f"Micro-averaged Precision: {overall_precision_recall_f1['micro_avg']['precision']:.3f}")
        print(f"Micro-averaged Recall: {overall_precision_recall_f1['micro_avg']['recall']:.3f}")
        print(f"Micro-averaged F1-score: {overall_precision_recall_f1['micro_avg']['f1_score']:.3f}")

        print(f"\nPer-class Metrics:")
        for class_name, metrics in overall_precision_recall_f1['per_class'].items():
            print(f"{class_name}:")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1-score: {metrics['f1_score']:.3f}")
            print(f"  Support: {metrics['support']}")

    print("\nResults by label type:")
    for label, data in label_results_aggregate.items():
        if data["total"] > 0:
            accuracy = data["correct"] / data["total"]
            print(f"{label}: {accuracy:.2%} accuracy ({data['correct']}/{data['total']})")

    # Create timestamp for filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create directories
    results_dir = "llm_judge_results"
    error_dir = os.path.join(results_dir, "errors")
    incorrect_judgments_dir = os.path.join(error_dir, "incorrect_judgments")
    validation_errors_dir = os.path.join(error_dir, "validation_errors")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(incorrect_judgments_dir, exist_ok=True)
    os.makedirs(validation_errors_dir, exist_ok=True)

    # Save validation errors if any
    if validation_errors > 0:
        validation_error_file = os.path.join(validation_errors_dir, f"validation_errors_{timestamp}.json")

        validation_error_data = {
            "timestamp": timestamp,
            "total_validation_errors": validation_errors,
            "note": "These samples were NOT included in accuracy calculations due to invalid response format",
            "errors": validation_error_responses
        }

        with open(validation_error_file, 'w', encoding='utf-8') as f:
            json.dump(validation_error_data, f, indent=2, ensure_ascii=False)

        print(f"Validation errors saved to {validation_error_file}")

    # Save failed judgments if any
    if all_incorrect:
        print(f"\nFound {len(all_incorrect)} incorrect judgments out of {total_samples} successfully processed samples.")

        incorrect_judgments_file = os.path.join(incorrect_judgments_dir, f"incorrect_judgments_{timestamp}.json")

        incorrect_judgments_data = {
            "timestamp": timestamp,
            "summary": {
                "total_samples_processed": total_samples,
                "incorrect_judgments": len(all_incorrect),
                "accuracy": accuracy,
                "validation_errors": validation_errors,
                "api_errors": api_errors
            },
            "incorrect_judgments": []
        }

        for idx, judgment, true_label, sample, judge_answer in all_incorrect:
            judgment_entry = {
                "sample_index": idx,
                "judge_decision": judgment,
                "true_label": true_label,
                "judge_answer": judge_answer,
                "sample_data": {
                    "question": sample['question'],
                    "model_answer": sample['model_answer'],
                    "gold_answer": sample['gold_answer'],
                    "label": sample['label']
                }
            }
            incorrect_judgments_data["incorrect_judgments"].append(judgment_entry)

        with open(incorrect_judgments_file, 'w', encoding='utf-8') as f:
            json.dump(incorrect_judgments_data, f, indent=2, ensure_ascii=False)

        print(f"Incorrect judgments saved to {incorrect_judgments_file}")

    accuracy = total_correct / total_samples if total_samples > 0 else 0

    # Save overall results summary
    results_summary_file = os.path.join(results_dir, f"llm_judge_results_{timestamp}.json")

    results_summary = {
        "model_name": model_name,
        "timestamp": timestamp,
        "test_mode": test_mode,
        "samples_processed": total_samples,
        "total_expected_samples": total_expected_samples,
        "processing_success_rate": processing_success_rate,
        "overall_accuracy": accuracy,
        "errors": {
            "validation_errors": validation_errors,
            "api_errors": api_errors
        },
        "token_usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "average_tokens_per_sample": total_tokens / total_samples if total_samples > 0 else 0,
            "average_input_tokens_per_sample": total_input_tokens / total_samples if total_samples > 0 else 0,
            "average_output_tokens_per_sample": total_output_tokens / total_samples if total_samples > 0 else 0,
            "median_input_tokens_per_batch": statistics.median(batch_input_tokens) if batch_input_tokens else 0,
            "median_output_tokens_per_batch": statistics.median(batch_output_tokens) if batch_output_tokens else 0,
            "median_total_tokens_per_batch": statistics.median(batch_total_tokens) if batch_total_tokens else 0
        },
        "processing_time": {
            "overall_processing_time_seconds": overall_processing_time,
            "average_batch_time_seconds": sum(batch_processing_times) / len(batch_processing_times) if batch_processing_times else 0,
            "median_batch_time_seconds": statistics.median(batch_processing_times) if batch_processing_times else 0,
            "min_batch_time_seconds": min(batch_processing_times) if batch_processing_times else 0,
            "max_batch_time_seconds": max(batch_processing_times) if batch_processing_times else 0,
            "average_time_per_sample_seconds": total_processing_time / total_samples if total_samples > 0 else 0,
            "total_batches_processed": len(batch_processing_times)
        },
        "precision_recall_f1": overall_precision_recall_f1,
        "confusion_matrix": overall_confusion_matrix,
        "accuracy_by_label": {}
    }

    # Add accuracy by label to summary
    for label, data in label_results_aggregate.items():
        if data["total"] > 0:
            label_accuracy = data["correct"] / data["total"]
            results_summary["accuracy_by_label"][label] = {
                "accuracy": label_accuracy,
                "correct": data["correct"],
                "total": data["total"]
            }

    with open(results_summary_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"Results summary saved to {results_summary_file}")

if __name__ == "__main__":
    main()