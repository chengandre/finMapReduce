import concurrent.futures
import datetime
import glob
import json
import multiprocessing
import os
import random
import time

from langchain.prompts import load_prompt
from tqdm import tqdm

from utils import GPT

def load_all_samples():
    """Load all samples from all JSONL files in the results directory"""
    results_dir = 'financebench/results'
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
            # Store sample index (1-based), judge's judgment, true label, and the full sample
            incorrect_judgments.append((i+1, judgment, true_label, samples[i]))

    # Return the results as a dictionary
    accuracy = correct / len(samples) if samples else 0
    return {
        "success": True,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(samples),
        "incorrect_judgments": incorrect_judgments
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

def process_batch(batch_id, samples_batch, prompt_template):
    """Process a batch of samples using the GPT wrapper"""
    # Format samples into context
    context = format_context(samples_batch)

    llm = GPT(
        model_name="deepseek/deepseek-r1-0528:free",
        temperature=0.01,
        max_tokens=8000,
        provider="openrouter",
        key=None
    )

    try:
        response = llm(prompt_template, context=context)

        # Get both the parsed JSON and raw response from the wrapper
        judge_response_json = response.get('json', {})
        raw_response = response.get('raw_response')
        raw_content = raw_response.content if raw_response and hasattr(raw_response, 'content') else str(response)

        # Evaluate the judge's performance with validation
        eval_results = evaluate_judge_performance(judge_response_json, samples_batch, raw_content)

        if eval_results["success"]:
            label_results = analyze_results_by_label(judge_response_json, samples_batch)
            eval_results["label_results"] = label_results

        return eval_results

    except Exception as e:
        return {
            "success": False,
            "error": f"Error processing batch {batch_id}: {str(e)}",
            "raw_response": str(e)
        }

def main():
    # Load all samples
    all_samples = load_all_samples()

    if not all_samples:
        print("No samples found. Please check the data files.")
        return

    # Test mode - limit to 20 samples
    test_mode = True  # Set to False to process all samples
    max_samples = 20  # Number of samples to process in test mode

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
    prompt_template = load_prompt('map_reduce/judge_prompt.yml')

    # Batch size for processing
    batch_size = 5  # Each batch contains 5 samples as per the prompt format

    # Split samples into batches
    batches = [all_samples[i:i+batch_size] for i in range(0, len(all_samples), batch_size)]

    # If the last batch has fewer than 5 samples, remove it or pad it
    if len(batches[-1]) < 5:
        batches.pop()

    print(f"Processing {len(batches)} batches with {len(batches) * batch_size} samples total")

    max_workers = 10

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

    # Process batches in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_batch = {
            executor.submit(process_batch, i, batch, prompt_template): (i, batch)
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

                except Exception as e:
                    api_errors += 1
                    print(f"\nUnexpected error in batch {batch_id}: {e}")

                # Update progress
                pbar.update(1)

                # Add a small delay to avoid rate limiting
                time.sleep(0.1)

    # Print results
    print("\n===== Results =====")

    # Calculate accuracy only on samples that were successfully evaluated
    accuracy = total_correct / total_samples if total_samples > 0 else 0

    # Calculate total samples that should have been processed
    total_expected_samples = len(batches) * batch_size

    # Calculate percentage of samples successfully processed
    processing_success_rate = total_samples / total_expected_samples if total_expected_samples > 0 else 0

    print(f"Processing success rate: {processing_success_rate:.2%} ({total_samples}/{total_expected_samples} samples)")
    print(f"Errors: {validation_errors} validation errors, {api_errors} API errors")
    print(f"Overall Accuracy: {accuracy:.2%} ({total_correct}/{total_samples}) of successfully processed samples")

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

        for idx, judgment, true_label, sample in all_incorrect:
            judgment_entry = {
                "sample_index": idx,
                "judge_decision": judgment,
                "true_label": true_label,
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

    # Save overall results summary
    results_summary_file = os.path.join(results_dir, f"llm_judge_results_{timestamp}.json")

    results_summary = {
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