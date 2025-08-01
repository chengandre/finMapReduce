from langchain.prompts import load_prompt
from utils import load_pdf_chunk, load_markdown_chunk, calculate_token_usage_summary, calculate_accuracy_by_question_type
from datetime import datetime
from tqdm import tqdm

import concurrent.futures
import time
import json
import os


def preprocess_results(results):
    """
    filters the score of the results in the intermediate stage
    """
    modified_results = []
    for result in results:
        score = result.get("json", {}).get("relevance_score", 0)
        if score > 7:
            modified_results.append(result)

    return modified_results


def mapreduce_qa_documents(llm, chunked_docs, final_query, map_query):
    """
    The function helps in querying pdf for questions

    Args:
        llm: the langauge model on which queries/summarsation would be done on
        documents (list): the pages on which summarisation would work on
        query (str): the question to be asked for the document

    Returns:
        result_final (str): the final result of the entire query
        results (list): the intermediate response for the question asked
        total_docs (int): total chunks used to process the pdf
        time_to_process (float): the time taken to process the entire file
        token_stats (dict): statistics of token usage for map and reduce phases

    """
    # Split the documents into smaller chunks

    t1 = time.time()

    # Load prompts from YAML
    map_prompt = load_prompt("prompts/map_prompt.yml")
    total_docs = len(chunked_docs)

    # Track map phase token usage
    map_input_tokens = 0
    map_output_tokens = 0

    def process_chunk(chunk):
        return llm(map_prompt, context=chunk, final_query=final_query)

    print("Map phase started, calling LLMs")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_chunk, chunked_docs))
    print("Map phase completed")
    print("Results:")
    print(results)

    # Count tokens from map phase using usage_metadata from raw_response
    for result in results:
        raw_response = result.get('raw_response')
        if raw_response and hasattr(raw_response, 'usage_metadata') and raw_response.usage_metadata:
            map_input_tokens += raw_response.usage_metadata.get("input_tokens", 0)
            map_output_tokens += raw_response.usage_metadata.get("output_tokens", 0)

    # modified_results = preprocess_results(results=results)

    # Load reduce prompt
    reduce_prompt = load_prompt("prompts/reduce_prompt.yml")

    # Extract content from results for reduce phase - handle JSON format with XML structure
    processed_results = []
    for i, result in enumerate(results, 1):
        # Get JSON data from the wrapper result
        result_json = result.get('json', {})
        if result_json:
            # Format the JSON data as XML for the reduce phase
            chunk_xml = f"      <chunk_{i}>\n"
            chunk_xml += f"        <summary>{result_json.get('summary', '')}</summary>\n"
            chunk_xml += f"        <terms>{result_json.get('terms', [])}</terms>\n"
            chunk_xml += f"        <evidence>{result_json.get('evidence', [])}</evidence>\n"
            chunk_xml += f"        <answer>{result_json.get('answer', '')}</answer>\n"
            chunk_xml += f"        <relevance_score>{result_json.get('relevance_score', 0)}</relevance_score>\n"
            chunk_xml += f"      </chunk_{i}>"
            processed_results.append(chunk_xml)
        else:
            # Fallback to raw response content in XML format
            raw_response = result.get('raw_response')
            if raw_response:
                content = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
                chunk_xml = f"      <chunk_{i}>\n"
                chunk_xml += f"        <summary>Raw response content</summary>\n"
                chunk_xml += f"        <terms>[]</terms>\n"
                chunk_xml += f"        <evidence>[\"{content}\"]</evidence>\n"
                chunk_xml += f"        <answer>{content}</answer>\n"
                chunk_xml += f"        <relevance_score>0</relevance_score>\n"
                chunk_xml += f"      </chunk_{i}>"
                processed_results.append(chunk_xml)

    results_text = "\n".join(processed_results)
    result_final = llm(reduce_prompt, map_results=results_text, final_query=final_query)

    # Count tokens from reduce phase using usage_metadata from raw_response
    reduce_input_tokens = 0
    reduce_output_tokens = 0
    final_raw_response = result_final.get('raw_response')
    if final_raw_response and hasattr(final_raw_response, 'usage_metadata') and final_raw_response.usage_metadata:
        reduce_input_tokens = final_raw_response.usage_metadata.get("input_tokens", 0)
        reduce_output_tokens = final_raw_response.usage_metadata.get("output_tokens", 0)

    time_to_process = time.time() - t1

    # Prepare token statistics
    token_stats = {
        "map_phase": {
            "input_tokens": map_input_tokens,
            "output_tokens": map_output_tokens
        },
        "reduce_phase": {
            "input_tokens": reduce_input_tokens,
            "output_tokens": reduce_output_tokens
        },
        "total": {
            "input_tokens": map_input_tokens + reduce_input_tokens,
            "output_tokens": map_output_tokens + reduce_output_tokens
        }
    }

    # print(modified_results)
    return result_final, results, total_docs, time_to_process, token_stats


def process_mapreduce_qa(files, selected_questions_dict, model_name, llm,
                         chunk_size, token_overlap, method="marker"):
    documents, token_count = [], 0
    for i in range(len(files)):
        temp_documents, temp_token_count = load_pdf_chunk(files[i], chunk_size, token_overlap, method=method)
        documents += temp_documents
        token_count += temp_token_count
    print("Documents loaded")
    final_answer_responses = []
    for final_query, map_query in selected_questions_dict.items():
        response_dict = {"chain": "MapReduce", "execution_time": datetime.now(), "model_name": model_name,
                         "query": final_query}

        response_dict["answer"], response_dict["int_result"], response_dict["total_chunks"], \
            response_dict["time_taken"], response_dict["token_stats"] = mapreduce_qa_documents(llm, documents, final_query, map_query)
        final_answer_responses.append(response_dict)
    return final_answer_responses


def load_financebench_data(jsonl_path, num_samples=None):
    """
    Load QA pairs from financebench_open_source.jsonl file

    Args:
        jsonl_path (str): Path to the financebench jsonl file
        num_samples (int): Number of samples to load

    Returns:
        list: List of QA dictionaries with necessary information
    """
    qa_data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    count = 0
    for line in lines:
        if num_samples is not None and count >= num_samples:
            break

        item = json.loads(line)

        # Create a QA pair with the necessary information
        qa_pair = {
            "doc_name": item["doc_name"],
            "question": item["question"],
            "answer": item["answer"],
            "justification": item["justification"] if item["justification"] else "No justification provided",
            "evidence": [ev["evidence_text"] for ev in item["evidence"]],
            "question_type": item["question_type"],
            "question_reasoning": item["question_reasoning"]
        }

        qa_data.append(qa_pair)
        count += 1

    return qa_data


def load_finqa_data(json_path, num_samples=None):
    """
    Load QA pairs from FinQA json file

    Args:
        json_path (str): Path to the FinQA json file
        num_samples (int): Number of samples to load

    Returns:
        list: List of QA dictionaries with necessary information
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    qa_data = []
    count = 0
    
    for item in data:
        if num_samples is not None and count >= num_samples:
            break

        # Create a QA pair with the necessary information
        qa_pair = {
            "doc_name": item["doc_name"],
            "question": item["question"],
            "answer": item["answer"],
            # "pretext": item.get("pretext", []),
            # "posttext": item.get("posttext", []),
            "filename": item.get("filename", ""),
            "explanation": item.get("explanation", "")
        }

        qa_data.append(qa_pair)
        count += 1

    return qa_data


def evaluate_with_llm_judge(llm, qa_data, batch_size=5):
    """
    Evaluate model answers using an LLM judge with multithreading

    Args:
        llm: LLM to use for judging
        qa_data (list): List of QA dictionaries containing question, golden answer and llm_answer
        batch_size (int): Number of samples to judge in one batch

    Returns:
        dict: Evaluation results with scores and detailed judgments
    """
    # Load the judge prompt template
    judge_prompt = load_prompt("prompts/judge_prompt.yml")

    # Prepare batches for evaluation
    total_samples = len(qa_data)
    batches = []

    for i in range(0, total_samples, batch_size):
        batch = []
        end_idx = min(i + batch_size, total_samples)

        for j in range(i, end_idx):
            sample = {
                "llm_evidence": qa_data[j]["llm_evidence"],
                "llm_reasoning": qa_data[j]["llm_reasoning"],
                "llm_answer": qa_data[j]["llm_answer"],
                "golden_answer": qa_data[j]["answer"],
                "question": qa_data[j]["question"]
            }
            batch.append(sample)

        batches.append((i // batch_size, batch))  # Include batch index for tracking

    def _process_batch(batch_data):
        """Process a single batch of evaluations"""
        batch_idx, batch = batch_data

        # Format the context for the judge prompt
        context_parts = []
        for i, sample in enumerate(batch, 1):

            item_block = (
                f"  <item>\n"
                f"    <item_number>{i}</item_number>\n"
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
            context_parts.append(item_block)

        # Wrap all items in a single root tag
        context = "<evaluation_items>\n" + "\n".join(context_parts) + "\n</evaluation_items>"

        try:
            # Get the judge's response
            judge_response = llm(judge_prompt, context=context)

            # Parse JSON response from the wrapper
            evaluation_data = judge_response.get('json', {})

            return {
                "batch_idx": batch_idx,
                "success": True,
                "evaluation_data": evaluation_data,
                "batch": batch
            }
        except Exception as e:
            print(f"Error processing batch {batch_idx + 1}: {str(e)}")
            fallback = {"evaluation_results": [
                {"evaluation_number": i+1,
                 "reasoning": f"Processing error: {str(e)}",
                 "judgement": "Error"}
                for i in range(len(batch))
            ]}

            return {
                "batch_idx": batch_idx,
                "success": False,
                "evaluation_data": fallback,
                "batch": batch,
                "error": str(e)
            }

    print(f"Judge evaluation: Processing {len(batches)} batches with multithreading...")

    # Process batches in parallel
    all_judgments = []
    batch_results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # Submit all batch processing tasks
        future_to_batch = {executor.submit(_process_batch, batch_data): batch_data for batch_data in batches}

        # Collect results as they complete with a progress bar
        with tqdm(total=len(batches), desc="Evaluating batches") as pbar:
            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    result = future.result()
                    batch_results.append(result)
                    batch_idx = result['batch_idx']
                    pbar.update(1)
                    if not result['success']:
                        pbar.write(f"Error in batch {batch_idx + 1}: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    batch_data = future_to_batch[future]
                    batch_idx = batch_data[0]
                    pbar.write(f"Error in judge batch {batch_idx + 1}: {e}")
                    pbar.update(1)

    # Sort results by batch index to maintain order
    batch_results.sort(key=lambda x: x['batch_idx'])

    # Apply results back to qa_data
    for result in batch_results:
        batch_idx = result['batch_idx']
        evaluation_data = result['evaluation_data']
        all_judgments.append(evaluation_data)

        # Add judgment results back to qa_data
        for i, eval_result in enumerate(evaluation_data.get("evaluation_results", [])):
            qa_idx = batch_idx * batch_size + i
            if qa_idx < len(qa_data):
                qa_data[qa_idx]["judgment"] = eval_result.get("judgement", "Error")
                qa_data[qa_idx]["reasoning"] = eval_result.get("reasoning", "No reasoning provided")

    # Calculate overall statistics
    total_correct = 0
    total_coherent = 0
    total_deviated = 0
    total_incorrect = 0
    total_no_answer = 0
    total_samples = 0

    for qa_item in qa_data:
        judgment = qa_item.get("judgment", "Error")
        if judgment == "Correct":
            total_correct += 1
        elif judgment == "Coherent":
            total_coherent += 1
        elif judgment == "Deviated":
            total_deviated += 1
        elif judgment == "Incorrect":
            total_incorrect += 1
        elif judgment == "No answer":
            total_no_answer += 1
        total_samples += 1

    # Prepare overall results
    results = {
        "correct": total_correct,
        "coherent": total_coherent,
        "deviated": total_deviated,
        "incorrect": total_incorrect,
        "no_answer": total_no_answer,
        "total": total_samples,
        "accuracy": total_correct / total_samples if total_samples > 0 else 0,
        "detailed_judgments": all_judgments
    }

    return results


def process_single_qa(qa_pair, llm, chunk_size=36000, chunk_overlap=1000):
    """
    Process a single QA pair from financebench.

    Args:
        qa_pair (dict): QA pair dictionary containing question and doc_name
        llm: LLM instance to use
        chunk_size (int): Size of each document chunk
        chunk_overlap (int): Overlap between document chunks

    Returns:
        dict: Updated QA pair with LLM answer and token stats
    """
    # Get document name and question from the qa_pair
    doc_name = qa_pair["doc_name"]
    question = qa_pair["question"]

    # Load document chunks
    docs, token_count = load_pdf_chunk(doc_name, chunk_size, chunk_overlap, method="marker")

    # Load prompts from YAML
    map_prompt = load_prompt("prompts/map_prompt.yml")

    # Track map phase token usage
    map_input_tokens = 0
    map_output_tokens = 0

    def process_chunk(chunk):
        return llm(map_prompt, context=chunk.page_content, final_query=question)

    # Process chunks in parallel with executor
    chunks_count = len(docs)
    results = []

    # Use a smaller progress bar description to fit in console
    doc_basename = os.path.basename(doc_name)
    if len(doc_basename) > 20:
        doc_basename = doc_basename[:17] + "..."

    with concurrent.futures.ThreadPoolExecutor(min(chunks_count, 10)) as executor:
        # Submit all chunk processing tasks
        futures = {
            executor.submit(process_chunk, chunk): i
            for i, chunk in enumerate(docs)
        }

        # Process as completed with silent progress tracking
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # If a chunk fails, add a placeholder result
                print(f"Error processing chunk in {doc_basename}: {e}")
                results.append({"error": str(e)})

    # Count output tokens from map phase
    for result in results:
        raw_response = result.get('raw_response')
        if raw_response and hasattr(raw_response, 'usage_metadata') and raw_response.usage_metadata:
            map_input_tokens += raw_response.usage_metadata["input_tokens"]
            map_output_tokens += raw_response.usage_metadata["output_tokens"]

    # Load reduce prompt
    reduce_prompt = load_prompt("prompts/reduce_prompt.yml")

    # Process map results for reduce phase - handle JSON format with XML structure
    processed_results = []
    for i, result in enumerate(results, 1):
        # Get JSON data from the wrapper result
        result_json = result.get('json', {})
        if result_json:
            # Format the JSON data as XML for the reduce phase
            chunk_xml = f"      <chunk_{i}>\n"
            chunk_xml += f"        <summary>{result_json.get('summary', '')}</summary>\n"
            chunk_xml += f"        <terms>{result_json.get('terms', [])}</terms>\n"
            chunk_xml += f"        <evidence>{result_json.get('evidence', [])}</evidence>\n"
            chunk_xml += f"        <answer>{result_json.get('answer', '')}</answer>\n"
            chunk_xml += f"        <relevance_score>{result_json.get('relevance_score', 0)}</relevance_score>\n"
            chunk_xml += f"      </chunk_{i}>"
            processed_results.append(chunk_xml)
        else:
            # Fallback to raw response content in XML format
            raw_response = result.get('raw_response')
            if raw_response:
                content = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
                chunk_xml = f"      <chunk_{i}>\n"
                chunk_xml += f"        <summary>Raw response content</summary>\n"
                chunk_xml += f"        <terms>[]</terms>\n"
                chunk_xml += f"        <evidence>[\"{content}\"]</evidence>\n"
                chunk_xml += f"        <answer>{content}</answer>\n"
                chunk_xml += f"        <relevance_score>0</relevance_score>\n"
                chunk_xml += f"      </chunk_{i}>"
                processed_results.append(chunk_xml)

    results_text = "\n".join(processed_results)
    result_final = llm(reduce_prompt, map_results=results_text, final_query=question)

    # Get token usage from raw response
    final_raw_response = result_final.get('raw_response')
    reduce_input_tokens = 0
    reduce_output_tokens = 0
    if final_raw_response and hasattr(final_raw_response, 'usage_metadata') and final_raw_response.usage_metadata:
        reduce_input_tokens = final_raw_response.usage_metadata["input_tokens"]
        reduce_output_tokens = final_raw_response.usage_metadata["output_tokens"]

    # Parse JSON response from reduce phase
    reduce_json = result_final.get('json', {})
    if reduce_json:
        clean_answer = reduce_json.get("answer", "")
        clean_reasoning = reduce_json.get("reasoning", "No reasoning provided")
        clean_evidence = reduce_json.get("evidence", [])
    else:
        # Fallback to raw response content
        if final_raw_response:
            clean_answer = final_raw_response.content if hasattr(final_raw_response, 'content') else str(final_raw_response)
        else:
            clean_answer = str(result_final)
        clean_reasoning = "No reasoning provided"
        clean_evidence = []

    # Store the LLM answer and reasoning directly in the qa_pair dictionary
    qa_pair["llm_answer"] = clean_answer
    qa_pair["llm_reasoning"] = clean_reasoning
    qa_pair["llm_evidence"] = clean_evidence

    # Store token usage statistics
    qa_pair["token_stats"] = {
        "map_phase": {
            "input_tokens": map_input_tokens,
            "output_tokens": map_output_tokens
        },
        "reduce_phase": {
            "input_tokens": reduce_input_tokens,
            "output_tokens": reduce_output_tokens
        },
        "total": {
            "input_tokens": map_input_tokens + reduce_input_tokens,
            "output_tokens": map_output_tokens + reduce_output_tokens
        }
    }

    # Print completion message for this document
    # print(f"Completed processing: {os.path.basename(doc_name)}")

    return qa_pair


def process_single_finqa(qa_pair, llm, doc_dir, chunk_size=36000, chunk_overlap=1000):
    """
    Process a single QA pair from FinQA.

    Args:
        qa_pair (dict): QA pair dictionary containing question and doc_name
        llm: LLM instance to use
        doc_dir (str): Directory containing FinQA markdown documents
        chunk_size (int): Size of each document chunk
        chunk_overlap (int): Overlap between document chunks

    Returns:
        dict: Updated QA pair with LLM answer and token stats
    """
    # Get document name and question from the qa_pair
    doc_name = qa_pair["doc_name"]
    question = qa_pair["question"]
    
    # Construct full path to markdown file
    markdown_file = os.path.join(doc_dir, doc_name)

    # Load document chunks
    docs, token_count = load_markdown_chunk(markdown_file, chunk_size, chunk_overlap)
    
    if not docs:
        # Handle case where document couldn't be loaded
        print(f"Error loading document: {markdown_file}")
        qa_pair["llm_answer"] = "Error: Could not load document"
        qa_pair["llm_reasoning"] = "Document loading failed"
        qa_pair["llm_evidence"] = []
        qa_pair["token_stats"] = {
            "map_phase": {"input_tokens": 0, "output_tokens": 0},
            "reduce_phase": {"input_tokens": 0, "output_tokens": 0},
            "total": {"input_tokens": 0, "output_tokens": 0}
        }
        return qa_pair

    # Load prompts from YAML
    map_prompt = load_prompt("prompts/map_prompt.yml")

    # Track map phase token usage
    map_input_tokens = 0
    map_output_tokens = 0

    def process_chunk(chunk):
        return llm(map_prompt, context=chunk.page_content, final_query=question)

    # Process chunks in parallel with executor
    chunks_count = len(docs)
    results = []

    # Use a smaller progress bar description to fit in console
    doc_basename = os.path.basename(doc_name)
    if len(doc_basename) > 20:
        doc_basename = doc_basename[:17] + "..."

    with concurrent.futures.ThreadPoolExecutor(min(chunks_count, 10)) as executor:
        # Submit all chunk processing tasks
        futures = {
            executor.submit(process_chunk, chunk): i
            for i, chunk in enumerate(docs)
        }

        # Process as completed with silent progress tracking
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # If a chunk fails, add a placeholder result
                print(f"Error processing chunk in {doc_basename}: {e}")
                results.append({"error": str(e)})

    # Count output tokens from map phase
    for result in results:
        raw_response = result.get('raw_response')
        if raw_response and hasattr(raw_response, 'usage_metadata') and raw_response.usage_metadata:
            map_input_tokens += raw_response.usage_metadata["input_tokens"]
            map_output_tokens += raw_response.usage_metadata["output_tokens"]

    # Load reduce prompt
    reduce_prompt = load_prompt("prompts/reduce_prompt.yml")

    # Process map results for reduce phase - handle JSON format with XML structure
    processed_results = []
    for i, result in enumerate(results, 1):
        # Get JSON data from the wrapper result
        result_json = result.get('json', {})
        if result_json:
            # Format the JSON data as XML for the reduce phase
            chunk_xml = f"      <chunk_{i}>\n"
            chunk_xml += f"        <summary>{result_json.get('summary', '')}</summary>\n"
            chunk_xml += f"        <terms>{result_json.get('terms', [])}</terms>\n"
            chunk_xml += f"        <evidence>{result_json.get('evidence', [])}</evidence>\n"
            chunk_xml += f"        <answer>{result_json.get('answer', '')}</answer>\n"
            chunk_xml += f"        <relevance_score>{result_json.get('relevance_score', 0)}</relevance_score>\n"
            chunk_xml += f"      </chunk_{i}>"
            processed_results.append(chunk_xml)
        else:
            # Fallback to raw response content in XML format
            raw_response = result.get('raw_response')
            if raw_response:
                content = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
                chunk_xml = f"      <chunk_{i}>\n"
                chunk_xml += f"        <summary>Raw response content</summary>\n"
                chunk_xml += f"        <terms>[]</terms>\n"
                chunk_xml += f"        <evidence>[\"{content}\"]</evidence>\n"
                chunk_xml += f"        <answer>{content}</answer>\n"
                chunk_xml += f"        <relevance_score>0</relevance_score>\n"
                chunk_xml += f"      </chunk_{i}>"
                processed_results.append(chunk_xml)

    results_text = "\n".join(processed_results)
    result_final = llm(reduce_prompt, map_results=results_text, final_query=question)

    # Get token usage from raw response
    final_raw_response = result_final.get('raw_response')
    reduce_input_tokens = 0
    reduce_output_tokens = 0
    if final_raw_response and hasattr(final_raw_response, 'usage_metadata') and final_raw_response.usage_metadata:
        reduce_input_tokens = final_raw_response.usage_metadata["input_tokens"]
        reduce_output_tokens = final_raw_response.usage_metadata["output_tokens"]

    # Parse JSON response from reduce phase
    reduce_json = result_final.get('json', {})
    if reduce_json:
        clean_answer = reduce_json.get("answer", "")
        clean_reasoning = reduce_json.get("reasoning", "No reasoning provided")
        clean_evidence = reduce_json.get("evidence", [])
    else:
        # Fallback to raw response content
        if final_raw_response:
            clean_answer = final_raw_response.content if hasattr(final_raw_response, 'content') else str(final_raw_response)
        else:
            clean_answer = str(result_final)
        clean_reasoning = "No reasoning provided"
        clean_evidence = []

    # Store the LLM answer and reasoning directly in the qa_pair dictionary
    qa_pair["llm_answer"] = clean_answer
    qa_pair["llm_reasoning"] = clean_reasoning
    qa_pair["llm_evidence"] = clean_evidence

    # Store token usage statistics
    qa_pair["token_stats"] = {
        "map_phase": {
            "input_tokens": map_input_tokens,
            "output_tokens": map_output_tokens
        },
        "reduce_phase": {
            "input_tokens": reduce_input_tokens,
            "output_tokens": reduce_output_tokens
        },
        "total": {
            "input_tokens": map_input_tokens + reduce_input_tokens,
            "output_tokens": map_output_tokens + reduce_output_tokens
        }
    }

    return qa_pair


def process_financebench_qa(jsonl_path, model_name, llm, num_samples=None, chunk_size=36000, chunk_overlap=1000, max_concurrent_qa=3):
    """
    Process QA from financebench with parallel processing of QA pairs.

    Args:
        jsonl_path (str): Path to financebench jsonl file
        model_name (str): Name of the LLM model
        llm: LLM instance to use
        num_samples (int): Number of samples to process
        chunk_size (int): Size of each document chunk
        chunk_overlap (int): Overlap between document chunks
        max_concurrent_qa (int): Maximum number of QA pairs to process concurrently

    Returns:
        dict: Results containing model answers, golden answers, and evaluation results
    """
    print(f"Loading {num_samples if num_samples else 'all'} samples from financebench data...")
    qa_data = load_financebench_data(jsonl_path, num_samples)

    # # Print all document names before processing
    # print("\n=== Documents to be processed ===")
    # doc_names = [qa_pair["doc_name"] for qa_pair in qa_data]
    # for i, doc_name in enumerate(doc_names):
    #     print(f"{i+1}/{len(doc_names)}: {os.path.basename(doc_name)}")
    # print("===============================\n")

    t1 = time.time()
    print(f"Processing {len(qa_data)} QA pairs with {max_concurrent_qa} concurrent workers...")

    # Process QA pairs in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_qa) as executor:
        # Submit all QA processing tasks
        future_to_qa = {
            executor.submit(process_single_qa, qa_pair, llm, chunk_size, chunk_overlap): i
            for i, qa_pair in enumerate(qa_data)
        }

        # Create progress bar
        with tqdm(total=len(qa_data), desc="Processing QA pairs", unit="pair") as pbar:
            for future in concurrent.futures.as_completed(future_to_qa):
                qa_idx = future_to_qa[future]
                try:
                    updated_qa_pair = future.result()
                    doc_name = qa_data[qa_idx]['doc_name']
                    pbar.update(1)
                    pbar.set_postfix({"file": os.path.basename(doc_name)})
                except Exception as e:
                    pbar.write(f"Error processing QA pair {qa_idx+1}: {e}")
                    qa_data[qa_idx]["llm_answer"] = "Error during processing"
                    qa_data[qa_idx]["error"] = str(e)
                    pbar.update(1)

    process_time = time.time() - t1
    print(f"QA processing completed in {process_time:.1f} seconds ({process_time/len(qa_data):.1f}s per question)")

    # Evaluate using LLM judge
    print("Evaluating answers using LLM judge...")
    evaluation_results = evaluate_with_llm_judge(llm, qa_data)

    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "financebench_results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"results_{timestamp}.json")

    # Calculate enhanced statistics
    token_summary = calculate_token_usage_summary(qa_data)
    accuracy_by_type = calculate_accuracy_by_question_type(qa_data)
    
    results = {
        "approach": "MapReduce",
        "model_name": model_name,
        "execution_time": datetime.now().isoformat(),
        "time_taken": time.time() - t1,
        "num_samples": len(qa_data),
        "qa_data": qa_data,
        
        # Enhanced: Token usage summary
        "token_usage_summary": token_summary,
        
        "mapreduce_config": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "max_concurrent_qa": max_concurrent_qa
        },
        
        # Enhanced: Detailed evaluation summary
        "evaluation_summary": {
            "judgment_distribution": {
                "correct": evaluation_results["correct"],
                "coherent": evaluation_results["coherent"],
                "deviated": evaluation_results["deviated"], 
                "incorrect": evaluation_results["incorrect"],
                "no_answer": evaluation_results["no_answer"]
            },
            "total": evaluation_results["total"],
            "accuracy": evaluation_results["accuracy"],
            
            # Enhanced: Accuracy by question type
            "accuracy_by_question_type": accuracy_by_type,
            
            # Enhanced: Judgment percentages
            "judgment_percentages": {
                "correct": evaluation_results["correct"] / evaluation_results["total"] if evaluation_results["total"] > 0 else 0,
                "coherent": evaluation_results["coherent"] / evaluation_results["total"] if evaluation_results["total"] > 0 else 0,
                "deviated": evaluation_results["deviated"] / evaluation_results["total"] if evaluation_results["total"] > 0 else 0,
                "incorrect": evaluation_results["incorrect"] / evaluation_results["total"] if evaluation_results["total"] > 0 else 0,
                "no_answer": evaluation_results["no_answer"] / evaluation_results["total"] if evaluation_results["total"] > 0 else 0
            },
            
            # Keep original detailed_judgments for backward compatibility
            "detailed_judgments": evaluation_results.get("detailed_judgments", [])
        }
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")
    print(f"Accuracy: {evaluation_results['accuracy']:.2f}")

    return results


def process_finqa_qa(json_path, doc_dir, model_name, llm, num_samples=None, chunk_size=36000, chunk_overlap=1000, max_concurrent_qa=20):
    """
    Process QA from FinQA with parallel processing of QA pairs.

    Args:
        json_path (str): Path to FinQA json file
        doc_dir (str): Directory containing FinQA markdown documents
        model_name (str): Name of the LLM model
        llm: LLM instance to use
        num_samples (int): Number of samples to process
        chunk_size (int): Size of each document chunk
        chunk_overlap (int): Overlap between document chunks
        max_concurrent_qa (int): Maximum number of QA pairs to process concurrently

    Returns:
        dict: Results containing model answers, golden answers, and evaluation results
    """
    print(f"Loading {num_samples if num_samples else 'all'} samples from FinQA data...")
    qa_data = load_finqa_data(json_path, num_samples)

    t1 = time.time()
    print(f"Processing {len(qa_data)} QA pairs with {max_concurrent_qa} concurrent workers...")

    # Process QA pairs in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_qa) as executor:
        # Submit all QA processing tasks
        future_to_qa = {
            executor.submit(process_single_finqa, qa_pair, llm, doc_dir, chunk_size, chunk_overlap): i
            for i, qa_pair in enumerate(qa_data)
        }

        # Create progress bar
        with tqdm(total=len(qa_data), desc="Processing QA pairs", unit="pair") as pbar:
            for future in concurrent.futures.as_completed(future_to_qa):
                qa_idx = future_to_qa[future]
                try:
                    updated_qa_pair = future.result()
                    doc_name = qa_data[qa_idx]['doc_name']
                    pbar.update(1)
                    pbar.set_postfix({"file": os.path.basename(doc_name)})
                except Exception as e:
                    pbar.write(f"Error processing QA pair {qa_idx+1}: {e}")
                    qa_data[qa_idx]["llm_answer"] = "Error during processing"
                    qa_data[qa_idx]["error"] = str(e)
                    pbar.update(1)

    process_time = time.time() - t1
    print(f"QA processing completed in {process_time:.1f} seconds ({process_time/len(qa_data):.1f}s per question)")

    # Evaluate using LLM judge
    print("Evaluating answers using LLM judge...")
    evaluation_results = evaluate_with_llm_judge(llm, qa_data)

    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "finqa_results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"finqa_results_{timestamp}.json")

    # Calculate enhanced statistics
    token_summary = calculate_token_usage_summary(qa_data)
    
    results = {
        "approach": "MapReduce",
        "dataset": "FinQA",
        "model_name": model_name,
        "execution_time": datetime.now().isoformat(),
        "time_taken": time.time() - t1,
        "num_samples": len(qa_data),
        "qa_data": qa_data,
        
        # Enhanced: Token usage summary
        "token_usage_summary": token_summary,
        
        "mapreduce_config": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "max_concurrent_qa": max_concurrent_qa
        },
        
        # Enhanced: Detailed evaluation summary
        "evaluation_summary": {
            "judgment_distribution": {
                "correct": evaluation_results["correct"],
                "coherent": evaluation_results["coherent"],
                "deviated": evaluation_results["deviated"], 
                "incorrect": evaluation_results["incorrect"],
                "no_answer": evaluation_results["no_answer"]
            },
            "total": evaluation_results["total"],
            "accuracy": evaluation_results["accuracy"],
            
            # Enhanced: Judgment percentages
            "judgment_percentages": {
                "correct": evaluation_results["correct"] / evaluation_results["total"] if evaluation_results["total"] > 0 else 0,
                "coherent": evaluation_results["coherent"] / evaluation_results["total"] if evaluation_results["total"] > 0 else 0,
                "deviated": evaluation_results["deviated"] / evaluation_results["total"] if evaluation_results["total"] > 0 else 0,
                "incorrect": evaluation_results["incorrect"] / evaluation_results["total"] if evaluation_results["total"] > 0 else 0,
                "no_answer": evaluation_results["no_answer"] / evaluation_results["total"] if evaluation_results["total"] > 0 else 0
            },
            
            # Keep original detailed_judgments for backward compatibility
            "detailed_judgments": evaluation_results.get("detailed_judgments", [])
        }
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")
    print(f"Accuracy: {evaluation_results['accuracy']:.2f}")

    return results