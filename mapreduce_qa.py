from langchain.chains import LLMChain
from langchain.prompts import load_prompt
import concurrent.futures
import re
import time
from utils import load_pdf_chunk, num_tokens_from_string
from datetime import datetime
from tqdm import tqdm


def preprocess_results(results):
    """
    filters the score of the results in the intermediate stage
    """
    modified_results = []
    for result in results:
        try:
            # modified_results.append(result)
            score = 0
            # print(result)
            if "Score:" in result:
                score = int(re.search(r'Score:\s(\d+)', result).group(1))
                # print("Answer ========= > ", score, result.split("Score")[0])
                if score > 50:
                    modified_results.append(result)
        except Exception as e:
            print(e)

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

    # Extract content from results for reduce phase - handle JSON format
    processed_results = []
    for result in results:
        # Get JSON data from the wrapper result
        result_json = result.get('json', {})
        if result_json:
            # Format the JSON data for the reduce phase
            formatted_result = f"Summary: {result_json.get('summary', '')}\n"
            formatted_result += f"Terms: {', '.join(result_json.get('terms', []))}\n"
            formatted_result += f"Evidence: {'; '.join(result_json.get('evidence', []))}\n"
            formatted_result += f"Answer: {result_json.get('answer', '')}\n"
            formatted_result += f"Relevance Score: {result_json.get('relevance_score', 0)}\n"
            processed_results.append(formatted_result)
        else:
            # Fallback to raw response content
            raw_response = result.get('raw_response')
            if raw_response:
                content = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
                processed_results.append(content)

    results_text = "\n---\n".join(processed_results)
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
            "question": item["question"],
            "question_type": item["question_type"],
            "question_reasoning": item["question_reasoning"],
            "answer": item["answer"],
            "company": item["company"],
            "doc_name": item["doc_name"],
            "evidence": [ev["evidence_text"] for ev in item["evidence"]]
        }

        qa_data.append(qa_pair)
        count += 1

    return qa_data


def process_financebench_qa(jsonl_path, model_name, llm, num_samples=None, chunk_size=36000, chunk_overlap=1000):
    """
    Process QA from financebench.

    Args:
        jsonl_path (str): Path to financebench jsonl file
        model_name (str): Name of the LLM model
        llm: LLM instance to use
        num_samples (int): Number of samples to process
        chunk_size (int): Size of each document chunk
        chunk_overlap (int): Overlap between document chunks

    Returns:
        dict: Results containing model answers, golden answers, and evaluation results
    """
    print(f"Loading {num_samples if num_samples else 'all'} samples from financebench data...")
    qa_data = load_financebench_data(jsonl_path, num_samples)

    t1 = time.time()
    print("Processing QA pairs with MapReduce...")
    for i, qa_pair in enumerate(tqdm(qa_data, desc="Processing QA pairs")):
        # print(f"Processing QA pair {i+1}/{len(qa_data)}: {qa_pair['doc_name']}")

        # Get document name from the qa_pair
        doc_name = qa_pair["doc_name"]
        question = qa_pair["question"]

        docs, token_count = load_pdf_chunk(doc_name, chunk_size, chunk_overlap, method="marker")

        # Load prompts from YAML
        map_prompt = load_prompt("map_prompt.yml")

        # Track map phase token usage
        map_input_tokens = 0
        map_output_tokens = 0

        def process_chunk(chunk):
            return llm(map_prompt, context=chunk.page_content, final_query=question)

        # print("Map phase started, calling LLMs")
        with concurrent.futures.ThreadPoolExecutor(10) as executor:
            results = list(executor.map(process_chunk, docs))
        # print("Map phase completed")
        # print("Results:")
        # print(results)

        # Count output tokens from map phase
        for result in results:
            raw_response = result.get('raw_response')
            if raw_response and hasattr(raw_response, 'usage_metadata') and raw_response.usage_metadata:
                map_input_tokens += raw_response.usage_metadata["input_tokens"]
                map_output_tokens += raw_response.usage_metadata["output_tokens"]

        # Load reduce prompt
        reduce_prompt = load_prompt("reduce_prompt.yml")

        # Process map results for reduce phase - handle JSON format
        processed_results = []
        for result in results:
            # Get JSON data from the wrapper result
            result_json = result.get('json', {})
            if result_json:
                # Format the JSON data for the reduce phase
                formatted_result = f"Summary: {result_json.get('summary', '')}\n"
                formatted_result += f"Terms: {', '.join(result_json.get('terms', []))}\n"
                formatted_result += f"Evidence: {'; '.join(result_json.get('evidence', []))}\n"
                formatted_result += f"Answer: {result_json.get('answer', '')}\n"
                formatted_result += f"Relevance Score: {result_json.get('relevance_score', 0)}\n"
                processed_results.append(formatted_result)
            else:
                # Fallback to raw response content
                raw_response = result.get('raw_response')
                if raw_response:
                    content = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
                    processed_results.append(content)

        results_text = "\n---\n".join(processed_results)
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

    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "financebench_results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"results_{timestamp}.json")

    results = {
        "model_name": model_name,
        "execution_time": datetime.now().isoformat(),
        "time_taken": time.time() - t1,
        "num_samples": len(qa_data),
        "qa_data": qa_data,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")

    return results