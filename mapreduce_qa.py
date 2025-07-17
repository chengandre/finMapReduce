from langchain.chains import LLMChain
from langchain.prompts import load_prompt
import concurrent.futures
import re
import time
from utils import load_pdf_chunk, num_tokens_from_string
from datetime import datetime


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