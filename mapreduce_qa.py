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
    # Create the LLMChain
    llm_chain_map = LLMChain(prompt=map_prompt, llm=llm, memory=None)

    # Track map phase token usage
    map_input_tokens = 0
    map_output_tokens = 0

    def process_chunk(chunk):
        nonlocal map_input_tokens
        input_data = {"context": chunk, "final_query": final_query}
        # Count input tokens for this chunk
        chunk_input_tokens = num_tokens_from_string(f"{chunk} {final_query}", "cl100k_base")
        map_input_tokens += chunk_input_tokens
        return llm_chain_map.run(input_data)

    print("Map phase started, calling LLMs")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_chunk, chunked_docs))
    print("Map phase completed")
    print("Results:")
    print(results)

    # Count output tokens from map phase
    for result in results:
        map_output_tokens += num_tokens_from_string(result, "cl100k_base")

    # modified_results = preprocess_results(results=results)

    # Load reduce prompt
    reduce_prompt = load_prompt("prompts/reduce_prompt.yml")

    # Track reduce phase token usage
    results_text = "\n".join(results)
    reduce_input_tokens = num_tokens_from_string(f"{results_text} {final_query}", "cl100k_base")

    input_data_reduce = {"map_results": results, "final_query": final_query}
    llm_chain_reduce = LLMChain(prompt=reduce_prompt, llm=llm, memory=None)
    result_final = llm_chain_reduce.run(input_data_reduce)

    # Count output tokens from reduce phase
    reduce_output_tokens = num_tokens_from_string(result_final, "cl100k_base")

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
                         chunk_size, token_overlap, method):
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