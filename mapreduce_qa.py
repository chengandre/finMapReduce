from model_files.templates import question_prompt_template, combine_prompt_template
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import concurrent.futures
import re
import time
from utils import load_pdf_chunk
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


def mapreduce_qa_documents(llm, chunked_docs, query_reduce, query_map):
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

    """
    # Split the documents into smaller chunks

    t1 = time.time()

    map_prompt = PromptTemplate(template=question_prompt_template)
    total_docs = len(chunked_docs)
    # Create the LLMChain
    llm_chain_map = LLMChain(prompt=map_prompt, llm=llm, memory=None)

    def process_chunk(chunk):
        input_data = {"context": [chunk], "question_int": query_reduce}
        return llm_chain_map.run(input_data)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_chunk, chunked_docs))

    modified_results = preprocess_results(results=results)
    reduce_prompt = PromptTemplate(template=combine_prompt_template)
    input_data_reduce = {"summaries": modified_results, "question_final": query_map}
    llm_chain_reduce = LLMChain(prompt=reduce_prompt, llm=llm, memory=None)
    result_final = llm_chain_reduce.run(input_data_reduce)
    time_to_process = time.time() - t1
    # print(modified_results)
    return result_final, results, total_docs, time_to_process


def process_mapreduce_qa(files, selected_questions_dict, model_name, llm,
                         chunk_size, token_overlap):
    documents, token_count = [], 0
    for i in range(len(files)):
        temp_documents, temp_token_count = load_pdf_chunk(files[i], chunk_size, token_overlap)
        documents += temp_documents
        token_count += temp_token_count

    final_answer_responses = []
    for query_map, query_reduce in selected_questions_dict.items():
        response_dict = {"chain": "MapReduce", "execution_time": datetime.now(), "model_name": model_name,
                         "query": query_map}

        response_dict["answer"], response_dict["int_result"], response_dict["total_chunks"], \
            response_dict["time_taken"] = mapreduce_qa_documents(llm, documents, query_reduce, query_map)
        final_answer_responses.append(response_dict)
    return final_answer_responses