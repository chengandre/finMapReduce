from mapreduce_qa import process_mapreduce_qa
from utils import load_openai_model


files = ['Aegis/annual_report_2024.pdf']
questions = {'What is the net income of the company?': 'Summarize the revenue, expenses, and other financial figures that contribute to the net income of the company.'}
chunk_size = 2**17
token_overlap = 500

model_name = "gpt-4o-mini"
temperature = 0.01
max_tokens = 2000
llm = load_openai_model(model_name, temperature, max_tokens)
company_response = process_mapreduce_qa(files, questions, model_name, llm, chunk_size, token_overlap)