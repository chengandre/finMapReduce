from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, PDFMinerLoader
from langchain.text_splitter import CharacterTextSplitter
import tiktoken
import os


os.environ["OPENAI_API_KEY"] = ''

def load_openai_model(model_name, temperature, max_tokens):
    """
    load openai models

    Args:
        model_name (str): the model to be used from openai
    Returns:
        llm: the langauge model on which queries/summarsation would be done on
    """
    llm = ChatOpenAI(
        model_name=model_name,
        openai_api_key=os.environ["OPENAI_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return llm


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def load_pdf_chunk(pdf_file, chunk_size, chunk_overlap, method):
    """
    The function loads a pdf file and makes it ready for queryting

    Args:
        pdf_file (str): name of the pdf file to be processed

    Returns:
        pages (list): list of all the pages in the pdf
    """
    # Load the pdf file
    if method == "Load page-wise PDF":
        loader = PyPDFLoader(pdf_file)
    else:
        loader = PDFMinerLoader(pdf_file)
    documents = loader.load()

    token_count = num_tokens_from_string(str(documents), "cl100k_base")
    print(f'PDF Token Count: {token_count}')
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(documents)
    return documents, token_count