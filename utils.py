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


def _marker_parser(pdf_file, force_reparse=False):
    """
    Parse a PDF file using the marker CLI tool and convert to markdown.
    Checks if the PDF file is from financeBench or has been parsed already.

    Args:
        pdf_file (str): Path to the PDF file to be processed
        force_reparse (bool): Whether to reparse the PDF even if markdown already exists

    Returns:
        str: Path to the generated markdown file
    """
    # Get PDF filename without extension
    pdf_path = Path(pdf_file)
    pdf_name = pdf_path.stem

    # Check if pdf is already parsed from financeBench
    markdown_path = ".." / Path("marker_financebench") / pdf_name / f"{pdf_name}.md"
    if not force_reparse and markdown_path.exists():
        print(f"Found existing financeBench markdown for {pdf_name}: {markdown_path}")
        return str(markdown_path)

    # Create output directory if it doesn't exist
    markdown_path = Path("marker") / pdf_name
    markdown_path.mkdir(parents=True, exist_ok=True)
    if not force_reparse and markdown_path.exists():
        print(f"Found existing marker markdown for {pdf_name}: {markdown_path}")
        return str(markdown_path)

    # Run marker CLI command
    try:
        print(f"Parsing {pdf_file} with marker...")
        cmd = ["marker_single", pdf_file, "--output_dir", str(output_dir), "--output_format", "markdown", "--format_lines"]
        subprocess.run(cmd, check=True)

        # Check if markdown was generated
        if markdown_path.exists():
            print(f"Successfully parsed {pdf_file}. Markdown saved to {markdown_path}")
            return str(markdown_path)
        else:
            print(f"Marker didn't generate markdown for {pdf_file}")
            return None
    except Exception as e:
        print(f"Error parsing {pdf_file} with marker: {e}")
        return None


def _process_documents(documents, chunk_size, chunk_overlap, use_tiktoken=False):
    """Helper function to count tokens and split documents."""
    # Calculate token count from document content
    content = str(documents) if len(documents) > 1 else documents[0].page_content
    token_count = num_tokens_from_string(content, "cl100k_base")
    print(f'PDF Token Count: {token_count}')

    # Choose appropriate text splitter
    if use_tiktoken:
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    else:
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    split_documents = text_splitter.split_documents(documents)
    return split_documents, token_count


def load_pdf_chunk(pdf_file, chunk_size, chunk_overlap, method):
    """
    The function loads a pdf file and makes it ready for querying

    Args:
        pdf_file (str): name of the pdf file to be processed
        chunk_size (int): size of each chunk
        chunk_overlap (int): overlap between chunks
        method (str): method to use for PDF parsing

    Returns:
        pages (list): list of all the pages in the pdf
    """
    documents = None

    # Handle marker method separately due to its unique workflow
    if method == "marker":

        # Try parsing with marker
        markdown_path = _marker_parser(pdf_file)
        if markdown_path:
            with open(markdown_path, 'r', encoding='utf-8') as f:
                content = f.read()
            documents = [Document(page_content=content, metadata={"source": pdf_file})]
            return _process_documents(documents, chunk_size, chunk_overlap, use_tiktoken=True)

        # Fallback to PDFMinerLoader
        print("Marker parsing failed, falling back to PDFMinerLoader")
        method = "default"

    # Handle all other loader methods
    if method == "Load page-wise PDF":
        loader = PyPDFLoader(pdf_file)
    elif method == "pymu":
        loader = PyMuPDFLoader(pdf_file)
    elif method == "unstructured":
        loader = UnstructuredPDFLoader(pdf_file, mode="elements", strategy="hi_res")
    else:  # default case (including fallback from marker)
        loader = PDFMinerLoader(pdf_file)

    documents = loader.load()
    return _process_documents(documents, chunk_size, chunk_overlap, use_tiktoken=True)