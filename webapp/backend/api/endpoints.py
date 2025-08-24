import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from api.models import AnswerResponse, ErrorResponse, HealthResponse
from config import settings, validate_file_upload, get_model_config, get_api_key
from src.loaders.webapp_loader import WebappDatasetLoader
from src.core.factory import PipelineFactory
from src.llm.async_llm_client import create_async_rate_limited_llm, RateLimitConfig
import yaml
import asyncio


# Router for API endpoints
router = APIRouter()

# Global pipeline cache and prompts
_pipeline_cache: Dict[str, Any] = {}
_prompts_dict: Optional[Dict[str, Any]] = None


def load_prompts(prompt_set_name: str = "hybrid") -> Dict[str, Any]:
    """Load prompts from configuration file."""
    try:
        # Add the parent directory to path so we can import src.utils.document_processing as utils from the root
        root_dir = os.path.join(os.path.dirname(__file__), '../../..')
        if root_dir not in sys.path:
            sys.path.append(root_dir)

        # Change to root directory temporarily for prompt loading
        original_cwd = os.getcwd()
        os.chdir(root_dir)

        try:
            from src.utils.document_processing import load_prompt_set
            return load_prompt_set(prompt_set_name)
        finally:
            os.chdir(original_cwd)
    except Exception as e:
        # Fallback to basic prompts if configuration fails
        fallback_prompts = {
            'map_prompt': 'Based on the context: {context}\n\nAnswer the question: {question}',
            'reduce_prompt': 'Based on the map results: {context}\n\nProvide a final answer to: {question}',
            'judge_prompt': 'Evaluate the answer quality.'
        }
        print(f"Warning: Could not load prompts configuration, using fallback: {e}")
        return fallback_prompts


def get_or_create_pipeline(config: Dict[str, Any]) -> Any:
    """Get or create a cached pipeline instance."""
    # Create cache key from config
    pipeline_type = config.get('pipeline_type', 'mapreduce')
    format_or_strategy = config.get('format_type', 'hybrid') if pipeline_type == 'mapreduce' else config.get('strategy', 'start')
    cache_key = f"{config['model_name']}_{config['provider']}_{pipeline_type}_{format_or_strategy}_{config['pdf_parser']}"

    if cache_key not in _pipeline_cache:
        try:
            # Get API key
            api_key = get_api_key(config["provider"], config.get("key_type", "default"))
            if not api_key:
                raise ValueError(f"No API key available for provider: {config['provider']}. Please set OPENAI_API_KEY or OPENROUTER_API_KEY environment variable.")

            # Set environment variable for LLM client
            key_env_var = "SELF_OPENAI_API_KEY" if config.get("key_type") == "self" else "OPENAI_API_KEY"
            if config["provider"] == "openrouter":
                key_env_var = "OPENROUTER_API_KEY"

            # Temporarily set the API key in environment
            original_value = os.environ.get(key_env_var)
            os.environ[key_env_var] = api_key

            try:
                rate_config = RateLimitConfig(
                    requests_per_minute=config.get('requests_per_minute', 30000),
                    tokens_per_minute=config.get('tokens_per_minute', 150000000),
                    request_burst_size=config.get('request_burst_size', 3000)
                )

                # Create LLM instance
                llm = create_async_rate_limited_llm(
                    model_name=config["model_name"],
                    provider=config["provider"],
                    temperature=config["temperature"],
                    rate_limit_config=rate_config,
                    api_key_env=None  # Will use environment variable
                )

                # Create LLM instances for hybrid approach
                map_llm = create_async_rate_limited_llm(
                    model_name=config["model_name"],
                    temperature=config["temperature"],
                    provider=config["provider"],
                    api_key_env=None,
                    rate_limit_config=rate_config,
                    parse_json=False
                )

                reduce_llm = create_async_rate_limited_llm(
                    model_name=config["model_name"],
                    temperature=config["temperature"],
                    provider=config["provider"],
                    api_key_env=None,
                    rate_limit_config=rate_config,
                    parse_json=True
                )

                judge_rate_config = rate_config
                judge = create_async_rate_limited_llm(
                    model_name="gpt-4o-mini",  # Use a model that actually exists
                    temperature=1.0,
                    max_tokens=8192,
                    provider="openai",
                    api_key_env=None,
                    rate_limit_config=judge_rate_config,
                    parse_json=True
                )
            finally:
                # Restore original environment variable
                if original_value is not None:
                    os.environ[key_env_var] = original_value
                elif key_env_var in os.environ:
                    del os.environ[key_env_var]

            # Load prompts with specified prompt set
            prompts_dict = load_prompts(config.get('prompt_set', 'hybrid'))

            # Create webapp dataset loader
            webapp_loader = WebappDatasetLoader(
                pdf_parser=config["pdf_parser"],
                max_file_size=settings.max_file_size
            )

            # Create pipeline using factory
            pipeline_type = config.get('pipeline_type', 'mapreduce')

            if pipeline_type == 'mapreduce':
                pipeline = PipelineFactory.create_pipeline(
                    dataset='webapp',
                    format_type=config["format_type"],
                    llm=llm,
                    map_llm=map_llm,
                    reduce_llm=reduce_llm,
                    judge_llm=judge,
                    prompts_dict=prompts_dict,
                    dataset_loader=webapp_loader,
                    chunk_size=config["chunk_size"],
                    chunk_overlap=config["chunk_overlap"],
                    max_concurrent_chunks=config["max_concurrent_chunks"],
                    score_threshold=config.get('score_threshold', 5),
                    max_total_requests=config.get('max_total_requests', 1000)
                )
            else:  # truncation
                pipeline = PipelineFactory.create_truncation_pipeline(
                    dataset='webapp',
                    strategy=config.get('strategy', 'start'),
                    context_window=config.get('context_window', 128000),
                    buffer=config.get('buffer', 2000),
                    max_document_tokens=config.get('max_document_tokens'),
                    llm=llm,
                    judge_llm=judge,
                    prompts_dict=prompts_dict,
                    dataset_loader=webapp_loader,
                    max_total_requests=config.get('max_total_requests', 1000)
                )

            _pipeline_cache[cache_key] = pipeline

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create pipeline: {str(e)}")

    return _pipeline_cache[cache_key]


@router.post("/answer", response_model=AnswerResponse)
async def answer_question(
    file: UploadFile = File(...),
    question: str = Form(...),
    model_name: Optional[str] = Form(settings.default_model),
    provider: Optional[str] = Form(settings.default_provider),
    temperature: Optional[float] = Form(settings.default_temperature),
    chunk_size: Optional[int] = Form(settings.default_chunk_size),
    chunk_overlap: Optional[int] = Form(settings.default_chunk_overlap),
    format_type: Optional[str] = Form(settings.default_format_type),
    pdf_parser: Optional[str] = Form(settings.default_pdf_parser),
    max_concurrent_chunks: Optional[int] = Form(settings.default_max_concurrent_chunks),
    pipeline_type: Optional[str] = Form("mapreduce"),  # mapreduce or truncation
    strategy: Optional[str] = Form("start"),  # for truncation: start, end, smart
    context_window: Optional[int] = Form(128000),  # for truncation
    buffer: Optional[int] = Form(2000),  # for truncation
    max_document_tokens: Optional[int] = Form(None),  # for truncation
    score_threshold: Optional[int] = Form(5),  # for filtering
    prompt_set: Optional[str] = Form("hybrid"),  # prompt set to use
    requests_per_minute: Optional[int] = Form(30000),  # rate limiting
    tokens_per_minute: Optional[int] = Form(150000000),  # rate limiting
    request_burst_size: Optional[int] = Form(3000),  # rate limiting
    max_total_requests: Optional[int] = Form(1000),  # total concurrent requests
    key_type: Optional[str] = Form("default")
):
    """
    Process an uploaded document and answer a question about it.

    Accepts multipart form data with a file and question parameters.
    Returns structured answer with reasoning, evidence, and metadata.
    """
    temp_file_path = None
    request_id = f"req_{int(time.time())}_{hash(question) % 10000}"

    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Read file content to check size
        file_content = await file.read()
        file_size = len(file_content)

        # Validate file upload
        validate_file_upload(file.filename, file_size)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            tmp_file.write(file_content)
            temp_file_path = tmp_file.name

        # Get model configuration
        config = get_model_config(
            model_name=model_name,
            provider=provider,
            temperature=temperature,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            format_type=format_type,
            pdf_parser=pdf_parser,
            max_concurrent_chunks=max_concurrent_chunks,
            key_type=key_type
        )

        # Add webapp-specific parameters
        config['pipeline_type'] = pipeline_type
        config['strategy'] = strategy
        config['context_window'] = context_window
        config['buffer'] = buffer
        config['max_document_tokens'] = max_document_tokens
        config['score_threshold'] = score_threshold
        config['prompt_set'] = prompt_set
        config['requests_per_minute'] = requests_per_minute
        config['tokens_per_minute'] = tokens_per_minute
        config['request_burst_size'] = request_burst_size
        config['max_total_requests'] = max_total_requests

        # Get or create pipeline
        pipeline = get_or_create_pipeline(config)

        # Create QA pair for processing
        qa_pair = {
            "file_path": temp_file_path,
            "question": question,
            "doc_name": os.path.basename(file.filename),
            "answer": "",   # Not used for processing
            "evidence": []  # Not used for processing
        }

        # Process the document using async method
        result = await pipeline.process_single_qa_async(qa_pair)

        # Extract token statistics and create unified stats for frontend
        token_stats = result.get("token_stats", {})

        # Map fields for webapp response
        webapp_result = {
            "answer": result.get("llm_answer", ""),
            "reasoning": result.get("llm_reasoning", ""),
            "evidence": result.get("llm_evidence", []),
            "token_stats": token_stats,
            "timing_stats": token_stats.get("timing", {}),
            "chunk_stats": {
                "len_docs": token_stats.get("len_docs", 0),
                "filtering_stats": token_stats.get("filtering_stats", {}),
                "total_chunks": token_stats.get("filtering_stats", {}).get("chunks_before_filtering", 0),
                "chunks_after_filtering": token_stats.get("filtering_stats", {}).get("chunks_after_filtering", 0)
            },
            "request_id": request_id
        }

        return AnswerResponse(**webapp_result)

    # except HTTPException:
    #     raise
    # except ValueError as e:
    #     raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
      print(f"Full exception details: {e}")
      import traceback
      print(traceback.format_exc())
      raise e

    finally:
        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Warning: Could not clean up temp file {temp_file_path}: {e}")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version=settings.api_version
    )


@router.get("/models")
async def list_available_models():
    """List available models and providers."""
    return {
        "models": {
            "openai": [
                "gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"
            ],
            "openrouter": [
                "deepseek/deepseek-r1-0528:free",
                "anthropic/claude-3-haiku",
                "meta-llama/llama-3.1-8b-instruct:free"
            ]
        },
        "pipeline_types": ["mapreduce", "truncation"],
        "format_types": ["json", "hybrid", "plain_text"],  # for mapreduce
        "truncation_strategies": ["start", "end", "smart"],  # for truncation
        "pdf_parsers": ["marker", "pypdf", "pdfminer", "unstructured"],
        "prompt_sets": ["default", "hybrid", "baseline", "test"],
        "providers": ["openai", "openrouter"],
        "temperature_range": {"min": 0.0, "max": 2.0, "step": 0.1},
        "chunk_size_range": {"min": 1000, "max": 100000, "step": 1000},
        "score_threshold_range": {"min": 1, "max": 100, "step": 1}
    }


@router.post("/preview")
async def preview_document(
    file: UploadFile = File(...),
    pdf_parser: Optional[str] = Form("marker")
):
    """
    Preview document content without processing through QA pipeline.
    Returns first 2000 characters of the document for preview.
    """
    temp_file_path = None

    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Read file content to check size
        file_content = await file.read()
        file_size = len(file_content)

        # Validate file upload
        validate_file_upload(file.filename, file_size)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            tmp_file.write(file_content)
            temp_file_path = tmp_file.name

        # Create webapp loader for document processing
        webapp_loader = WebappDatasetLoader(pdf_parser=pdf_parser)

        # Load document content
        file_ext = Path(file.filename).suffix.lower()

        if file_ext == '.pdf':
            # Get full document text for preview
            full_text, token_count = webapp_loader.load_full_document({
                "file_path": temp_file_path
            })
        elif file_ext in ['.txt', '.md']:
            # Read text file directly
            with open(temp_file_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
            token_count = len(full_text) // 4  # Rough estimate
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # Limit preview to first 2000 characters for display
        preview_text = full_text[:2000]
        if len(full_text) > 2000:
            preview_text += "\n\n... (content truncated for preview)"

        return {
            "filename": file.filename,
            "file_type": file_ext,
            "file_size": file_size,
            "estimated_tokens": token_count,
            "preview_text": preview_text,
            "full_length": len(full_text),
            "is_truncated": len(full_text) > 2000
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")
    finally:
        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass