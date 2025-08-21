import os
import tempfile
import time
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from api.models import AnswerResponse, ErrorResponse, HealthResponse
from config import settings, validate_file_upload, get_model_config, get_api_key
from single_doc_pipeline import SingleDocPipeline
from utils import create_rate_limited_llm, RateLimitConfig
import yaml


# Router for API endpoints
router = APIRouter()

# Global pipeline cache
_pipeline_cache: Dict[str, SingleDocPipeline] = {}
_prompts_dict: Optional[Dict[str, Any]] = None


def load_prompts() -> Dict[str, Any]:
    """Load prompts from configuration file."""
    global _prompts_dict

    if _prompts_dict is None:
        try:
            # Load prompt configuration
            config_path = os.path.join(os.path.dirname(__file__), '../../../prompts/prompt_config.yml')
            with open(config_path, 'r') as f:
                prompt_config = yaml.safe_load(f)

            # Load default prompt set (use hybrid for webapp)
            # default_set = prompt_config.get('default_set', 'hybrid')
            prompt_files = prompt_config['prompt_sets']['hybrid']

            _prompts_dict = {}
            for prompt_key, filename in prompt_files.items():
                # filename already includes 'prompts/' prefix
                prompt_path = os.path.join(os.path.dirname(__file__), '../../..', filename)
                with open(prompt_path, 'r') as f:
                    _prompts_dict[prompt_key] = f.read().strip()

        except Exception as e:
            # Fallback to basic prompts if configuration fails
            _prompts_dict = {
                'map_prompt': 'Based on the context: {context}\n\nAnswer the question: {question_int}',
                'reduce_prompt': 'Based on the map results: {context}\n\nProvide a final answer to: {question}',
                'judge_prompt': 'Evaluate the answer quality.'
            }
            print(f"Warning: Could not load prompts configuration, using fallback: {e}")

    return _prompts_dict


def get_or_create_pipeline(config: Dict[str, Any]) -> SingleDocPipeline:
    """Get or create a cached pipeline instance."""
    # Create cache key from config
    cache_key = f"{config['model_name']}_{config['provider']}_{config['format_type']}_{config['pdf_parser']}"

    if cache_key not in _pipeline_cache:
        try:
            # Get API key
            api_key = get_api_key(config["provider"], config.get("key_type", "default"))
            if not api_key:
                raise ValueError(f"No API key available for provider: {config['provider']}")

            # Set temporary environment variable for LLM client
            key = "self"
            # os.environ[temp_env_var] = api_key

            try:
                rate_config = RateLimitConfig(
                    requests_per_minute=30000,
                    tokens_per_minute=150000000,
                    request_burst_size=3000
                )

                # Create LLM instance
                llm = create_rate_limited_llm(
                    model_name=config["model_name"],
                    provider=config["provider"],
                    temperature=config["temperature"],
                    rate_limit_config=rate_config,
                    api_key_env=key
                )

                # Create LLM instances for hybrid approach
                map_llm = create_rate_limited_llm(
                    model_name=config["model_name"],
                    temperature=config["temperature"],
                    api_key_env=key,
                    rate_limit_config=rate_config,
                    parse_json=False
                )

                reduce_llm = create_rate_limited_llm(
                    model_name=config["model_name"],
                    temperature=config["temperature"],
                    api_key_env=key,
                    rate_limit_config=rate_config,
                    parse_json=True
                )

                judge_rate_config = rate_config
                judge = create_rate_limited_llm(
                    model_name="gpt-5-nano",
                    temperature=1.0,
                    max_tokens=8192,
                    provider="openai",
                    api_key_env=key,
                    rate_limit_config=judge_rate_config,
                    parse_json=True
                )
            finally:
                # Clean up temporary environment variable
                # if temp_env_var in os.environ:
                #     del os.environ[temp_env_var]
                pass

            # Load prompts
            prompts_dict = load_prompts()

            # Create pipeline
            pipeline = SingleDocPipeline(
                llm=llm,
                map_llm=map_llm,
                reduce_llm=reduce_llm,
                judge_llm=judge,
                prompts_dict=prompts_dict,
                format_type=config["format_type"],
                pdf_parser=config["pdf_parser"],
                chunk_size=config["chunk_size"],
                chunk_overlap=config["chunk_overlap"],
                max_concurrent_chunks=config["max_concurrent_chunks"],
                max_file_size=settings.max_file_size
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

        # Get or create pipeline
        pipeline = get_or_create_pipeline(config)

        # Process the document
        result = pipeline.process_uploaded_file(temp_file_path, question, cleanup=False)

        # Ensure request_id is set
        result["request_id"] = request_id

        return AnswerResponse(**result)

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
                "gpt-4o-mini"
            ],
            "openrouter": [
                "deepseek/deepseek-r1-0528:free"
            ]
        },
        "format_types": ["json", "hybrid", "plain"],
        "pdf_parsers": ["marker", "pypdf", "pdfminer"]
    }