# Web Application Documentation

This document covers the complete web application interface, API endpoints, and integration details for FinMapReduce.

## Overview

The FinMapReduce web application provides a user-friendly interface for uploading documents and getting AI-powered answers to financial questions. Built with FastAPI backend and modern HTML/CSS/JavaScript frontend.

### Key Features

- **Drag-and-drop file upload**: Support for PDF, TXT, and MD files up to 50MB
- **Real-time document preview**: Preview document content before processing
- **Advanced configuration**: Full control over processing parameters
- **Pipeline selection**: Choose between MapReduce and Truncation approaches
- **Results visualization**: Structured display of answers with reasoning and evidence
- **Processing statistics**: Detailed metrics and timing information

## Architecture

### Three-Tier Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Frontend     │    │   API Layer     │    │ Processing Layer│
│  (Static HTML)  │◄──►│   (FastAPI)     │◄──►│   (Pipelines)   │
│                 │    │                 │    │                 │
│ • File Upload   │    │ • Endpoints     │    │ • MapReduce     │
│ • Preview       │    │ • Validation    │    │ • Truncation    │
│ • Configuration │    │ • Processing    │    │ • LLM Client    │
│ • Results UI    │    │ • Error Handling│    │ • Evaluation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Component Structure

```
webapp/
├── backend/                   # FastAPI backend
│   ├── main.py               # Application server
│   ├── config.py             # Configuration management
│   ├── requirements.txt      # Backend dependencies
│   └── api/
│       ├── __init__.py
│       ├── endpoints.py      # API route handlers
│       └── models.py         # Pydantic request/response models
└── frontend/                 # Static web interface
    ├── index.html            # Main web interface
    └── static/
        ├── app.js            # Frontend JavaScript logic
        └── style.css         # Styling and responsive design
```

## Backend API

### FastAPI Application (`main.py`)

The main application server provides:

```python
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os

from api.endpoints import router
from config import Config

# Create FastAPI application
app = FastAPI(
    title=Config.API_TITLE,
    version=Config.API_VERSION,
    description="AI-powered question answering for financial documents"
)

# Configure CORS for local development and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")

# Serve static files
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

@app.get("/")
async def root():
    """Serve the main web interface."""
    return FileResponse("frontend/index.html")

@app.get("/favicon.ico")
async def favicon():
    """Serve favicon."""
    favicon_path = "frontend/static/favicon.ico"
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    return {"message": "Favicon not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=Config.HOST,
        port=Config.PORT,
        log_level="info" if not Config.DEBUG else "debug"
    )
```

### Configuration (`config.py`)

Centralized configuration management:

```python
import os
from typing import List

class Config:
    """Application configuration from environment variables."""

    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # API metadata
    API_TITLE: str = os.getenv("API_TITLE", "MapReduce QA WebApp")
    API_VERSION: str = os.getenv("API_VERSION", "1.0.0")

    # File upload settings
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", 52428800))  # 50MB
    TEMP_DIR: str = os.getenv("TEMP_DIR", "/tmp/webapp_uploads")
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".txt", ".md"]

    # Default processing settings
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
    DEFAULT_PROVIDER: str = os.getenv("DEFAULT_PROVIDER", "openai")
    DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", 0.0))
    DEFAULT_CHUNK_SIZE: int = int(os.getenv("DEFAULT_CHUNK_SIZE", 32768))
    DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("DEFAULT_CHUNK_OVERLAP", 4096))
    DEFAULT_FORMAT_TYPE: str = os.getenv("DEFAULT_FORMAT_TYPE", "hybrid")
    DEFAULT_PDF_PARSER: str = os.getenv("DEFAULT_PDF_PARSER", "marker")
    DEFAULT_MAX_CONCURRENT_CHUNKS: int = int(os.getenv("DEFAULT_MAX_CONCURRENT_CHUNKS", 50))

    # LLM API settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    SELF_OPENAI_API_KEY: str = os.getenv("SELF_OPENAI_API_KEY", "")

    @classmethod
    def validate(cls) -> bool:
        """Validate configuration."""
        if not any([cls.OPENAI_API_KEY, cls.OPENROUTER_API_KEY, cls.SELF_OPENAI_API_KEY]):
            raise ValueError("At least one API key must be configured")

        if not os.path.exists(cls.TEMP_DIR):
            os.makedirs(cls.TEMP_DIR, exist_ok=True)

        return True
```

### API Endpoints (`api/endpoints.py`)

Complete API endpoint implementation:

```python
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
import asyncio
import os
import tempfile
import uuid
from typing import Optional, Dict, Any

from .models import (
    AnswerResponse, PreviewResponse, ModelsResponse, HealthResponse,
    ProcessingRequest, ErrorResponse
)
from ..config import Config
from src.core.factory import PipelineFactory
from src.loaders.webapp_loader import WebappDatasetLoader
from src.llm.async_llm_client import create_async_rate_limited_llm, RateLimitConfig
from src.llm.utils import load_prompt_set

router = APIRouter()

# Global cache for pipelines
pipeline_cache: Dict[str, Any] = {}

@router.post("/answer", response_model=AnswerResponse)
async def process_document(
    file: UploadFile = File(...),
    question: str = Form(...),
    model_name: str = Form(Config.DEFAULT_MODEL),
    provider: str = Form(Config.DEFAULT_PROVIDER),
    temperature: float = Form(Config.DEFAULT_TEMPERATURE),
    pipeline_type: str = Form("mapreduce"),
    format_type: str = Form(Config.DEFAULT_FORMAT_TYPE),
    chunk_size: int = Form(Config.DEFAULT_CHUNK_SIZE),
    chunk_overlap: int = Form(Config.DEFAULT_CHUNK_OVERLAP),
    score_threshold: int = Form(5),
    max_total_requests: int = Form(20),
    pdf_parser: str = Form(Config.DEFAULT_PDF_PARSER),
    truncation_strategy: str = Form("start"),
    context_window: int = Form(128000),
    buffer: int = Form(2000)
) -> AnswerResponse:
    """
    Main document processing endpoint.

    Accepts multipart form data with file upload and processing parameters.
    Returns structured answer with reasoning, evidence, and processing statistics.
    """

    request_id = str(uuid.uuid4())
    temp_path = None

    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in Config.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {Config.ALLOWED_EXTENSIONS}"
            )

        # Check file size
        contents = await file.read()
        if len(contents) > Config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {Config.MAX_FILE_SIZE} bytes"
            )

        # Save to temporary file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_ext,
            dir=Config.TEMP_DIR
        ) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        # Create pipeline key for caching
        pipeline_key = _create_pipeline_key(
            model_name, provider, temperature, pipeline_type,
            format_type, chunk_size, chunk_overlap, pdf_parser
        )

        # Get or create pipeline
        pipeline = await _get_or_create_pipeline(
            pipeline_key, model_name, provider, temperature,
            pipeline_type, format_type, chunk_size, chunk_overlap,
            score_threshold, max_total_requests, pdf_parser,
            truncation_strategy, context_window, buffer,
            temp_path, question
        )

        # Process document
        results = await pipeline.process_dataset_async(
            data_path="dummy_path",  # Not used with webapp loader
            model_name=model_name,
            num_samples=1,
            score_threshold=score_threshold
        )

        # Extract and format results
        if not results.get('qa_data'):
            raise HTTPException(status_code=500, detail="No results generated")

        qa_result = results['qa_data'][0]

        # Handle evidence format (can be string or list)
        evidence = qa_result.get('llm_evidence', [])
        if isinstance(evidence, str):
            evidence = [evidence] if evidence else []

        response = AnswerResponse(
            answer=qa_result.get('llm_answer', 'No answer generated'),
            reasoning=qa_result.get('llm_reasoning', ''),
            evidence=evidence,
            token_stats=results.get('token_usage_summary', {}),
            timing_stats={
                'total_time': results.get('time_taken', 0),
                'document_loading_time': results.get('document_loading_time', 0),
                'processing_time': qa_result.get('processing_time', 0)
            },
            chunk_stats={
                'total_chunks': results.get('total_chunks', 0),
                'processed_chunks': results.get('processed_chunks', 0),
                'avg_chunk_size': results.get('avg_chunk_size', 0)
            },
            request_id=request_id
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        # Cleanup temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass  # File cleanup failed, but don't fail the request

@router.post("/preview", response_model=PreviewResponse)
async def preview_document(file: UploadFile = File(...)) -> PreviewResponse:
    """
    Document preview endpoint.

    Returns document metadata and content preview without full processing.
    """

    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in Config.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {Config.ALLOWED_EXTENSIONS}"
            )

        # Read file contents
        contents = await file.read()
        file_size = len(contents)

        if file_size > Config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {Config.MAX_FILE_SIZE} bytes"
            )

        # Generate preview
        if file_ext == '.pdf':
            preview_text = await _preview_pdf(contents)
        else:
            # Text-based files
            try:
                preview_text = contents.decode('utf-8')
            except UnicodeDecodeError:
                preview_text = contents.decode('utf-8', errors='ignore')

        # Estimate tokens (rough approximation)
        estimated_tokens = len(preview_text.split()) * 1.3  # Rough token estimation

        # Create content preview (first 2000 characters)
        content_preview = preview_text[:2000]
        if len(preview_text) > 2000:
            content_preview += "... (truncated)"

        return PreviewResponse(
            filename=file.filename,
            file_size=file_size,
            file_type=file_ext[1:],  # Remove dot
            estimated_tokens=int(estimated_tokens),
            content_preview=content_preview,
            total_length=len(preview_text)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview error: {str(e)}")

@router.get("/models", response_model=ModelsResponse)
async def get_available_models() -> ModelsResponse:
    """
    Get available models and configuration options.

    Returns supported models, providers, formats, and default parameters.
    """

    return ModelsResponse(
        models=[
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
            "deepseek/deepseek-r1-0528:free",
            "meta-llama/llama-3.1-8b-instruct:free"
        ],
        providers=["openai", "openrouter"],
        formats=["json", "hybrid", "plain_text"],
        pipeline_types=["mapreduce", "truncation"],
        pdf_parsers=["marker", "pypdf", "pymu", "pdfminer"],
        truncation_strategies=["start", "end"],
        defaults={
            "model_name": Config.DEFAULT_MODEL,
            "provider": Config.DEFAULT_PROVIDER,
            "temperature": Config.DEFAULT_TEMPERATURE,
            "chunk_size": Config.DEFAULT_CHUNK_SIZE,
            "chunk_overlap": Config.DEFAULT_CHUNK_OVERLAP,
            "format_type": Config.DEFAULT_FORMAT_TYPE,
            "pdf_parser": Config.DEFAULT_PDF_PARSER,
            "pipeline_type": "mapreduce",
            "score_threshold": 5,
            "max_total_requests": 20,
            "truncation_strategy": "start",
            "context_window": 128000,
            "buffer": 2000
        }
    )

@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint for monitoring.

    Returns service status and basic statistics.
    """

    try:
        # Check API key availability
        api_keys_available = bool(
            Config.OPENAI_API_KEY or
            Config.OPENROUTER_API_KEY or
            Config.SELF_OPENAI_API_KEY
        )

        # Check temporary directory
        temp_dir_accessible = os.path.exists(Config.TEMP_DIR) and os.access(Config.TEMP_DIR, os.W_OK)

        # Overall status
        status = "healthy" if api_keys_available and temp_dir_accessible else "degraded"

        return HealthResponse(
            status=status,
            timestamp=asyncio.get_event_loop().time(),
            cache_size=len(pipeline_cache),
            config_status={
                "api_keys_configured": api_keys_available,
                "temp_dir_accessible": temp_dir_accessible,
                "max_file_size": Config.MAX_FILE_SIZE,
                "allowed_extensions": Config.ALLOWED_EXTENSIONS
            }
        )

    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            timestamp=asyncio.get_event_loop().time(),
            cache_size=0,
            error=str(e)
        )

# Helper functions

def _create_pipeline_key(*args) -> str:
    """Create cache key from pipeline parameters."""
    return "_".join(str(arg) for arg in args)

async def _get_or_create_pipeline(
    pipeline_key: str, model_name: str, provider: str, temperature: float,
    pipeline_type: str, format_type: str, chunk_size: int, chunk_overlap: int,
    score_threshold: int, max_total_requests: int, pdf_parser: str,
    truncation_strategy: str, context_window: int, buffer: int,
    file_path: str, question: str
):
    """Get existing pipeline or create new one."""

    if pipeline_key not in pipeline_cache:
        # Create LLM client
        llm = create_async_rate_limited_llm(
            model_name=model_name,
            provider=provider,
            temperature=temperature,
            parse_json=(format_type in ['json', 'hybrid']),
            rate_limit_config=RateLimitConfig(
                requests_per_minute=5000,
                tokens_per_minute=4000000,
                request_burst_size=500
            )
        )

        # Load prompts
        prompt_set = "hybrid" if format_type == "hybrid" else "standard"
        prompts = load_prompt_set(prompt_set)

        # Create appropriate pipeline
        if pipeline_type == "mapreduce":
            pipeline = PipelineFactory.create_pipeline(
                dataset='webapp',
                format_type=format_type,
                llm=llm,
                prompts_dict=prompts,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                max_total_requests=max_total_requests,
                pdf_parser=pdf_parser
            )
        else:  # truncation
            pipeline = PipelineFactory.create_truncation_pipeline(
                dataset='webapp',
                strategy=truncation_strategy,
                context_window=context_window,
                buffer=buffer,
                llm=llm,
                prompts_dict=prompts,
                max_total_requests=max_total_requests,
                pdf_parser=pdf_parser
            )

        pipeline_cache[pipeline_key] = pipeline
    else:
        pipeline = pipeline_cache[pipeline_key]

    # Update dataset loader with current file and question
    pipeline.dataset_loader = WebappDatasetLoader(file_path, question)

    return pipeline

async def _preview_pdf(contents: bytes) -> str:
    """Generate preview text from PDF contents."""
    import tempfile
    import os

    # Save to temporary file for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(contents)
        temp_path = temp_file.name

    try:
        # Use document processing utility for preview
        from src.utils.document_processing import load_full_document

        preview_text, _ = load_full_document(temp_path, method="pypdf")  # Fast preview
        return preview_text

    except Exception:
        # Fallback to basic text extraction
        return "PDF preview not available"
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
```

### API Models (`api/models.py`)

Pydantic models for request/response validation:

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import time

class ProcessingRequest(BaseModel):
    """Request model for document processing."""
    question: str = Field(..., description="Question to ask about the document")
    model_name: str = Field("gpt-4o-mini", description="LLM model to use")
    provider: str = Field("openai", description="LLM provider")
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Model temperature")
    pipeline_type: str = Field("mapreduce", description="Pipeline type")
    format_type: str = Field("hybrid", description="Output format type")
    chunk_size: int = Field(32768, gt=0, description="Document chunk size")
    chunk_overlap: int = Field(4096, ge=0, description="Chunk overlap size")
    score_threshold: int = Field(5, ge=0, le=10, description="Relevance score threshold")
    max_total_requests: int = Field(20, gt=0, description="Maximum concurrent requests")
    pdf_parser: str = Field("marker", description="PDF parsing method")
    truncation_strategy: str = Field("start", description="Truncation strategy")
    context_window: int = Field(128000, gt=0, description="Context window size")
    buffer: int = Field(2000, ge=0, description="Response buffer size")

class AnswerResponse(BaseModel):
    """Response model for document processing results."""
    answer: str = Field(..., description="Generated answer")
    reasoning: str = Field("", description="Reasoning behind the answer")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")
    token_stats: Dict[str, Any] = Field(default_factory=dict, description="Token usage statistics")
    timing_stats: Dict[str, float] = Field(default_factory=dict, description="Processing timing statistics")
    chunk_stats: Dict[str, Any] = Field(default_factory=dict, description="Document chunk statistics")
    request_id: str = Field(..., description="Unique request identifier")

class PreviewResponse(BaseModel):
    """Response model for document preview."""
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    file_type: str = Field(..., description="File type/extension")
    estimated_tokens: int = Field(..., description="Estimated token count")
    content_preview: str = Field(..., description="Content preview text")
    total_length: int = Field(..., description="Total content length")

class ModelsResponse(BaseModel):
    """Response model for available models and configuration options."""
    models: List[str] = Field(..., description="Available LLM models")
    providers: List[str] = Field(..., description="Available LLM providers")
    formats: List[str] = Field(..., description="Available output formats")
    pipeline_types: List[str] = Field(..., description="Available pipeline types")
    pdf_parsers: List[str] = Field(..., description="Available PDF parsers")
    truncation_strategies: List[str] = Field(..., description="Available truncation strategies")
    defaults: Dict[str, Any] = Field(..., description="Default configuration values")

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status (healthy/degraded/unhealthy)")
    timestamp: float = Field(default_factory=time.time, description="Check timestamp")
    cache_size: int = Field(..., description="Number of cached pipelines")
    config_status: Optional[Dict[str, Any]] = Field(None, description="Configuration status details")
    error: Optional[str] = Field(None, description="Error message if unhealthy")

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request identifier")
```

## Frontend Interface

### Main Interface (`frontend/index.html`)

Modern, responsive web interface:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FinMapReduce - AI Document Analysis</title>
    <link rel="stylesheet" href="/static/style.css">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body>
    <!-- Header -->
    <header class="header">
        <div class="container">
            <h1><i class="fas fa-chart-line"></i> FinMapReduce</h1>
            <p>AI-Powered Financial Document Analysis</p>
        </div>
    </header>

    <!-- Main Content -->
    <main class="main-content">
        <div class="container">
            <!-- Three-panel layout -->
            <div class="panels">

                <!-- Upload Panel -->
                <div class="panel upload-panel">
                    <h2><i class="fas fa-upload"></i> Upload Document</h2>

                    <!-- File Upload Area -->
                    <div class="upload-area" id="uploadArea">
                        <div class="upload-content">
                            <i class="fas fa-cloud-upload-alt upload-icon"></i>
                            <p>Drag & drop your document here</p>
                            <p class="upload-hint">or <span class="browse-link">browse files</span></p>
                            <input type="file" id="fileInput" accept=".pdf,.txt,.md" hidden>
                        </div>
                        <div class="upload-progress" id="uploadProgress" style="display: none;">
                            <div class="progress-bar">
                                <div class="progress-fill" id="progressFill"></div>
                            </div>
                            <p class="progress-text" id="progressText">Uploading...</p>
                        </div>
                    </div>

                    <!-- File Info -->
                    <div class="file-info" id="fileInfo" style="display: none;">
                        <div class="file-details">
                            <span class="file-name" id="fileName"></span>
                            <span class="file-size" id="fileSize"></span>
                        </div>
                        <button class="btn-remove" id="removeFile">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>

                    <!-- Question Input -->
                    <div class="question-section">
                        <label for="questionInput">Your Question:</label>
                        <textarea
                            id="questionInput"
                            placeholder="e.g., What was the revenue growth in 2023?"
                            rows="3"
                        ></textarea>
                    </div>

                    <!-- Action Buttons -->
                    <div class="action-buttons">
                        <button id="previewBtn" class="btn btn-secondary" disabled>
                            <i class="fas fa-eye"></i> Preview Document
                        </button>
                        <button id="processBtn" class="btn btn-primary" disabled>
                            <i class="fas fa-play"></i> Get Answer
                        </button>
                    </div>

                    <!-- Advanced Configuration -->
                    <div class="advanced-config">
                        <button class="config-toggle" id="configToggle">
                            <i class="fas fa-cog"></i> Advanced Options
                        </button>
                        <div class="config-panel" id="configPanel" style="display: none;">
                            <!-- Model Configuration -->
                            <div class="config-group">
                                <label>Model:</label>
                                <select id="modelSelect">
                                    <option value="gpt-4o-mini">GPT-4o Mini</option>
                                    <option value="gpt-4o">GPT-4o</option>
                                    <option value="deepseek/deepseek-r1-0528:free">DeepSeek R1 (Free)</option>
                                </select>
                            </div>

                            <!-- Pipeline Configuration -->
                            <div class="config-group">
                                <label>Approach:</label>
                                <select id="pipelineSelect">
                                    <option value="mapreduce">MapReduce (Comprehensive)</option>
                                    <option value="truncation">Truncation (Fast)</option>
                                </select>
                            </div>

                            <!-- Format Configuration -->
                            <div class="config-group">
                                <label>Format:</label>
                                <select id="formatSelect">
                                    <option value="hybrid">Hybrid (Recommended)</option>
                                    <option value="json">JSON</option>
                                    <option value="plain_text">Plain Text</option>
                                </select>
                            </div>

                            <!-- Additional Parameters -->
                            <div class="config-group">
                                <label>Temperature:</label>
                                <input type="range" id="temperatureSlider" min="0" max="1" step="0.1" value="0">
                                <span class="range-value" id="temperatureValue">0.0</span>
                            </div>

                            <div class="config-group">
                                <label>Chunk Size:</label>
                                <select id="chunkSizeSelect">
                                    <option value="16384">16K (Fast)</option>
                                    <option value="32768" selected>32K (Balanced)</option>
                                    <option value="65536">64K (Comprehensive)</option>
                                </select>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Preview Panel -->
                <div class="panel preview-panel">
                    <h2><i class="fas fa-file-alt"></i> Document Preview</h2>
                    <div class="preview-content" id="previewContent">
                        <div class="preview-placeholder">
                            <i class="fas fa-file-text placeholder-icon"></i>
                            <p>Upload a document to see preview</p>
                        </div>
                    </div>

                    <!-- Document Stats -->
                    <div class="document-stats" id="documentStats" style="display: none;">
                        <div class="stat">
                            <span class="stat-label">Size:</span>
                            <span class="stat-value" id="statSize">-</span>
                        </div>
                        <div class="stat">
                            <span class="stat-label">Type:</span>
                            <span class="stat-value" id="statType">-</span>
                        </div>
                        <div class="stat">
                            <span class="stat-label">Est. Tokens:</span>
                            <span class="stat-value" id="statTokens">-</span>
                        </div>
                    </div>
                </div>

                <!-- Results Panel -->
                <div class="panel results-panel">
                    <h2><i class="fas fa-lightbulb"></i> AI Analysis</h2>
                    <div class="results-content" id="resultsContent">
                        <div class="results-placeholder">
                            <i class="fas fa-robot placeholder-icon"></i>
                            <p>Processing results will appear here</p>
                        </div>
                    </div>

                    <!-- Processing Status -->
                    <div class="processing-status" id="processingStatus" style="display: none;">
                        <div class="status-spinner"></div>
                        <p class="status-text" id="statusText">Processing your question...</p>
                    </div>

                    <!-- Results Display -->
                    <div class="results-display" id="resultsDisplay" style="display: none;">
                        <!-- Answer -->
                        <div class="result-section">
                            <h3><i class="fas fa-comment-alt"></i> Answer</h3>
                            <div class="answer-text" id="answerText"></div>
                        </div>

                        <!-- Reasoning -->
                        <div class="result-section">
                            <h3><i class="fas fa-brain"></i> Reasoning</h3>
                            <div class="reasoning-text" id="reasoningText"></div>
                        </div>

                        <!-- Evidence -->
                        <div class="result-section">
                            <h3><i class="fas fa-quote-left"></i> Evidence</h3>
                            <div class="evidence-list" id="evidenceList"></div>
                        </div>

                        <!-- Statistics -->
                        <div class="result-section collapsible">
                            <h3 class="collapsible-header">
                                <i class="fas fa-chart-bar"></i> Processing Statistics
                                <i class="fas fa-chevron-down toggle-icon"></i>
                            </h3>
                            <div class="collapsible-content" id="statisticsContent">
                                <div class="stats-grid" id="statsGrid"></div>
                            </div>
                        </div>
                    </div>
                </div>

            </div>
        </div>
    </main>

    <!-- Footer -->
    <footer class="footer">
        <div class="container">
            <p>&copy; 2025 FinMapReduce. AI-powered financial document analysis.</p>
        </div>
    </footer>

    <!-- Error Modal -->
    <div class="modal" id="errorModal" style="display: none;">
        <div class="modal-content">
            <div class="modal-header">
                <h3><i class="fas fa-exclamation-triangle"></i> Error</h3>
                <button class="modal-close" id="closeErrorModal">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="modal-body">
                <p id="errorMessage"></p>
            </div>
            <div class="modal-footer">
                <button class="btn btn-primary" id="errorModalOk">OK</button>
            </div>
        </div>
    </div>

    <script src="/static/app.js"></script>
</body>
</html>
```

### JavaScript Application (`frontend/static/app.js`)

Complete frontend logic implementation:

```javascript
// FinMapReduce Web Application
// Frontend JavaScript for document upload and processing

class FinMapReduceApp {
    constructor() {
        this.currentFile = null;
        this.currentPreview = null;
        this.processing = false;

        this.initializeElements();
        this.bindEvents();
        this.loadDefaultConfiguration();
    }

    initializeElements() {
        // File upload elements
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.uploadProgress = document.getElementById('uploadProgress');
        this.fileInfo = document.getElementById('fileInfo');

        // Form elements
        this.questionInput = document.getElementById('questionInput');
        this.previewBtn = document.getElementById('previewBtn');
        this.processBtn = document.getElementById('processBtn');

        // Configuration elements
        this.configToggle = document.getElementById('configToggle');
        this.configPanel = document.getElementById('configPanel');
        this.modelSelect = document.getElementById('modelSelect');
        this.pipelineSelect = document.getElementById('pipelineSelect');
        this.formatSelect = document.getElementById('formatSelect');
        this.temperatureSlider = document.getElementById('temperatureSlider');
        this.temperatureValue = document.getElementById('temperatureValue');
        this.chunkSizeSelect = document.getElementById('chunkSizeSelect');

        // Display elements
        this.previewContent = document.getElementById('previewContent');
        this.documentStats = document.getElementById('documentStats');
        this.resultsContent = document.getElementById('resultsContent');
        this.processingStatus = document.getElementById('processingStatus');
        this.resultsDisplay = document.getElementById('resultsDisplay');

        // Error modal
        this.errorModal = document.getElementById('errorModal');
    }

    bindEvents() {
        // File upload events
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        this.uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        document.getElementById('removeFile').addEventListener('click', () => this.removeFile());

        // Button events
        this.previewBtn.addEventListener('click', () => this.previewDocument());
        this.processBtn.addEventListener('click', () => this.processDocument());

        // Configuration events
        this.configToggle.addEventListener('click', () => this.toggleConfiguration());
        this.temperatureSlider.addEventListener('input', (e) => {
            this.temperatureValue.textContent = parseFloat(e.target.value).toFixed(1);
        });

        // Question input validation
        this.questionInput.addEventListener('input', () => this.validateForm());

        // Modal events
        document.getElementById('closeErrorModal').addEventListener('click', () => this.hideError());
        document.getElementById('errorModalOk').addEventListener('click', () => this.hideError());

        // Collapsible sections
        document.querySelectorAll('.collapsible-header').forEach(header => {
            header.addEventListener('click', () => this.toggleCollapsible(header));
        });
    }

    async loadDefaultConfiguration() {
        try {
            const response = await fetch('/api/models');
            if (response.ok) {
                const config = await response.json();
                this.populateConfigurationOptions(config);
            }
        } catch (error) {
            console.warn('Could not load default configuration:', error);
        }
    }

    populateConfigurationOptions(config) {
        // Populate model options
        this.modelSelect.innerHTML = '';
        config.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = this.formatModelName(model);
            option.selected = model === config.defaults.model_name;
            this.modelSelect.appendChild(option);
        });

        // Set other defaults
        this.pipelineSelect.value = config.defaults.pipeline_type;
        this.formatSelect.value = config.defaults.format_type;
        this.temperatureSlider.value = config.defaults.temperature;
        this.temperatureValue.textContent = config.defaults.temperature.toFixed(1);
        this.chunkSizeSelect.value = config.defaults.chunk_size;
    }

    formatModelName(model) {
        const modelNames = {
            'gpt-4o-mini': 'GPT-4o Mini',
            'gpt-4o': 'GPT-4o',
            'gpt-4-turbo': 'GPT-4 Turbo',
            'deepseek/deepseek-r1-0528:free': 'DeepSeek R1 (Free)',
            'meta-llama/llama-3.1-8b-instruct:free': 'Llama 3.1 8B (Free)'
        };
        return modelNames[model] || model;
    }

    // File handling methods
    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('drag-over');
    }

    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.setFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const files = e.target.files;
        if (files.length > 0) {
            this.setFile(files[0]);
        }
    }

    setFile(file) {
        // Validate file type
        const allowedTypes = ['.pdf', '.txt', '.md'];
        const fileExt = '.' + file.name.split('.').pop().toLowerCase();

        if (!allowedTypes.includes(fileExt)) {
            this.showError('Unsupported file type. Please upload PDF, TXT, or MD files.');
            return;
        }

        // Validate file size (50MB limit)
        const maxSize = 50 * 1024 * 1024;
        if (file.size > maxSize) {
            this.showError('File too large. Maximum size is 50MB.');
            return;
        }

        this.currentFile = file;
        this.displayFileInfo(file);
        this.validateForm();
    }

    displayFileInfo(file) {
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('fileSize').textContent = this.formatFileSize(file.size);

        this.uploadArea.style.display = 'none';
        this.fileInfo.style.display = 'flex';
    }

    removeFile() {
        this.currentFile = null;
        this.currentPreview = null;

        this.fileInfo.style.display = 'none';
        this.uploadArea.style.display = 'block';
        this.fileInput.value = '';

        // Clear preview
        this.previewContent.innerHTML = `
            <div class="preview-placeholder">
                <i class="fas fa-file-text placeholder-icon"></i>
                <p>Upload a document to see preview</p>
            </div>
        `;
        this.documentStats.style.display = 'none';

        this.validateForm();
    }

    // Document preview
    async previewDocument() {
        if (!this.currentFile) return;

        try {
            const formData = new FormData();
            formData.append('file', this.currentFile);

            this.previewContent.innerHTML = `
                <div class="loading-spinner">
                    <i class="fas fa-spinner fa-spin"></i>
                    <p>Generating preview...</p>
                </div>
            `;

            const response = await fetch('/api/preview', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Preview failed');
            }

            const preview = await response.json();
            this.displayPreview(preview);

        } catch (error) {
            this.showError('Preview failed: ' + error.message);
            this.previewContent.innerHTML = `
                <div class="preview-placeholder">
                    <i class="fas fa-exclamation-triangle placeholder-icon"></i>
                    <p>Preview not available</p>
                </div>
            `;
        }
    }

    displayPreview(preview) {
        // Display preview content
        this.previewContent.innerHTML = `
            <div class="preview-text">${this.escapeHtml(preview.content_preview)}</div>
        `;

        // Display document statistics
        document.getElementById('statSize').textContent = this.formatFileSize(preview.file_size);
        document.getElementById('statType').textContent = preview.file_type.toUpperCase();
        document.getElementById('statTokens').textContent = preview.estimated_tokens.toLocaleString();

        this.documentStats.style.display = 'grid';
        this.currentPreview = preview;
    }

    // Document processing
    async processDocument() {
        if (!this.currentFile || !this.questionInput.value.trim() || this.processing) return;

        this.processing = true;
        this.processBtn.disabled = true;
        this.processBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';

        // Show processing status
        this.resultsDisplay.style.display = 'none';
        this.processingStatus.style.display = 'block';
        document.getElementById('statusText').textContent = 'Processing your question...';

        try {
            const formData = this.buildProcessingRequest();

            const response = await fetch('/api/answer', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Processing failed');
            }

            const results = await response.json();
            this.displayResults(results);

        } catch (error) {
            this.showError('Processing failed: ' + error.message);
            this.resetProcessingState();
        }
    }

    buildProcessingRequest() {
        const formData = new FormData();
        formData.append('file', this.currentFile);
        formData.append('question', this.questionInput.value.trim());
        formData.append('model_name', this.modelSelect.value);
        formData.append('provider', this.getProviderForModel(this.modelSelect.value));
        formData.append('temperature', this.temperatureSlider.value);
        formData.append('pipeline_type', this.pipelineSelect.value);
        formData.append('format_type', this.formatSelect.value);
        formData.append('chunk_size', this.chunkSizeSelect.value);

        // Add advanced parameters if needed
        formData.append('chunk_overlap', Math.floor(parseInt(this.chunkSizeSelect.value) * 0.125));
        formData.append('score_threshold', '5');
        formData.append('max_total_requests', '20');
        formData.append('pdf_parser', 'marker');

        return formData;
    }

    getProviderForModel(model) {
        if (model.includes('/')) {
            return 'openrouter';  // Models with '/' are typically from OpenRouter
        }
        return 'openai';
    }

    displayResults(results) {
        // Hide processing status
        this.processingStatus.style.display = 'none';

        // Display answer
        document.getElementById('answerText').innerHTML = this.formatText(results.answer);

        // Display reasoning
        document.getElementById('reasoningText').innerHTML = this.formatText(results.reasoning);

        // Display evidence
        const evidenceList = document.getElementById('evidenceList');
        evidenceList.innerHTML = '';

        if (results.evidence && results.evidence.length > 0) {
            results.evidence.forEach((item, index) => {
                const evidenceItem = document.createElement('div');
                evidenceItem.className = 'evidence-item';
                evidenceItem.innerHTML = `
                    <div class="evidence-number">${index + 1}</div>
                    <div class="evidence-text">${this.escapeHtml(item)}</div>
                `;
                evidenceList.appendChild(evidenceItem);
            });
        } else {
            evidenceList.innerHTML = '<p class="no-evidence">No specific evidence provided.</p>';
        }

        // Display statistics
        this.displayStatistics(results);

        // Show results
        this.resultsDisplay.style.display = 'block';
        this.resetProcessingState();
    }

    displayStatistics(results) {
        const statsGrid = document.getElementById('statsGrid');
        const stats = [
            { label: 'Processing Time', value: `${results.timing_stats.total_time?.toFixed(1) || 0}s` },
            { label: 'Input Tokens', value: (results.token_stats.total_input_tokens || 0).toLocaleString() },
            { label: 'Output Tokens', value: (results.token_stats.total_output_tokens || 0).toLocaleString() },
            { label: 'Total Chunks', value: (results.chunk_stats.total_chunks || 0).toString() },
            { label: 'Processed Chunks', value: (results.chunk_stats.processed_chunks || 0).toString() }
        ];

        statsGrid.innerHTML = '';
        stats.forEach(stat => {
            const statItem = document.createElement('div');
            statItem.className = 'stat-item';
            statItem.innerHTML = `
                <div class="stat-label">${stat.label}</div>
                <div class="stat-value">${stat.value}</div>
            `;
            statsGrid.appendChild(statItem);
        });
    }

    resetProcessingState() {
        this.processing = false;
        this.processBtn.disabled = false;
        this.processBtn.innerHTML = '<i class="fas fa-play"></i> Get Answer';
    }

    // UI helper methods
    validateForm() {
        const hasFile = this.currentFile !== null;
        const hasQuestion = this.questionInput.value.trim().length > 0;
        const isValid = hasFile && hasQuestion && !this.processing;

        this.previewBtn.disabled = !hasFile;
        this.processBtn.disabled = !isValid;
    }

    toggleConfiguration() {
        const isVisible = this.configPanel.style.display === 'block';
        this.configPanel.style.display = isVisible ? 'none' : 'block';

        const icon = this.configToggle.querySelector('i');
        icon.className = isVisible ? 'fas fa-cog' : 'fas fa-times';
    }

    toggleCollapsible(header) {
        const content = header.nextElementSibling;
        const icon = header.querySelector('.toggle-icon');
        const isOpen = content.style.display === 'block';

        content.style.display = isOpen ? 'none' : 'block';
        icon.className = isOpen ? 'fas fa-chevron-down' : 'fas fa-chevron-up';
    }

    showError(message) {
        document.getElementById('errorMessage').textContent = message;
        this.errorModal.style.display = 'block';
    }

    hideError() {
        this.errorModal.style.display = 'none';
    }

    // Utility methods
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    formatText(text) {
        if (!text) return '';
        // Convert newlines to HTML breaks and preserve formatting
        return this.escapeHtml(text).replace(/\n/g, '<br>');
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new FinMapReduceApp();
});
```

### Styling (`frontend/static/style.css`)

Modern, responsive CSS design:

```css
/* FinMapReduce Web Application Styles */

/* Reset and base styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

:root {
    /* Color scheme */
    --primary-color: #2563eb;
    --primary-hover: #1d4ed8;
    --secondary-color: #64748b;
    --success-color: #059669;
    --warning-color: #d97706;
    --error-color: #dc2626;

    /* Background colors */
    --bg-primary: #f8fafc;
    --bg-secondary: #ffffff;
    --bg-accent: #f1f5f9;

    /* Text colors */
    --text-primary: #1e293b;
    --text-secondary: #475569;
    --text-muted: #64748b;

    /* Border and shadow */
    --border-color: #e2e8f0;
    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);

    /* Spacing */
    --spacing-xs: 0.5rem;
    --spacing-sm: 0.75rem;
    --spacing-md: 1rem;
    --spacing-lg: 1.5rem;
    --spacing-xl: 2rem;
    --spacing-2xl: 3rem;

    /* Border radius */
    --radius-sm: 0.375rem;
    --radius-md: 0.5rem;
    --radius-lg: 0.75rem;

    /* Typography */
    --font-sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --font-mono: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
}

body {
    font-family: var(--font-sans);
    line-height: 1.6;
    color: var(--text-primary);
    background-color: var(--bg-primary);
}

/* Header */
.header {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-hover) 100%);
    color: white;
    padding: var(--spacing-xl) 0;
    box-shadow: var(--shadow-md);
}

.header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: var(--spacing-sm);
}

.header h1 i {
    margin-right: var(--spacing-md);
    color: #fbbf24;
}

.header p {
    font-size: 1.1rem;
    opacity: 0.9;
}

/* Container */
.container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 0 var(--spacing-md);
}

/* Main content */
.main-content {
    padding: var(--spacing-2xl) 0;
    min-height: calc(100vh - 200px);
}

/* Three-panel layout */
.panels {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: var(--spacing-xl);
    align-items: start;
}

@media (max-width: 1200px) {
    .panels {
        grid-template-columns: 1fr;
        gap: var(--spacing-lg);
    }
}

/* Panel styles */
.panel {
    background: var(--bg-secondary);
    border-radius: var(--radius-lg);
    padding: var(--spacing-xl);
    box-shadow: var(--shadow-md);
    border: 1px solid var(--border-color);
}

.panel h2 {
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: var(--spacing-lg);
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: var(--spacing-md);
}

.panel h2 i {
    color: var(--primary-color);
}

/* Upload area */
.upload-area {
    border: 2px dashed var(--border-color);
    border-radius: var(--radius-md);
    padding: var(--spacing-2xl);
    text-align: center;
    background: var(--bg-accent);
    transition: all 0.3s ease;
    cursor: pointer;
    margin-bottom: var(--spacing-lg);
}

.upload-area:hover {
    border-color: var(--primary-color);
    background: #eff6ff;
}

.upload-area.drag-over {
    border-color: var(--primary-color);
    background: #eff6ff;
    transform: scale(1.02);
}

.upload-icon {
    font-size: 3rem;
    color: var(--primary-color);
    margin-bottom: var(--spacing-md);
}

.upload-area p {
    color: var(--text-secondary);
    margin-bottom: var(--spacing-sm);
}

.browse-link {
    color: var(--primary-color);
    cursor: pointer;
    text-decoration: underline;
}

.browse-link:hover {
    color: var(--primary-hover);
}

/* Upload progress */
.upload-progress {
    margin-top: var(--spacing-md);
}

.progress-bar {
    width: 100%;
    height: 8px;
    background: var(--bg-accent);
    border-radius: var(--radius-sm);
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--primary-color), var(--success-color));
    transition: width 0.3s ease;
    width: 0%;
}

.progress-text {
    margin-top: var(--spacing-sm);
    color: var(--text-secondary);
    font-size: 0.9rem;
}

/* File info */
.file-info {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--spacing-md);
    background: var(--bg-accent);
    border-radius: var(--radius-md);
    margin-bottom: var(--spacing-lg);
}

.file-details {
    display: flex;
    flex-direction: column;
}

.file-name {
    font-weight: 600;
    color: var(--text-primary);
}

.file-size {
    font-size: 0.9rem;
    color: var(--text-secondary);
}

.btn-remove {
    background: var(--error-color);
    color: white;
    border: none;
    border-radius: 50%;
    width: 32px;
    height: 32px;
    cursor: pointer;
    transition: background 0.2s ease;
}

.btn-remove:hover {
    background: #b91c1c;
}

/* Question section */
.question-section {
    margin-bottom: var(--spacing-lg);
}

.question-section label {
    display: block;
    font-weight: 600;
    margin-bottom: var(--spacing-sm);
    color: var(--text-primary);
}

.question-section textarea {
    width: 100%;
    padding: var(--spacing-md);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    font-family: inherit;
    font-size: 1rem;
    resize: vertical;
    transition: border-color 0.2s ease;
}

.question-section textarea:focus {
    outline: none;
    border-color: var(--primary-color);
    box-shadow: 0 0 0 3px rgb(37 99 235 / 0.1);
}

/* Buttons */
.action-buttons {
    display: flex;
    gap: var(--spacing-md);
    margin-bottom: var(--spacing-lg);
}

.btn {
    padding: var(--spacing-md) var(--spacing-lg);
    border: none;
    border-radius: var(--radius-md);
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
    font-size: 1rem;
    flex: 1;
    justify-content: center;
}

.btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
}

.btn-primary {
    background: var(--primary-color);
    color: white;
}

.btn-primary:hover:not(:disabled) {
    background: var(--primary-hover);
    transform: translateY(-1px);
}

.btn-secondary {
    background: var(--secondary-color);
    color: white;
}

.btn-secondary:hover:not(:disabled) {
    background: #475569;
    transform: translateY(-1px);
}

/* Advanced configuration */
.advanced-config {
    border-top: 1px solid var(--border-color);
    padding-top: var(--spacing-lg);
}

.config-toggle {
    background: var(--bg-accent);
    border: 1px solid var(--border-color);
    padding: var(--spacing-md);
    border-radius: var(--radius-md);
    cursor: pointer;
    width: 100%;
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
    font-weight: 600;
    color: var(--text-primary);
    transition: background 0.2s ease;
}

.config-toggle:hover {
    background: #e2e8f0;
}

.config-panel {
    margin-top: var(--spacing-md);
    padding: var(--spacing-lg);
    background: var(--bg-accent);
    border-radius: var(--radius-md);
}

.config-group {
    margin-bottom: var(--spacing-md);
}

.config-group:last-child {
    margin-bottom: 0;
}

.config-group label {
    display: block;
    font-weight: 600;
    margin-bottom: var(--spacing-sm);
    color: var(--text-primary);
}

.config-group select,
.config-group input[type="range"] {
    width: 100%;
    padding: var(--spacing-sm);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-sm);
    font-size: 0.9rem;
}

.range-value {
    margin-left: var(--spacing-sm);
    font-weight: 600;
    color: var(--primary-color);
}

/* Preview content */
.preview-content {
    min-height: 300px;
    max-height: 500px;
    overflow-y: auto;
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    padding: var(--spacing-md);
    background: var(--bg-accent);
    margin-bottom: var(--spacing-lg);
}

.preview-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 200px;
    color: var(--text-muted);
}

.placeholder-icon {
    font-size: 3rem;
    margin-bottom: var(--spacing-md);
    opacity: 0.5;
}

.preview-text {
    font-family: var(--font-mono);
    font-size: 0.9rem;
    line-height: 1.5;
    white-space: pre-wrap;
    color: var(--text-secondary);
}

/* Document stats */
.document-stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: var(--spacing-md);
}

.stat {
    text-align: center;
    padding: var(--spacing-sm);
    background: var(--bg-accent);
    border-radius: var(--radius-sm);
}

.stat-label {
    display: block;
    font-size: 0.8rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.stat-value {
    display: block;
    font-weight: 700;
    color: var(--text-primary);
    font-size: 1.1rem;
}

/* Processing status */
.processing-status {
    text-align: center;
    padding: var(--spacing-2xl);
}

.status-spinner {
    width: 48px;
    height: 48px;
    border: 4px solid var(--bg-accent);
    border-top: 4px solid var(--primary-color);
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 0 auto var(--spacing-lg);
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.status-text {
    color: var(--text-secondary);
    font-size: 1.1rem;
}

/* Results display */
.results-display {
    animation: fadeIn 0.5s ease-in;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.result-section {
    margin-bottom: var(--spacing-xl);
    padding-bottom: var(--spacing-lg);
    border-bottom: 1px solid var(--border-color);
}

.result-section:last-child {
    border-bottom: none;
}

.result-section h3 {
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: var(--spacing-md);
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
}

.result-section h3 i {
    color: var(--primary-color);
}

.answer-text,
.reasoning-text {
    font-size: 1.1rem;
    line-height: 1.7;
    color: var(--text-primary);
}

.answer-text {
    background: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%);
    padding: var(--spacing-lg);
    border-radius: var(--radius-md);
    border-left: 4px solid var(--primary-color);
}

/* Evidence */
.evidence-list {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-md);
}

.evidence-item {
    display: flex;
    gap: var(--spacing-md);
    padding: var(--spacing-md);
    background: var(--bg-accent);
    border-radius: var(--radius-md);
}

.evidence-number {
    background: var(--primary-color);
    color: white;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    font-weight: 700;
    flex-shrink: 0;
}

.evidence-text {
    flex: 1;
    color: var(--text-secondary);
    line-height: 1.6;
}

.no-evidence {
    color: var(--text-muted);
    font-style: italic;
    text-align: center;
    padding: var(--spacing-lg);
}

/* Collapsible sections */
.collapsible-header {
    cursor: pointer;
    user-select: none;
    justify-content: space-between;
}

.collapsible-header:hover {
    color: var(--primary-color);
}

.toggle-icon {
    transition: transform 0.3s ease;
}

.collapsible-content {
    display: none;
}

/* Statistics */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: var(--spacing-md);
}

.stat-item {
    background: var(--bg-accent);
    padding: var(--spacing-md);
    border-radius: var(--radius-md);
    text-align: center;
}

.stat-item .stat-label {
    font-size: 0.9rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: var(--spacing-xs);
}

.stat-item .stat-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--primary-color);
}

/* Loading spinner */
.loading-spinner {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 200px;
    color: var(--text-secondary);
}

.loading-spinner i {
    font-size: 2rem;
    margin-bottom: var(--spacing-md);
    color: var(--primary-color);
}

/* Error modal */
.modal {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
}

.modal-content {
    background: var(--bg-secondary);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-lg);
    max-width: 500px;
    width: 90%;
    max-height: 80vh;
    overflow-y: auto;
}

.modal-header {
    padding: var(--spacing-lg);
    border-bottom: 1px solid var(--border-color);
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.modal-header h3 {
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--error-color);
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
}

.modal-close {
    background: none;
    border: none;
    font-size: 1.2rem;
    cursor: pointer;
    color: var(--text-muted);
    padding: var(--spacing-sm);
}

.modal-close:hover {
    color: var(--text-primary);
}

.modal-body {
    padding: var(--spacing-lg);
}

.modal-body p {
    color: var(--text-secondary);
    line-height: 1.6;
}

.modal-footer {
    padding: var(--spacing-lg);
    border-top: 1px solid var(--border-color);
    text-align: right;
}

/* Footer */
.footer {
    background: var(--text-primary);
    color: white;
    text-align: center;
    padding: var(--spacing-lg) 0;
    margin-top: var(--spacing-2xl);
}

.footer p {
    opacity: 0.8;
}

/* Responsive design */
@media (max-width: 768px) {
    .header h1 {
        font-size: 2rem;
    }

    .panels {
        grid-template-columns: 1fr;
        gap: var(--spacing-lg);
    }

    .panel {
        padding: var(--spacing-lg);
    }

    .action-buttons {
        flex-direction: column;
    }

    .document-stats {
        grid-template-columns: 1fr;
    }

    .stats-grid {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 480px) {
    .container {
        padding: 0 var(--spacing-sm);
    }

    .upload-area {
        padding: var(--spacing-lg);
    }

    .upload-icon {
        font-size: 2rem;
    }

    .panel h2 {
        font-size: 1.3rem;
    }
}

/* Utility classes */
.text-center { text-align: center; }
.text-muted { color: var(--text-muted); }
.mb-0 { margin-bottom: 0; }
.mt-lg { margin-top: var(--spacing-lg); }
.hidden { display: none; }

/* Print styles */
@media print {
    .header,
    .footer,
    .upload-panel,
    .processing-status {
        display: none;
    }

    .panels {
        grid-template-columns: 1fr 1fr;
        gap: var(--spacing-md);
    }

    .panel {
        box-shadow: none;
        border: 1px solid var(--border-color);
        page-break-inside: avoid;
    }
}
```

This comprehensive web application documentation covers all aspects of the frontend and backend implementation, providing complete information for users and developers working with the FinMapReduce web interface.