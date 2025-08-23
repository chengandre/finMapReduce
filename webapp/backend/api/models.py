from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, List, Union


class AnswerRequest(BaseModel):
    """Request model for the answer endpoint."""
    model_config = ConfigDict(protected_namespaces=())

    question: str = Field(..., description="The question to answer")
    model_name: Optional[str] = Field("gpt-4o-mini", description="LLM model name")
    provider: Optional[str] = Field("openai", description="LLM provider (openai/openrouter)")
    temperature: Optional[float] = Field(0.0, description="Model temperature")
    chunk_size: Optional[int] = Field(36000, description="Document chunk size")
    chunk_overlap: Optional[int] = Field(1000, description="Chunk overlap size")
    format_type: Optional[str] = Field("hybrid", description="Output format (json/hybrid/plain)")
    pdf_parser: Optional[str] = Field("marker", description="PDF parser method")
    max_concurrent_chunks: Optional[int] = Field(20, description="Max concurrent chunks in map phase")
    pipeline_type: Optional[str] = Field("mapreduce", description="Pipeline type (mapreduce/truncation)")
    strategy: Optional[str] = Field("start", description="Truncation strategy (start/end/smart) - for truncation pipeline")
    context_window: Optional[int] = Field(128000, description="Context window size - for truncation pipeline")
    buffer: Optional[int] = Field(2000, description="Buffer size for response tokens - for truncation pipeline")
    max_document_tokens: Optional[int] = Field(None, description="Override for max document tokens - for truncation pipeline")
    score_threshold: Optional[int] = Field(5, description="Score threshold for filtering map results")
    prompt_set: Optional[str] = Field("hybrid", description="Prompt set to use (default/hybrid/baseline/etc.)")
    requests_per_minute: Optional[int] = Field(30000, description="Rate limit: requests per minute")
    tokens_per_minute: Optional[int] = Field(150000000, description="Rate limit: tokens per minute")
    request_burst_size: Optional[int] = Field(3000, description="Rate limit: maximum burst size")
    max_total_requests: Optional[int] = Field(1000, description="Maximum total concurrent requests")
    key_type: Optional[str] = Field("default", description="API key type (default/self)")


class AnswerResponse(BaseModel):
    """Response model for the answer endpoint."""
    answer: str = Field(..., description="The generated answer")
    reasoning: str = Field(..., description="Reasoning behind the answer")
    evidence: Union[str, List[str]] = Field(..., description="Evidence from the document")
    token_stats: Dict[str, Any] = Field(..., description="Token usage statistics")
    timing_stats: Dict[str, Any] = Field(..., description="Processing time statistics")
    chunk_stats: Dict[str, Any] = Field(..., description="Document chunk statistics")
    request_id: str = Field(..., description="Unique request identifier")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: Dict[str, str] = Field(..., description="Error details")

    @classmethod
    def create(cls, message: str, error_type: str = "Error", request_id: Optional[str] = None):
        """Create an error response."""
        error_dict = {
            "message": message,
            "type": error_type
        }
        if request_id:
            error_dict["request_id"] = request_id

        return cls(error=error_dict)


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Response timestamp")
    version: str = Field(..., description="API version")