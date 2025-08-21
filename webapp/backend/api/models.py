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