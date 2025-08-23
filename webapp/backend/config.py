import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file in parent directory
env_path = os.path.join(os.path.dirname(__file__), '../../.env')
loaded = load_dotenv(env_path)
print(f"Loading .env from: {env_path}")
print(f".env loaded successfully: {loaded}")
print(f"OPENAI_API_KEY: {'set' if os.getenv('OPENAI_API_KEY') else 'not set'}")
print(f"OPENROUTER_API_KEY: {'set' if os.getenv('OPENROUTER_API_KEY') else 'not set'}")


class Settings:
    """Configuration settings for the webapp."""

    def __init__(self):
        # API Keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.self_openai_api_key = os.getenv("SELF_OPENAI_API_KEY")

        # Default LLM Settings
        self.default_model = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
        self.default_provider = os.getenv("DEFAULT_PROVIDER", "openai")
        self.default_temperature = float(os.getenv("DEFAULT_TEMPERATURE", "0.0"))
        self.default_chunk_size = int(os.getenv("DEFAULT_CHUNK_SIZE", "32768"))
        self.default_chunk_overlap = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "4096"))
        self.default_format_type = os.getenv("DEFAULT_FORMAT_TYPE", "hybrid")
        self.default_pdf_parser = os.getenv("DEFAULT_PDF_PARSER", "marker")
        self.default_max_concurrent_chunks = int(os.getenv("DEFAULT_MAX_CONCURRENT_CHUNKS", "50"))

        # File Upload Settings
        self.max_file_size = int(os.getenv("MAX_FILE_SIZE", str(50 * 1024 * 1024)))  # 50MB
        self.allowed_extensions = [".pdf", ".txt", ".md"]
        self.temp_dir = os.getenv("TEMP_DIR", "/tmp")

        # API Settings
        self.cors_origins = ["*"]
        self.api_title = os.getenv("API_TITLE", "MapReduce QA WebApp")
        self.api_version = os.getenv("API_VERSION", "1.0.0")


# Global settings instance
settings = Settings()


def get_api_key(provider: str = "openai", key_type: str = "default") -> Optional[str]:
    """
    Get API key based on provider and key type.

    Args:
        provider: "openai" or "openrouter"
        key_type: "default" or "self"

    Returns:
        API key string or None
    """
    if provider == "openai":
        if key_type == "self":
            return settings.self_openai_api_key or os.getenv("SELF_OPENAI_API_KEY")
        else:
            return settings.openai_api_key or os.getenv("OPENAI_API_KEY")
    elif provider == "openrouter":
        return settings.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def validate_file_upload(filename: str, file_size: int) -> None:
    """
    Validate uploaded file.

    Args:
        filename: Name of the uploaded file
        file_size: Size of the file in bytes

    Raises:
        ValueError: If file is invalid
    """
    # Check file extension
    file_ext = os.path.splitext(filename)[1].lower()
    if file_ext not in settings.allowed_extensions:
        raise ValueError(f"Unsupported file type: {file_ext}. Allowed types: {', '.join(settings.allowed_extensions)}")

    # Check file size
    if file_size > settings.max_file_size:
        max_size_mb = settings.max_file_size / 1024 / 1024
        actual_size_mb = file_size / 1024 / 1024
        raise ValueError(f"File size ({actual_size_mb:.1f}MB) exceeds maximum allowed size ({max_size_mb:.1f}MB)")


def get_model_config(model_name: Optional[str] = None,
                    provider: Optional[str] = None,
                    temperature: Optional[float] = None,
                    **kwargs) -> dict:
    """
    Get model configuration with defaults.

    Args:
        model_name: Model name (defaults to default_model)
        provider: Provider name (defaults to default_provider)
        temperature: Temperature (defaults to default_temperature)
        **kwargs: Additional model parameters

    Returns:
        Dictionary with model configuration
    """
    config = {
        "model_name": model_name or settings.default_model,
        "provider": provider or settings.default_provider,
        "temperature": temperature if temperature is not None else settings.default_temperature,
        "chunk_size": kwargs.get("chunk_size", settings.default_chunk_size),
        "chunk_overlap": kwargs.get("chunk_overlap", settings.default_chunk_overlap),
        "format_type": kwargs.get("format_type", settings.default_format_type),
        "pdf_parser": kwargs.get("pdf_parser", settings.default_pdf_parser),
        "max_concurrent_chunks": kwargs.get("max_concurrent_chunks", settings.default_max_concurrent_chunks),
    }

    # Add any additional kwargs
    for key, value in kwargs.items():
        if key not in config:
            config[key] = value

    return config