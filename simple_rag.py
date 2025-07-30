"""
Simple RAG Baseline System

A lightweight retrieval-augmented generation system that can ingest documents
with metadata and perform similarity-based retrieval for question answering.
Supports both document-specific queries (filtered by metadata) and global queries.
"""

import os
import logging
import pickle
import re
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationUtils:
    """Utility class for input validation and file validation"""

    # Supported file extensions
    SUPPORTED_EXTENSIONS = {'.pdf', '.md'}

    # Valid document ID pattern (alphanumeric, underscores, hyphens)
    DOCUMENT_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')

    @staticmethod
    def validate_document_name(document_name: str) -> None:
        """
        Validate document name for ingestion

        Args:
            document_name: Name of the document to validate

        Raises:
            ValidationError: If document name is invalid
        """
        if not document_name or not document_name.strip():
            raise ValidationError("Document name cannot be empty")

        # Check for potentially problematic characters
        if any(char in document_name for char in ['<', '>', ':', '"', '|', '?', '*']):
            raise ValidationError(
                f"Document name '{document_name}' contains invalid characters. "
                "Avoid: < > : \" | ? *"
            )

        # Check length
        if len(document_name) > 255:
            raise ValidationError(
                f"Document name too long ({len(document_name)} characters). "
                "Maximum length is 255 characters."
            )

    @staticmethod
    def validate_document_id(document_id: str) -> None:
        """
        Validate document ID format

        Args:
            document_id: Document ID to validate

        Raises:
            ValidationError: If document ID is invalid
        """
        if not document_id or not document_id.strip():
            raise ValidationError("Document ID cannot be empty")

        # Remove whitespace for validation
        clean_id = document_id.strip()

        # Check length
        if len(clean_id) < 1:
            raise ValidationError("Document ID cannot be empty after removing whitespace")

        if len(clean_id) > 100:
            raise ValidationError(
                f"Document ID too long ({len(clean_id)} characters). "
                "Maximum length is 100 characters."
            )

        # Check format (alphanumeric, underscores, hyphens only)
        if not ValidationUtils.DOCUMENT_ID_PATTERN.match(clean_id):
            raise ValidationError(
                f"Document ID '{clean_id}' contains invalid characters. "
                "Only letters, numbers, underscores, and hyphens are allowed."
            )

    @staticmethod
    def validate_query(query: str) -> str:
        """
        Validate and clean query string

        Args:
            query: Query string to validate

        Returns:
            Cleaned query string

        Raises:
            ValidationError: If query is invalid
        """
        if query is None:
            raise ValidationError("Query cannot be None2")

        if not isinstance(query, str):
            raise ValidationError(f"Query must be a string, got {type(query).__name__}")

        # Clean whitespace
        clean_query = query.strip()

        if not clean_query:
            raise ValidationError("Query cannot be empty")

        # Check length
        if len(clean_query) > 10000:
            raise ValidationError(
                "Query too long."
            )

        # Check for potentially problematic patterns
        if clean_query.count('\n') > 50:
            raise ValidationError("Query contains too many line breaks")

        return clean_query

    @staticmethod
    def validate_top_k(top_k: Union[int, str]) -> int:
        """
        Validate top_k parameter

        Args:
            top_k: Number of results to retrieve

        Returns:
            Validated top_k value

        Raises:
            ValidationError: If top_k is invalid
        """
        # Handle None case
        if top_k is None:
            raise ValidationError("top_k must be an integer, got NoneType")

        # Handle float case (reject floats)
        if isinstance(top_k, float):
            raise ValidationError(f"top_k must be an integer, got {type(top_k).__name__}")

        # Try to convert to int if not already
        if not isinstance(top_k, int):
            try:
                top_k = int(top_k)
            except (ValueError, TypeError):
                raise ValidationError(f"top_k must be an integer, got {type(top_k).__name__}")

        if top_k < 1:
            raise ValidationError(f"top_k must be at least 1, got {top_k}")

        if top_k > 100:
            raise ValidationError(f"top_k too large ({top_k}). Maximum is 100.")

        return top_k

    @staticmethod
    def validate_chunk_parameters(chunk_size: int, chunk_overlap: int) -> Tuple[int, int]:
        """
        Validate chunk size and overlap parameters

        Args:
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks

        Returns:
            Tuple of validated (chunk_size, chunk_overlap)

        Raises:
            ValidationError: If parameters are invalid
        """
        # Validate chunk_size
        if not isinstance(chunk_size, int):
            try:
                chunk_size = int(chunk_size)
            except (ValueError, TypeError):
                raise ValidationError(f"chunk_size must be an integer, got {type(chunk_size).__name__}")

        if chunk_size < 100:
            raise ValidationError(f"chunk_size too small ({chunk_size}). Minimum is 100.")

        if chunk_size > 10000:
            raise ValidationError(f"chunk_size too large ({chunk_size}). Maximum is 10,000.")

        # Validate chunk_overlap
        if not isinstance(chunk_overlap, int):
            try:
                chunk_overlap = int(chunk_overlap)
            except (ValueError, TypeError):
                raise ValidationError(f"chunk_overlap must be an integer, got {type(chunk_overlap).__name__}")

        if chunk_overlap < 0:
            raise ValidationError(f"chunk_overlap cannot be negative, got {chunk_overlap}")

        if chunk_overlap >= chunk_size:
            raise ValidationError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )

        return chunk_size, chunk_overlap


@dataclass
class QueryResponse:
    """Response structure for RAG queries"""
    answer: str
    source_chunks: List['Document']
    document_ids: List[str]
    confidence_scores: List[float]
    query_time: float


class SimpleRAGError(Exception):
    """Base exception for SimpleRAG system"""
    pass


class DocumentProcessingError(SimpleRAGError):
    """Exception raised during document processing"""
    pass


class QueryError(SimpleRAGError):
    """Exception raised during query processing"""
    pass


class ValidationError(SimpleRAGError):
    """Exception raised during input validation"""
    pass


class FileValidationError(DocumentProcessingError):
    """Exception raised during file validation"""
    pass


class DuplicateDocumentError(DocumentProcessingError):
    """Exception raised when attempting to ingest duplicate document ID"""
    def __init__(self, document_id: str, message: str = None):
        self.document_id = document_id
        if message is None:
            message = f"Document ID '{document_id}' already exists"
        super().__init__(message)


if __name__ == "__main__":
    # Basic test to verify imports and initialization
    try:
        print("SimpleRAG foundation utilities loaded successfully!")
        # Test basic validation
        ValidationUtils.validate_document_id("test_doc_123")
        ValidationUtils.validate_query("What is the revenue?")
        print("Validation utilities working correctly!")
    except Exception as e:
        print(f"Error testing foundation utilities: {e}")