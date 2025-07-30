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


# LangChain imports
try:
    from langchain.schema import Document
    from langchain_community.vectorstores import FAISS
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: langchain not available")

# Local embeddings imports and setup
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available, falling back to OpenAI embeddings")
    try:
        from langchain_openai import OpenAIEmbeddings
        OPENAI_EMBEDDINGS_AVAILABLE = True
    except ImportError:
        OPENAI_EMBEDDINGS_AVAILABLE = False
        print("Warning: langchain_openai not available either")


class LocalEmbeddings:
    """Custom local embeddings using sentence-transformers with model caching"""

    # Class-level cache for models to avoid repeated downloads
    _model_cache = {}

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required for local embeddings")

        self.model_name = model_name

        # Check if model is already cached
        if model_name not in self._model_cache:
            logger.info(f"Loading local embedding model: {model_name}")
            self._model_cache[model_name] = SentenceTransformer(model_name)
            logger.info(f"Cached local embedding model: {model_name}")
        else:
            logger.info(f"Using cached local embedding model: {model_name}")

        self.model = self._model_cache[model_name]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        embedding = self.model.encode([text], convert_to_tensor=False)
        return embedding[0].tolist()

    def __call__(self, text: str) -> List[float]:
        """Make the class callable for LangChain compatibility"""
        return self.embed_query(text)

    @classmethod
    def clear_cache(cls):
        """Clear the model cache (useful for testing or memory management)"""
        cls._model_cache.clear()
        logger.info("Cleared embedding model cache")


class VectorStore:
    """Manages FAISS vector index and embeddings using LangChain's FAISS wrapper"""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", use_local: bool = True):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("langchain is required for vector store functionality")
            
        self.embedding_model = embedding_model
        self.use_local = use_local

        if use_local and SENTENCE_TRANSFORMERS_AVAILABLE:
            # Use local embeddings
            self.embeddings = LocalEmbeddings(embedding_model)
            logger.info(f"VectorStore initialized with local embedding model: {embedding_model}")
        else:
            # Fallback to OpenAI embeddings
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.warning("sentence-transformers not available, using OpenAI embeddings")
            if not OPENAI_EMBEDDINGS_AVAILABLE:
                raise ImportError("Neither sentence-transformers nor OpenAI embeddings available")
            self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
            logger.info("VectorStore initialized with OpenAI embeddings")

        self.vector_store = None  # LangChain FAISS vector store

    def add_documents(self, documents: List['Document']) -> None:
        """Add documents to the vector store"""
        if not documents:
            logger.warning("No documents provided to add_documents")
            return

        logger.info(f"Adding {len(documents)} documents to vector store")

        try:
            if self.vector_store is None:
                # Create new FAISS vector store with first batch of documents
                logger.info("Creating new FAISS vector store...")
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
                logger.info(f"Created FAISS vector store with {len(documents)} documents")
            else:
                # Add documents to existing vector store
                logger.info("Adding documents to existing FAISS vector store...")
                self.vector_store.add_documents(documents)
                logger.info(f"Added {len(documents)} documents to existing vector store")

            total_docs = len(self.vector_store.docstore._dict)
            logger.info(f"Total documents in vector store: {total_docs}")

        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise SimpleRAGError(f"Failed to add documents to vector store: {str(e)}")

    def similarity_search(self, query: str, document_id: Optional[str] = None,
                         top_k: int = 5) -> List[Tuple['Document', float]]:
        """Perform similarity search with optional document filtering"""
        if self.vector_store is None:
            logger.warning("No documents in vector store")
            return []

        try:
            logger.info(f"Performing similarity search for query: {query[:50]}...")

            if document_id:
                # For document-specific search, we need to search more and then filter
                # since LangChain FAISS doesn't have built-in metadata filtering
                search_k = min(100, top_k * 10)  # Search more to account for filtering
                results_with_scores = self.vector_store.similarity_search_with_score(
                    query, k=search_k
                )

                # Filter by document_id
                filtered_results = []
                for doc, score in results_with_scores:
                    if doc.metadata.get("document_id") == document_id:
                        filtered_results.append((doc, score))
                        if len(filtered_results) >= top_k:
                            break

                results = filtered_results[:top_k]
                logger.info(f"Found {len(results)} documents matching document_id '{document_id}'")

            else:
                # Global search across all documents
                results_with_scores = self.vector_store.similarity_search_with_score(
                    query, k=top_k
                )
                results = [(doc, score) for doc, score in results_with_scores]
                logger.info(f"Found {len(results)} relevant documents globally")

            return results

        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            raise QueryError(f"Failed to perform similarity search: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        if self.vector_store is None:
            return {
                "total_documents": 0,
                "embedding_model": self.embedding_model
            }

        return {
            "total_documents": len(self.vector_store.docstore._dict),
            "embedding_model": self.embedding_model
        }

    def clear(self) -> None:
        """Clear the vector store"""
        self.vector_store = None
        logger.info("Vector store cleared")

    def save_local(self, folder_path: str) -> None:
        """Save the vector store to local disk"""
        if self.vector_store is not None:
            self.vector_store.save_local(folder_path)
            logger.info(f"Vector store saved to {folder_path}")

    def load_local(self, folder_path: str) -> None:
        """Load the vector store from local disk"""
        try:
            self.vector_store = FAISS.load_local(
                folder_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Vector store loaded from {folder_path}")
        except Exception as e:
            logger.error(f"Error loading vector store from {folder_path}: {str(e)}")
            raise SimpleRAGError(f"Failed to load vector store: {str(e)}")


if __name__ == "__main__":
    # Basic test to verify imports and initialization
    try:
        print("SimpleRAG foundation utilities loaded successfully!")
        # Test basic validation
        ValidationUtils.validate_document_id("test_doc_123")
        ValidationUtils.validate_query("What is the revenue?")
        print("Validation utilities working correctly!")
        
        # Test embeddings if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            print("Testing LocalEmbeddings...")
            embeddings = LocalEmbeddings()
            test_embedding = embeddings.embed_query("test query")
            print(f"Embedding dimension: {len(test_embedding)}")
            print("LocalEmbeddings working correctly!")
        else:
            print("sentence-transformers not available, skipping embeddings test")
            
        # Test VectorStore initialization if available
        if LANGCHAIN_AVAILABLE:
            print("Testing VectorStore initialization...")
            try:
                vector_store = VectorStore()
                stats = vector_store.get_stats()
                print(f"VectorStore stats: {stats}")
                print("VectorStore working correctly!")
            except Exception as vs_error:
                print(f"VectorStore test failed (expected if dependencies missing): {vs_error}")
        else:
            print("langchain not available, skipping vector store test")
            
    except Exception as e:
        print(f"Error testing utilities: {e}")