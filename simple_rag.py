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
    from langchain.prompts import PromptTemplate, load_prompt
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: langchain not available")

# Import existing utilities
try:
    from utils import load_pdf_chunk, GPT
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    print("Warning: utils module not available")

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


class DocumentProcessor:
    """Handles document loading and chunking using existing utilities"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if not UTILS_AVAILABLE:
            raise ImportError("utils module is required for document processing")
            
        # Validate chunk parameters
        self.chunk_size, self.chunk_overlap = ValidationUtils.validate_chunk_parameters(
            chunk_size, chunk_overlap
        )
        logger.info(f"DocumentProcessor initialized with chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")

    def validate_document_file(self, document_name: str) -> None:
        """
        Validate document file before processing

        Args:
            document_name: Name of the document to validate

        Raises:
            FileValidationError: If file validation fails
        """
        try:
            # Basic name validation
            ValidationUtils.validate_document_name(document_name)

            # Note: We don't validate file existence here because load_pdf_chunk
            # handles file discovery across multiple directories
            logger.info(f"Document name validation passed for: {document_name}")

        except ValidationError as e:
            raise FileValidationError(f"File validation failed: {str(e)}")
        except Exception as e:
            raise FileValidationError(f"Unexpected error during file validation: {str(e)}")

    def process_document(self, document_name: str, document_id: str) -> List['Document']:
        """
        Process a document using existing load_pdf_chunk function and add metadata

        Args:
            document_name: Name of the document file (e.g., "apple_2020_10k")
                          load_pdf_chunk will automatically search for the right documents in the right directories
            document_id: Unique identifier for the document (e.g., "APPLE_2020")

        Returns:
            List of Document objects with metadata

        Raises:
            FileValidationError: If file validation fails
            DocumentProcessingError: If document processing fails
        """
        try:
            logger.info(f"Processing document: {document_name} with ID: {document_id}")

            # Validate inputs
            self.validate_document_file(document_name)
            ValidationUtils.validate_document_id(document_id)

            # Use existing load_pdf_chunk function with marker method for best results
            # load_pdf_chunk will automatically search for the document in appropriate directories
            try:
                chunks, token_count = load_pdf_chunk(
                    pdf_file=document_name,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    method="marker"
                )
            except FileNotFoundError as e:
                raise DocumentProcessingError(
                    f"Document '{document_name}' not found. Please ensure the file exists in the expected directories. "
                    f"Original error: {str(e)}"
                )
            except Exception as e:
                # Handle various processing errors from load_pdf_chunk
                error_msg = str(e).lower()
                if "permission" in error_msg or "access" in error_msg:
                    raise DocumentProcessingError(
                        f"Permission denied accessing document '{document_name}'. "
                        f"Please check file permissions. Original error: {str(e)}"
                    )
                elif "corrupt" in error_msg or "invalid" in error_msg or "format" in error_msg:
                    raise DocumentProcessingError(
                        f"Document '{document_name}' appears to be corrupted or in an unsupported format. "
                        f"Please verify the file integrity. Original error: {str(e)}"
                    )
                else:
                    raise DocumentProcessingError(
                        f"Failed to load document '{document_name}': {str(e)}"
                    )

            # Validate processing results
            if not chunks:
                raise DocumentProcessingError(
                    f"No content could be extracted from document '{document_name}'. "
                    "The document may be empty, corrupted, or in an unsupported format."
                )

            if token_count == 0:
                logger.warning(f"Document '{document_name}' processed but no tokens counted")

            logger.info(f"Document processed: {len(chunks)} chunks, {token_count} tokens")

            # Add metadata to each chunk
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                # Validate chunk content
                if not chunk.page_content or not chunk.page_content.strip():
                    logger.warning(f"Empty chunk {i} in document {document_name}, skipping")
                    continue

                # Create new metadata that includes our document_id
                enhanced_metadata = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "source_name": document_name,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk.page_content),
                    **chunk.metadata  # Preserve existing metadata
                }

                # Create new Document with enhanced metadata
                if LANGCHAIN_AVAILABLE:
                    enhanced_chunk = Document(
                        page_content=chunk.page_content,
                        metadata=enhanced_metadata
                    )
                    processed_chunks.append(enhanced_chunk)
                else:
                    # Fallback if langchain not available
                    processed_chunks.append(chunk)

            # Final validation
            if not processed_chunks:
                raise DocumentProcessingError(
                    f"No valid chunks could be created from document '{document_name}'. "
                    "All chunks were empty or invalid."
                )

            logger.info(f"Added metadata to {len(processed_chunks)} chunks for document {document_id}")
            return processed_chunks

        except (FileValidationError, ValidationError):
            # Re-raise validation errors as-is
            raise
        except DocumentProcessingError:
            # Re-raise document processing errors as-is
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing document {document_name}: {str(e)}")
            raise DocumentProcessingError(f"Unexpected error processing document {document_name}: {str(e)}")


class QueryProcessor:
    """Handles answer generation using existing GPT wrapper with structured prompts"""

    def __init__(self, llm_model: str = "gpt-4o-mini", temperature: float = 0.1):
        if not UTILS_AVAILABLE:
            raise ImportError("utils module is required for query processing")
            
        self.llm = GPT(
            model_name=llm_model,
            temperature=temperature,
            max_tokens=1500,
            provider="openrouter"
        )
        logger.info(f"QueryProcessor initialized with model: {llm_model}")

    def generate_answer(self, question: str, context_chunks: List['Document']) -> str:
        """
        Generate answer from retrieved context chunks using structured prompt

        Args:
            question: The user's question
            context_chunks: List of relevant document chunks

        Returns:
            Generated answer with source attribution
        """
        if not context_chunks:
            return "I couldn't find any relevant information to answer your question."

        try:
            # Format context from chunks
            context = self._format_context(context_chunks)

            # Create structured prompt similar to finMapReduce/prompts
            prompt_template = load_prompt("prompts/rag_prompt.yml")

            # Generate answer using the GPT wrapper
            response = self.llm(prompt_template, context=context, question=question)

            # Extract and process the structured response
            answer = self._process_response(response, context_chunks)

            logger.info(f"Generated answer for question: {question}...")
            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"I encountered an error while generating the answer: {str(e)}"

    def _create_answer_prompt(self):
        """
        Load the structured prompt template from YAML file

        Returns:
            PromptTemplate for answer generation
        """
        return load_prompt("prompts/rag_prompt.yml")

    def _format_context(self, chunks: List['Document']) -> str:
        """
        Format retrieved chunks into a structured context string

        Args:
            chunks: List of Document objects with content and metadata

        Returns:
            Formatted context string with metadata
        """
        if not chunks:
            return ""

        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            # Extract metadata
            doc_id = chunk.metadata.get("document_id", "Unknown")
            chunk_index = chunk.metadata.get("chunk_index", "Unknown")
            source_name = chunk.metadata.get("source_name", chunk.metadata.get("source_path", "Unknown"))

            # Format each chunk with clear structure
            context_part = f"""[Chunk {i}]
Document ID: {doc_id}
Chunk Index: {chunk_index}
Source: {source_name}
Content:
{chunk.page_content.strip()}

---"""
            context_parts.append(context_part)

        return "\n".join(context_parts)

    def _process_response(self, response: Dict[str, Any], context_chunks: List['Document']) -> str:
        """
        Process the structured JSON response from the LLM

        Args:
            response: Response from GPT wrapper
            context_chunks: Original context chunks for fallback

        Returns:
            Formatted answer with source attribution
        """
        try:
            # Extract JSON response
            if isinstance(response, dict) and 'json' in response:
                json_response = response['json']
            elif isinstance(response, dict) and 'raw_response' in response:
                # Fallback to raw response if JSON parsing failed
                return self._fallback_response(response['raw_response'].content, context_chunks)
            else:
                return self._fallback_response(str(response), context_chunks)

            # Extract components from structured response
            reasoning = json_response.get("reasoning", "")
            evidence = json_response.get("evidence", [])
            answer = json_response.get("answer", "")
            confidence = json_response.get("confidence", "medium")
            sources = json_response.get("sources", [])

            # Format the final response
            formatted_answer = self._format_final_answer(
                answer, reasoning, evidence, confidence, sources
            )

            return formatted_answer

        except Exception as e:
            logger.warning(f"Error processing structured response: {str(e)}")
            # Fallback to simple response processing
            raw_content = response.get('raw_response', {}).content if isinstance(response, dict) else str(response)
            return self._fallback_response(raw_content, context_chunks)

    def _format_final_answer(self, answer: str, reasoning: str, evidence: List[str],
                           confidence: str, sources: List[str]) -> str:
        """
        Format the final answer with all components

        Args:
            answer: Main answer text
            reasoning: Reasoning process
            evidence: Supporting evidence
            confidence: Confidence level
            sources: Source document IDs

        Returns:
            Formatted final answer
        """
        formatted_parts = []

        # Main answer
        if answer:
            formatted_parts.append(answer)

        # Add confidence indicator
        if confidence:
            formatted_parts.append(f"\n**Confidence:** {confidence.title()}")

        # Add sources if available
        if sources:
            sources_text = ", ".join(sources)
            formatted_parts.append(f"**Sources:** {sources_text}")

        # Add evidence if available and not too verbose (limit to first 3 items)
        if evidence:
            limited_evidence = evidence[:3]  # Take only first 3 pieces of evidence
            evidence_text = "\n".join([f"- {ev[:200]}..." if len(ev) > 200 else f"- {ev}" for ev in limited_evidence])
            formatted_parts.append(f"**Supporting Evidence:**\n{evidence_text}")

        return "\n\n".join(formatted_parts)

    def _fallback_response(self, raw_content: str, context_chunks: List['Document']) -> str:
        """
        Fallback response processing when structured parsing fails

        Args:
            raw_content: Raw response content
            context_chunks: Original context chunks

        Returns:
            Formatted response with basic source attribution
        """
        # Extract unique document IDs for source attribution
        document_ids = set()
        for chunk in context_chunks:
            doc_id = chunk.metadata.get("document_id")
            if doc_id:
                document_ids.add(doc_id)

        # Add basic source attribution
        if document_ids:
            sources_text = ", ".join(sorted(document_ids))
            return f"{raw_content}\n\n**Sources:** {sources_text}"

        return raw_content


class SimpleRAG:
    """Main orchestrator class for the RAG system"""

    def __init__(self,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 llm_model: str = "gpt-4o-mini",
                 use_local_embeddings: bool = True):
        """
        Initialize the SimpleRAG system

        Args:
            embedding_model: Embedding model to use (local or OpenAI)
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            llm_model: LLM model for answer generation
            use_local_embeddings: Whether to use local embeddings
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("langchain is required for SimpleRAG system")
        if not UTILS_AVAILABLE:
            raise ImportError("utils module is required for SimpleRAG system")
            
        self.document_processor = DocumentProcessor(chunk_size, chunk_overlap)
        self.vector_store = VectorStore(embedding_model, use_local_embeddings)
        self.query_processor = QueryProcessor(llm_model)
        self.document_ids = set()

        logger.info("SimpleRAG system initialized successfully")

    def ingest_document(self, document_name: str, document_id: str,
                       overwrite: bool = False,
                       on_duplicate: str = "error") -> bool:
        """
        Ingest a document with associated metadata

        Args:
            document_name: Name of the document file (e.g., "apple_2020_10k")
                          load_pdf_chunk will automatically search for the right documents in the right directories
            document_id: Unique identifier for the document
            overwrite: Whether to overwrite existing document with same ID (deprecated, use on_duplicate)
            on_duplicate: How to handle duplicate document IDs:
                         - "error": Raise DuplicateDocumentError (default)
                         - "skip": Skip ingestion and return False
                         - "overwrite": Replace existing document

        Returns:
            True if successful, False if skipped

        Raises:
            ValidationError: If inputs are invalid
            FileValidationError: If file validation fails
            DocumentProcessingError: If document processing fails
            DuplicateDocumentError: If document ID already exists and on_duplicate="error"
        """
        try:
            logger.info(f"Ingesting document: {document_name} with ID: {document_id}")

            # Validate inputs
            ValidationUtils.validate_document_name(document_name)
            ValidationUtils.validate_document_id(document_id)

            # Validate on_duplicate parameter
            valid_duplicate_options = {"error", "skip", "overwrite"}
            if on_duplicate not in valid_duplicate_options:
                raise ValidationError(
                    f"Invalid on_duplicate option '{on_duplicate}'. "
                    f"Must be one of: {', '.join(valid_duplicate_options)}"
                )

            # Handle deprecated overwrite parameter
            if overwrite and on_duplicate == "error":
                logger.warning("The 'overwrite' parameter is deprecated. Use on_duplicate='overwrite' instead.")
                on_duplicate = "overwrite"

            # Check for duplicate document ID
            if document_id in self.document_ids:
                if on_duplicate == "error":
                    available_ids = sorted(list(self.document_ids))
                    raise DuplicateDocumentError(
                        document_id,
                        f"Document ID '{document_id}' already exists. "
                        f"Available options: use on_duplicate='skip' or on_duplicate='overwrite'. "
                        f"Existing document IDs: {available_ids}"
                    )
                elif on_duplicate == "skip":
                    logger.info(f"Skipping ingestion of document ID '{document_id}' (already exists)")
                    return False
                elif on_duplicate == "overwrite":
                    logger.info(f"Overwriting existing document ID '{document_id}'")

            # Process document using DocumentProcessor
            try:
                chunks = self.document_processor.process_document(document_name, document_id)
            except (FileValidationError, ValidationError):
                # Re-raise validation errors as-is
                raise
            except DocumentProcessingError as e:
                # Enhance error message with context
                raise DocumentProcessingError(
                    f"Failed to process document '{document_name}' with ID '{document_id}': {str(e)}"
                )

            # Validate processing results
            if not chunks:
                raise DocumentProcessingError(
                    f"No chunks generated for document '{document_name}'. "
                    "The document may be empty or could not be processed."
                )

            # Add chunks to vector store
            try:
                self.vector_store.add_documents(chunks)
            except Exception as e:
                raise DocumentProcessingError(
                    f"Failed to add document '{document_name}' to vector store: {str(e)}"
                )

            # Track document ID
            self.document_ids.add(document_id)

            logger.info(f"Successfully ingested document '{document_name}' with ID '{document_id}' ({len(chunks)} chunks)")
            return True

        except (ValidationError, FileValidationError, DocumentProcessingError, DuplicateDocumentError):
            # Re-raise known errors as-is
            raise
        except Exception as e:
            logger.error(f"Unexpected error ingesting document {document_name}: {str(e)}")
            raise DocumentProcessingError(f"Unexpected error ingesting document '{document_name}': {str(e)}")

    def query(self, question: str, document_id: Optional[str] = None,
              top_k: int = 5) -> QueryResponse:
        """
        Query the system with optional document filtering

        Args:
            question: The question to ask
            document_id: Optional document ID to filter results
            top_k: Number of top results to retrieve

        Returns:
            QueryResponse with answer and metadata

        Raises:
            ValidationError: If inputs are invalid
            QueryError: If query processing fails
        """
        start_time = time.time()

        try:
            logger.info(f"Processing query: {question[:50] if question else 'None'}... (document_id: {document_id}, top_k: {top_k})")

            # Comprehensive input validation
            clean_question = ValidationUtils.validate_query(question)
            validated_top_k = ValidationUtils.validate_top_k(top_k)

            # Validate document_id if provided
            if document_id is not None:
                ValidationUtils.validate_document_id(document_id)

                if document_id not in self.document_ids:
                    available_ids = sorted(list(self.document_ids))
                    if not available_ids:
                        raise QueryError(
                            f"Document ID '{document_id}' not found. No documents have been ingested yet. "
                            "Please ingest documents before querying."
                        )
                    else:
                        raise QueryError(
                            f"Document ID '{document_id}' not found. "
                            f"Available document IDs: {available_ids}"
                        )

            # Check if any documents have been ingested
            if len(self.document_ids) == 0:
                raise QueryError(
                    "No documents have been ingested yet. Please ingest at least one document before querying."
                )

            # Perform similarity search
            try:
                search_results = self.vector_store.similarity_search(
                    query=clean_question,
                    document_id=document_id,
                    top_k=validated_top_k
                )
            except Exception as e:
                raise QueryError(f"Vector search failed: {str(e)}")

            # Handle no results case
            if not search_results:
                query_time = time.time() - start_time
                no_results_message = self._generate_no_results_message(document_id)
                return QueryResponse(
                    answer=no_results_message,
                    source_chunks=[],
                    document_ids=[],
                    confidence_scores=[],
                    query_time=query_time
                )

            # Extract and validate chunks and scores
            source_chunks = []
            confidence_scores = []

            for doc, score in search_results:
                if doc is None:
                    logger.warning("Received None document in search results, skipping")
                    continue

                if not hasattr(doc, 'page_content'):
                    logger.warning("Received document without page_content attribute, skipping")
                    continue

                if not doc.page_content or not doc.page_content.strip():
                    logger.warning("Received document with empty content, skipping")
                    continue

                source_chunks.append(doc)
                confidence_scores.append(float(score) if score is not None else 0.0)

            # Final validation of results
            if not source_chunks:
                query_time = time.time() - start_time
                return QueryResponse(
                    answer="I found some results but they contained no valid content to answer your question.",
                    source_chunks=[],
                    document_ids=[],
                    confidence_scores=[],
                    query_time=query_time
                )

            # Extract unique document IDs from results
            result_document_ids = []
            for chunk in source_chunks:
                doc_id = chunk.metadata.get("document_id", "Unknown")
                if doc_id not in result_document_ids:
                    result_document_ids.append(doc_id)

            # Generate answer using QueryProcessor
            try:
                answer = self.query_processor.generate_answer(clean_question, source_chunks)
            except Exception as e:
                logger.error(f"Answer generation failed: {str(e)}")
                # Provide fallback response
                answer = (
                    f"I found relevant information but encountered an error generating the answer: {str(e)}. "
                    f"Found {len(source_chunks)} relevant chunks from documents: {', '.join(result_document_ids)}"
                )

            query_time = time.time() - start_time

            logger.info(f"Query completed in {query_time:.2f}s, found {len(source_chunks)} relevant chunks")

            return QueryResponse(
                answer=answer,
                source_chunks=source_chunks,
                document_ids=result_document_ids,
                confidence_scores=confidence_scores,
                query_time=query_time
            )

        except (ValidationError, QueryError):
            # Re-raise known errors as-is
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing query: {str(e)}")
            raise QueryError(f"Unexpected error processing query: {str(e)}")

    def _generate_no_results_message(self, document_id: Optional[str]) -> str:
        """
        Generate appropriate message when no results are found

        Args:
            document_id: Document ID that was searched (if any)

        Returns:
            Appropriate no-results message
        """
        if document_id:
            return (
                f"I couldn't find any relevant information in document '{document_id}' to answer your question. "
                f"You might try rephrasing your question or searching across all documents."
            )
        else:
            available_docs = sorted(list(self.document_ids))
            return (
                f"I couldn't find any relevant information to answer your question across all {len(available_docs)} documents. "
                f"You might try rephrasing your question or searching within a specific document. "
                f"Available documents: {', '.join(available_docs[:5])}{'...' if len(available_docs) > 5 else ''}"
            )

    def get_document_ids(self) -> List[str]:
        """Get list of all ingested document IDs"""
        return sorted(list(self.document_ids))

    def check_document_exists(self, document_id: str) -> bool:
        """
        Check if a document ID already exists

        Args:
            document_id: Document ID to check

        Returns:
            True if document exists, False otherwise
        """
        try:
            ValidationUtils.validate_document_id(document_id)
            return document_id in self.document_ids
        except ValidationError:
            return False

    def get_duplicate_handling_options(self, document_id: str) -> Dict[str, str]:
        """
        Get available options for handling duplicate document IDs

        Args:
            document_id: Document ID that already exists

        Returns:
            Dictionary of option codes and descriptions
        """
        if document_id not in self.document_ids:
            return {}

        return {
            "skip": f"Skip ingestion and keep existing document '{document_id}'",
            "overwrite": f"Replace existing document '{document_id}' with new content",
            "error": f"Raise an error (default behavior)"
        }

    def get_status(self) -> Dict[str, Any]:
        """Get system status information"""
        vector_stats = self.vector_store.get_stats()

        return {
            "total_documents": len(self.document_ids),
            "document_ids": sorted(list(self.document_ids)),
            "embedding_model": self.vector_store.embedding_model,
            "use_local_embeddings": self.vector_store.use_local,
            "chunk_size": self.document_processor.chunk_size,
            "chunk_overlap": self.document_processor.chunk_overlap,
            "vector_store_documents": vector_stats.get("total_documents", 0),
            "system_ready": len(self.document_ids) > 0
        }

    def clear_documents(self) -> None:
        """Clear all ingested documents and reset the system"""
        self.document_ids.clear()
        self.vector_store.clear()
        logger.info("All documents cleared from the system")

    def save_index(self, folder_path: str) -> None:
        """Save the vector index to disk for persistence"""
        try:
            os.makedirs(folder_path, exist_ok=True)

            # Save vector store
            self.vector_store.save_local(folder_path)

            # Save document IDs
            ids_file = os.path.join(folder_path, "document_ids.pkl")
            with open(ids_file, 'wb') as f:
                pickle.dump(self.document_ids, f)

            logger.info(f"Index and metadata saved to {folder_path}")

        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise SimpleRAGError(f"Failed to save index: {str(e)}")

    def load_index(self, folder_path: str) -> None:
        """Load the vector index from disk"""
        try:
            # Load vector store
            self.vector_store.load_local(folder_path)

            # Load document IDs
            ids_file = os.path.join(folder_path, "document_ids.pkl")
            if os.path.exists(ids_file):
                with open(ids_file, 'rb') as f:
                    self.document_ids = pickle.load(f)
            else:
                logger.warning("No document_ids.pkl found, document IDs may be incomplete")
                self.document_ids = set()

            logger.info(f"Index and metadata loaded from {folder_path}")

        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            raise SimpleRAGError(f"Failed to load index: {str(e)}")


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
            
        # Test DocumentProcessor initialization if available
        if UTILS_AVAILABLE:
            print("Testing DocumentProcessor initialization...")
            try:
                doc_processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
                print("DocumentProcessor working correctly!")
            except Exception as dp_error:
                print(f"DocumentProcessor test failed: {dp_error}")
        else:
            print("utils module not available, skipping document processor test")
            
        # Test QueryProcessor initialization if available
        if UTILS_AVAILABLE and LANGCHAIN_AVAILABLE:
            print("Testing QueryProcessor initialization...")
            try:
                query_processor = QueryProcessor()
                print("QueryProcessor working correctly!")
            except Exception as qp_error:
                print(f"QueryProcessor test failed: {qp_error}")
        else:
            print("Required dependencies not available, skipping query processor test")
            
        # Test complete SimpleRAG system if available
        if UTILS_AVAILABLE and LANGCHAIN_AVAILABLE:
            print("Testing complete SimpleRAG system...")
            try:
                rag = SimpleRAG()
                status = rag.get_status()
                print(f"SimpleRAG system status: {status}")
                print("SimpleRAG system initialized successfully!")
            except Exception as rag_error:
                print(f"SimpleRAG system test failed: {rag_error}")
        else:
            print("Required dependencies not available, skipping SimpleRAG system test")
            
    except Exception as e:
        print(f"Error testing utilities: {e}")