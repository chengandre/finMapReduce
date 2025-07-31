"""
Comprehensive tests for error handling and validation in SimpleRAG system

Tests cover:
- File validation for document ingestion
- Input validation for queries and parameters
- Duplicate document ID handling
- Error scenarios and edge cases
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document

from simple_rag import (
    SimpleRAG,
    DocumentProcessor,
    VectorStore,
    QueryProcessor,
    ValidationUtils,
    SimpleRAGError,
    DocumentProcessingError,
    QueryError,
    ValidationError,
    FileValidationError,
    DuplicateDocumentError,
    QueryResponse
)


class TestValidationUtils:
    """Test suite for ValidationUtils class"""
    
    def test_validate_document_name_valid(self):
        """Test valid document names"""
        valid_names = [
            "apple_2020_10k",
            "document.pdf",
            "My Document 2023",
            "test_file_123",
            "report-final.docx"
        ]
        
        for name in valid_names:
            # Should not raise any exception
            ValidationUtils.validate_document_name(name)
    
    def test_validate_document_name_invalid(self):
        """Test invalid document names"""
        invalid_cases = [
            ("", "Document name cannot be empty"),
            ("   ", "Document name cannot be empty"),
            ("file<name", "contains invalid characters"),
            ("file>name", "contains invalid characters"),
            ("file:name", "contains invalid characters"),
            ("file\"name", "contains invalid characters"),
            ("file|name", "contains invalid characters"),
            ("file?name", "contains invalid characters"),
            ("file*name", "contains invalid characters"),
            ("a" * 256, "too long")
        ]
        
        for invalid_name, expected_error in invalid_cases:
            with pytest.raises(ValidationError, match=expected_error):
                ValidationUtils.validate_document_name(invalid_name)
    
    def test_validate_document_id_valid(self):
        """Test valid document IDs"""
        valid_ids = [
            "APPLE_2020",
            "test123",
            "DOC-1",
            "my_document_id",
            "A1B2C3",
            "report_final_2023"
        ]
        
        for doc_id in valid_ids:
            # Should not raise any exception
            ValidationUtils.validate_document_id(doc_id)
    
    def test_validate_document_id_invalid(self):
        """Test invalid document IDs"""
        invalid_cases = [
            ("", "Document ID cannot be empty"),
            ("   ", "Document ID cannot be empty"),
            ("doc id", "contains invalid characters"),
            ("doc@id", "contains invalid characters"),
            ("doc.id", "contains invalid characters"),
            ("doc/id", "contains invalid characters"),
            ("doc\\id", "contains invalid characters"),
            ("a" * 101, "too long")
        ]
        
        for invalid_id, expected_error in invalid_cases:
            with pytest.raises(ValidationError, match=expected_error):
                ValidationUtils.validate_document_id(invalid_id)
    
    def test_validate_query_valid(self):
        """Test valid queries"""
        valid_queries = [
            "What is the revenue?",
            "How much did Apple make in 2020?",
            "Tell me about the financial performance",
            "A" * 1000,  # Long but valid query
            "Query with\nnewlines\nare ok"
        ]
        
        for query in valid_queries:
            result = ValidationUtils.validate_query(query)
            assert result == query.strip()
    
    def test_validate_query_invalid(self):
        """Test invalid queries"""
        invalid_cases = [
            (None, "Query cannot be None"),
            ("", "Query cannot be empty"),
            ("   ", "Query cannot be empty"),
            ("A" * 10001, "Query too long."),
            ("Query\n" * 52, "Query contains too many line breaks")
        ]
        
        for invalid_query, expected_error in invalid_cases:
            with pytest.raises(ValidationError, match=expected_error):
                ValidationUtils.validate_query(invalid_query)
    
    def test_validate_top_k_valid(self):
        """Test valid top_k values"""
        valid_values = [1, 5, 10, 50, 100]
        
        for value in valid_values:
            result = ValidationUtils.validate_top_k(value)
            assert result == value
    
    def test_validate_top_k_string_conversion(self):
        """Test top_k string to int conversion"""
        assert ValidationUtils.validate_top_k("5") == 5
        assert ValidationUtils.validate_top_k("10") == 10
    
    def test_validate_top_k_invalid(self):
        """Test invalid top_k values"""
        invalid_cases = [
            (0, "must be at least 1"),
            (-1, "must be at least 1"),
            (101, "too large"),
            ("invalid", "must be an integer"),
            (None, "must be an integer"),
            (5.5, "must be an integer")
        ]
        
        for invalid_value, expected_error in invalid_cases:
            with pytest.raises(ValidationError, match=expected_error):
                ValidationUtils.validate_top_k(invalid_value)
    
    def test_validate_chunk_parameters_valid(self):
        """Test valid chunk parameters"""
        valid_cases = [
            (1000, 200),
            (500, 100),
            (2000, 0),
            (10000, 1000)
        ]
        
        for chunk_size, chunk_overlap in valid_cases:
            result_size, result_overlap = ValidationUtils.validate_chunk_parameters(chunk_size, chunk_overlap)
            assert result_size == chunk_size
            assert result_overlap == chunk_overlap
    
    def test_validate_chunk_parameters_invalid(self):
        """Test invalid chunk parameters"""
        invalid_cases = [
            (99, 50, "chunk_size too small"),
            (10001, 100, "chunk_size too large"),
            (1000, -1, "chunk_overlap cannot be negative"),
            (1000, 1000, "chunk_overlap.*must be less than chunk_size"),
            (1000, 1001, "chunk_overlap.*must be less than chunk_size"),
            ("invalid", 100, "chunk_size must be an integer"),
            (1000, "invalid", "chunk_overlap must be an integer")
        ]
        
        for chunk_size, chunk_overlap, expected_error in invalid_cases:
            with pytest.raises(ValidationError, match=expected_error):
                ValidationUtils.validate_chunk_parameters(chunk_size, chunk_overlap)


class TestDocumentProcessorErrorHandling:
    """Test error handling in DocumentProcessor"""
    
    @pytest.fixture
    def mock_load_pdf_chunk(self):
        """Mock the load_pdf_chunk function"""
        with patch('simple_rag.load_pdf_chunk') as mock:
            yield mock
    
    def test_initialization_invalid_parameters(self):
        """Test DocumentProcessor initialization with invalid parameters"""
        invalid_cases = [
            (99, 50),  # chunk_size too small
            (1000, 1000),  # overlap >= chunk_size
            ("invalid", 100),  # non-integer chunk_size
            (1000, -1)  # negative overlap
        ]
        
        for chunk_size, chunk_overlap in invalid_cases:
            with pytest.raises(ValidationError):
                DocumentProcessor(chunk_size, chunk_overlap)
    
    def test_validate_document_file_invalid_names(self):
        """Test file validation with invalid document names"""
        processor = DocumentProcessor()
        
        invalid_names = [
            "",
            "   ",
            "file<name",
            "file>name",
            "a" * 256
        ]
        
        for invalid_name in invalid_names:
            with pytest.raises(FileValidationError):
                processor.validate_document_file(invalid_name)
    
    def test_process_document_invalid_inputs(self, mock_load_pdf_chunk):
        """Test document processing with invalid inputs"""
        processor = DocumentProcessor()
        
        # Invalid document name
        with pytest.raises(FileValidationError):
            processor.process_document("", "VALID_ID")
        
        # Invalid document ID
        with pytest.raises(ValidationError):
            processor.process_document("valid_doc", "")
    
    def test_process_document_file_not_found(self, mock_load_pdf_chunk):
        """Test document processing when file is not found"""
        processor = DocumentProcessor()
        mock_load_pdf_chunk.side_effect = FileNotFoundError("File not found")
        
        with pytest.raises(DocumentProcessingError, match="not found"):
            processor.process_document("nonexistent.pdf", "TEST_DOC")
    
    def test_process_document_permission_error(self, mock_load_pdf_chunk):
        """Test document processing with permission error"""
        processor = DocumentProcessor()
        mock_load_pdf_chunk.side_effect = PermissionError("Permission denied")
        
        with pytest.raises(DocumentProcessingError, match="Permission denied"):
            processor.process_document("restricted.pdf", "TEST_DOC")
    
    def test_process_document_corrupted_file(self, mock_load_pdf_chunk):
        """Test document processing with corrupted file"""
        processor = DocumentProcessor()
        mock_load_pdf_chunk.side_effect = Exception("File appears to be corrupted")
        
        with pytest.raises(DocumentProcessingError, match="corrupted"):
            processor.process_document("corrupted.pdf", "TEST_DOC")
    
    def test_process_document_no_chunks(self, mock_load_pdf_chunk):
        """Test document processing when no chunks are generated"""
        processor = DocumentProcessor()
        mock_load_pdf_chunk.return_value = ([], 0)
        
        with pytest.raises(DocumentProcessingError, match="No content could be extracted"):
            processor.process_document("empty.pdf", "TEST_DOC")
    
    def test_process_document_empty_chunks(self, mock_load_pdf_chunk):
        """Test document processing with empty chunks"""
        processor = DocumentProcessor()
        empty_chunks = [
            Document(page_content="", metadata={}),
            Document(page_content="   ", metadata={}),
            # Note: Cannot create Document with None content due to pydantic validation
            # This simulates what would happen with empty/whitespace-only chunks
        ]
        mock_load_pdf_chunk.return_value = (empty_chunks, 0)
        
        with pytest.raises(DocumentProcessingError, match="No valid chunks could be created"):
            processor.process_document("empty_chunks.pdf", "TEST_DOC")


class TestSimpleRAGErrorHandling:
    """Test error handling in SimpleRAG main class"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock all dependencies for isolated testing"""
        with patch('simple_rag.DocumentProcessor') as mock_doc_proc, \
             patch('simple_rag.VectorStore') as mock_vector_store, \
             patch('simple_rag.QueryProcessor') as mock_query_proc:
            
            # Setup mock instances
            mock_doc_proc_instance = Mock()
            mock_vector_store_instance = Mock()
            mock_query_proc_instance = Mock()
            
            mock_doc_proc.return_value = mock_doc_proc_instance
            mock_vector_store.return_value = mock_vector_store_instance
            mock_query_proc.return_value = mock_query_proc_instance
            
            # Configure mock vector store
            mock_vector_store_instance.embedding_model = "test-model"
            mock_vector_store_instance.use_local = True
            mock_vector_store_instance.get_stats.return_value = {"total_documents": 0}
            
            # Configure mock document processor
            mock_doc_proc_instance.chunk_size = 1000
            mock_doc_proc_instance.chunk_overlap = 200
            
            yield {
                'doc_processor': mock_doc_proc_instance,
                'vector_store': mock_vector_store_instance,
                'query_processor': mock_query_proc_instance
            }
    
    @pytest.fixture
    def simple_rag(self, mock_dependencies):
        """Create SimpleRAG instance with mocked dependencies"""
        mock_llm = Mock()
        return SimpleRAG(llm=mock_llm)
    
    def test_ingest_document_invalid_inputs(self, simple_rag):
        """Test document ingestion with invalid inputs"""
        # Invalid document name
        with pytest.raises(ValidationError, match="Document name cannot be empty"):
            simple_rag.ingest_document("", "VALID_ID")
        
        # Invalid document ID
        with pytest.raises(ValidationError, match="Document ID cannot be empty"):
            simple_rag.ingest_document("valid_doc", "")
        
        # Invalid on_duplicate option
        with pytest.raises(ValidationError, match="Invalid on_duplicate option"):
            simple_rag.ingest_document("valid_doc", "VALID_ID", on_duplicate="invalid")
    
    def test_ingest_document_duplicate_error(self, simple_rag, mock_dependencies):
        """Test duplicate document ID with error handling"""
        # Add existing document ID
        simple_rag.document_ids.add("EXISTING_DOC")
        
        # Test with default error behavior
        with pytest.raises(DuplicateDocumentError) as exc_info:
            simple_rag.ingest_document("test_doc.pdf", "EXISTING_DOC")
        
        assert "EXISTING_DOC" in str(exc_info.value)
        assert "already exists" in str(exc_info.value)
        assert exc_info.value.document_id == "EXISTING_DOC"
    
    def test_ingest_document_duplicate_skip(self, simple_rag, mock_dependencies):
        """Test duplicate document ID with skip option"""
        # Add existing document ID
        simple_rag.document_ids.add("EXISTING_DOC")
        
        # Test skip behavior
        result = simple_rag.ingest_document("test_doc.pdf", "EXISTING_DOC", on_duplicate="skip")
        
        assert result is False
        mock_dependencies['doc_processor'].process_document.assert_not_called()
    
    def test_ingest_document_duplicate_overwrite(self, simple_rag, mock_dependencies):
        """Test duplicate document ID with overwrite option"""
        # Add existing document ID
        simple_rag.document_ids.add("EXISTING_DOC")
        
        # Setup mock responses
        mock_chunks = [Document(page_content="New content", metadata={"document_id": "EXISTING_DOC"})]
        mock_dependencies['doc_processor'].process_document.return_value = mock_chunks
        
        # Test overwrite behavior
        result = simple_rag.ingest_document("test_doc.pdf", "EXISTING_DOC", on_duplicate="overwrite")
        
        assert result is True
        mock_dependencies['doc_processor'].process_document.assert_called_once()
        mock_dependencies['vector_store'].add_documents.assert_called_once_with(mock_chunks)
    
    def test_ingest_document_processing_error(self, simple_rag, mock_dependencies):
        """Test document ingestion with processing error"""
        mock_dependencies['doc_processor'].process_document.side_effect = DocumentProcessingError("Processing failed")
        
        with pytest.raises(DocumentProcessingError, match="Processing failed"):
            simple_rag.ingest_document("bad_doc.pdf", "BAD_DOC")
    
    def test_ingest_document_vector_store_error(self, simple_rag, mock_dependencies):
        """Test document ingestion with vector store error"""
        mock_chunks = [Document(page_content="Test content", metadata={})]
        mock_dependencies['doc_processor'].process_document.return_value = mock_chunks
        mock_dependencies['vector_store'].add_documents.side_effect = Exception("Vector store error")
        
        with pytest.raises(DocumentProcessingError, match="Failed to add document.*to vector store"):
            simple_rag.ingest_document("test_doc.pdf", "TEST_DOC")
    
    def test_query_invalid_inputs(self, simple_rag):
        """Test query with invalid inputs"""
        # Add a document to make system ready
        simple_rag.document_ids.add("TEST_DOC")
        
        # Empty query
        with pytest.raises(ValidationError, match="Query cannot be empty"):
            simple_rag.query("")
        
        # None query
        with pytest.raises(ValidationError, match="Query cannot be None"):
            simple_rag.query(None)
        
        # Invalid top_k
        with pytest.raises(ValidationError, match="must be at least 1"):
            simple_rag.query("Valid question", top_k=0)
        
        with pytest.raises(ValidationError, match="too large"):
            simple_rag.query("Valid question", top_k=101)
    
    def test_query_invalid_document_id(self, simple_rag):
        """Test query with invalid document ID"""
        simple_rag.document_ids.add("VALID_DOC")
        
        # Invalid document ID format
        with pytest.raises(ValidationError, match="contains invalid characters"):
            simple_rag.query("Test question", document_id="invalid id")
        
        # Non-existent document ID
        with pytest.raises(QueryError, match="Document ID 'NONEXISTENT' not found"):
            simple_rag.query("Test question", document_id="NONEXISTENT")
    
    def test_query_no_documents_ingested(self, simple_rag):
        """Test query when no documents have been ingested"""
        with pytest.raises(QueryError, match="No documents have been ingested yet"):
            simple_rag.query("Test question")
    
    def test_query_no_documents_with_filter(self, simple_rag):
        """Test query with document filter when no documents exist"""
        with pytest.raises(QueryError, match="No documents have been ingested yet"):
            simple_rag.query("Test question", document_id="NONEXISTENT")
    
    def test_query_vector_search_error(self, simple_rag, mock_dependencies):
        """Test query with vector search error"""
        simple_rag.document_ids.add("TEST_DOC")
        mock_dependencies['vector_store'].similarity_search.side_effect = Exception("Search failed")
        
        with pytest.raises(QueryError, match="Vector search failed"):
            simple_rag.query("Test question")
    
    def test_query_answer_generation_error(self, simple_rag, mock_dependencies):
        """Test query with answer generation error"""
        simple_rag.document_ids.add("TEST_DOC")
        
        # Setup successful search but failed answer generation
        mock_chunks = [Document(page_content="Test content", metadata={"document_id": "TEST_DOC"})]
        mock_dependencies['vector_store'].similarity_search.return_value = [(mock_chunks[0], 0.9)]
        mock_dependencies['query_processor'].generate_answer.side_effect = Exception("Answer generation failed")
        
        # Should not raise error but provide fallback response
        response = simple_rag.query("Test question")
        
        assert "encountered an error generating the answer" in response.answer
        assert len(response.source_chunks) == 1
        assert response.document_ids == ["TEST_DOC"]
    
    def test_query_no_results_global(self, simple_rag, mock_dependencies):
        """Test query with no results found (global search)"""
        simple_rag.document_ids.update(["DOC1", "DOC2", "DOC3"])
        mock_dependencies['vector_store'].similarity_search.return_value = []
        
        response = simple_rag.query("Irrelevant question")
        
        assert "couldn't find any relevant information" in response.answer
        assert "across all 3 documents" in response.answer
        assert "DOC1, DOC2, DOC3" in response.answer
        assert len(response.source_chunks) == 0
    
    def test_query_no_results_document_specific(self, simple_rag, mock_dependencies):
        """Test query with no results found (document-specific search)"""
        simple_rag.document_ids.add("SPECIFIC_DOC")
        mock_dependencies['vector_store'].similarity_search.return_value = []
        
        response = simple_rag.query("Irrelevant question", document_id="SPECIFIC_DOC")
        
        assert "couldn't find any relevant information in document 'SPECIFIC_DOC'" in response.answer
        assert "try rephrasing" in response.answer
        assert len(response.source_chunks) == 0
    
    def test_query_invalid_search_results(self, simple_rag, mock_dependencies):
        """Test query with invalid search results"""
        simple_rag.document_ids.add("TEST_DOC")
        
        # Mock search results with None documents and empty content
        invalid_results = [
            (None, 0.9),  # None document
            (Document(page_content="", metadata={}), 0.8),  # Empty content
            # Note: Cannot create Document with None content due to pydantic validation
            # The None document case above covers the scenario where search returns invalid results
        ]
        mock_dependencies['vector_store'].similarity_search.return_value = invalid_results
        
        response = simple_rag.query("Test question")
        
        assert "found some results but they contained no valid content" in response.answer
        assert len(response.source_chunks) == 0
    
    def test_check_document_exists(self, simple_rag):
        """Test document existence checking"""
        simple_rag.document_ids.add("EXISTING_DOC")
        
        assert simple_rag.check_document_exists("EXISTING_DOC") is True
        assert simple_rag.check_document_exists("NONEXISTENT_DOC") is False
        
        # Invalid document ID should return False
        assert simple_rag.check_document_exists("invalid id") is False
    
    def test_get_duplicate_handling_options(self, simple_rag):
        """Test getting duplicate handling options"""
        simple_rag.document_ids.add("EXISTING_DOC")
        
        options = simple_rag.get_duplicate_handling_options("EXISTING_DOC")
        
        assert "skip" in options
        assert "overwrite" in options
        assert "error" in options
        assert "EXISTING_DOC" in options["skip"]
        assert "EXISTING_DOC" in options["overwrite"]
        
        # Non-existent document should return empty dict
        empty_options = simple_rag.get_duplicate_handling_options("NONEXISTENT")
        assert empty_options == {}


class TestDuplicateDocumentError:
    """Test DuplicateDocumentError exception"""
    
    def test_default_message(self):
        """Test default error message"""
        error = DuplicateDocumentError("TEST_DOC")
        assert error.document_id == "TEST_DOC"
        assert "TEST_DOC" in str(error)
        assert "already exists" in str(error)
    
    def test_custom_message(self):
        """Test custom error message"""
        custom_message = "Custom error message for TEST_DOC"
        error = DuplicateDocumentError("TEST_DOC", custom_message)
        assert error.document_id == "TEST_DOC"
        assert str(error) == custom_message


class TestErrorHandlingIntegration:
    """Integration tests for error handling across components"""
    
    def test_end_to_end_validation_flow(self):
        """Test complete validation flow from input to processing"""
        try:
            # This should work with real components if available
            from utils import GPT
            mock_llm = GPT(model_name="gpt-4o-mini", temperature=0.1, max_tokens=1500, provider="openrouter", key=None)
            rag = SimpleRAG(llm=mock_llm)
            
            # Test various validation scenarios
            with pytest.raises(ValidationError):
                rag.ingest_document("", "VALID_ID")
            
            with pytest.raises(QueryError):
                rag.query("Valid question")  # No documents ingested
            
        except ImportError:
            # Skip if dependencies not available
            pytest.skip("Required dependencies not available for integration test")
    
    def test_error_message_clarity(self):
        """Test that error messages are clear and actionable"""
        try:
            from utils import GPT
            mock_llm = GPT(model_name="gpt-4o-mini", temperature=0.1, max_tokens=1500, provider="openrouter", key=None)
            rag = SimpleRAG(llm=mock_llm)
            
            # Test that error messages provide helpful information
            try:
                rag.query("Test question")
            except QueryError as e:
                assert "No documents have been ingested yet" in str(e)
                assert "Please ingest" in str(e)
            
        except ImportError:
            pytest.skip("Required dependencies not available for integration test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])