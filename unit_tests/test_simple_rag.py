"""
Unit tests for SimpleRAG orchestrator class
"""

import pytest
import tempfile
import shutil
import os
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document

from simple_rag import (
    SimpleRAG, 
    SimpleRAGError, 
    DocumentProcessingError, 
    QueryError,
    QueryResponse
)


class TestSimpleRAG:
    """Test cases for SimpleRAG main orchestrator class"""
    
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
    
    def test_initialization(self, mock_dependencies):
        """Test SimpleRAG initialization"""
        mock_llm = Mock()
        rag = SimpleRAG(
            llm=mock_llm,
            embedding_model="custom-model",
            chunk_size=500,
            chunk_overlap=100,
            use_local_embeddings=False
        )
        
        assert isinstance(rag.document_ids, set)
        assert len(rag.document_ids) == 0
        assert rag.document_processor is not None
        assert rag.vector_store is not None
        assert rag.query_processor is not None
    
    def test_ingest_document_success(self, simple_rag, mock_dependencies):
        """Test successful document ingestion"""
        # Setup mock responses
        mock_chunks = [
            Document(page_content="Test content 1", metadata={"document_id": "TEST_DOC"}),
            Document(page_content="Test content 2", metadata={"document_id": "TEST_DOC"})
        ]
        mock_dependencies['doc_processor'].process_document.return_value = mock_chunks
        
        # Test ingestion
        result = simple_rag.ingest_document("test_doc.pdf", "TEST_DOC")
        
        assert result is True
        assert "TEST_DOC" in simple_rag.document_ids
        mock_dependencies['doc_processor'].process_document.assert_called_once_with("test_doc.pdf", "TEST_DOC")
        mock_dependencies['vector_store'].add_documents.assert_called_once_with(mock_chunks)
    
    def test_ingest_document_duplicate_id_no_overwrite(self, simple_rag, mock_dependencies):
        """Test ingestion with duplicate document ID without overwrite"""
        # Add existing document ID
        simple_rag.document_ids.add("EXISTING_DOC")
        
        # Test ingestion
        result = simple_rag.ingest_document("test_doc.pdf", "EXISTING_DOC")
        
        assert result is False
        mock_dependencies['doc_processor'].process_document.assert_not_called()
        mock_dependencies['vector_store'].add_documents.assert_not_called()
    
    def test_ingest_document_duplicate_id_with_overwrite(self, simple_rag, mock_dependencies):
        """Test ingestion with duplicate document ID with overwrite"""
        # Add existing document ID
        simple_rag.document_ids.add("EXISTING_DOC")
        
        # Setup mock responses
        mock_chunks = [Document(page_content="New content", metadata={"document_id": "EXISTING_DOC"})]
        mock_dependencies['doc_processor'].process_document.return_value = mock_chunks
        
        # Test ingestion with overwrite
        result = simple_rag.ingest_document("test_doc.pdf", "EXISTING_DOC", overwrite=True)
        
        assert result is True
        assert "EXISTING_DOC" in simple_rag.document_ids
        mock_dependencies['doc_processor'].process_document.assert_called_once_with("test_doc.pdf", "EXISTING_DOC")
        mock_dependencies['vector_store'].add_documents.assert_called_once_with(mock_chunks)
    
    def test_ingest_document_no_chunks(self, simple_rag, mock_dependencies):
        """Test ingestion when no chunks are generated"""
        # Setup mock to return empty chunks
        mock_dependencies['doc_processor'].process_document.return_value = []
        
        # Test ingestion
        result = simple_rag.ingest_document("empty_doc.pdf", "EMPTY_DOC")
        
        assert result is False
        assert "EMPTY_DOC" not in simple_rag.document_ids
        mock_dependencies['vector_store'].add_documents.assert_not_called()
    
    def test_ingest_document_processing_error(self, simple_rag, mock_dependencies):
        """Test ingestion with document processing error"""
        # Setup mock to raise error
        mock_dependencies['doc_processor'].process_document.side_effect = DocumentProcessingError("Processing failed")
        
        # Test ingestion
        with pytest.raises(DocumentProcessingError, match="Processing failed"):
            simple_rag.ingest_document("bad_doc.pdf", "BAD_DOC")
        
        assert "BAD_DOC" not in simple_rag.document_ids
    
    def test_ingest_document_unexpected_error(self, simple_rag, mock_dependencies):
        """Test ingestion with unexpected error"""
        # Setup mock to raise unexpected error
        mock_dependencies['doc_processor'].process_document.side_effect = ValueError("Unexpected error")
        
        # Test ingestion
        with pytest.raises(DocumentProcessingError, match="Failed to ingest document"):
            simple_rag.ingest_document("error_doc.pdf", "ERROR_DOC")
        
        assert "ERROR_DOC" not in simple_rag.document_ids
    
    def test_query_success(self, simple_rag, mock_dependencies):
        """Test successful query execution"""
        # Setup system state
        simple_rag.document_ids.add("TEST_DOC")
        
        # Setup mock responses
        mock_chunks = [
            Document(page_content="Relevant content", metadata={"document_id": "TEST_DOC"})
        ]
        mock_search_results = [(mock_chunks[0], 0.95)]
        mock_dependencies['vector_store'].similarity_search.return_value = mock_search_results
        mock_dependencies['query_processor'].generate_answer.return_value = "Generated answer"
        
        # Test query
        response = simple_rag.query("What is the test about?")
        
        assert isinstance(response, QueryResponse)
        assert response.answer == "Generated answer"
        assert len(response.source_chunks) == 1
        assert response.document_ids == ["TEST_DOC"]
        assert len(response.confidence_scores) == 1
        assert response.query_time > 0
        
        mock_dependencies['vector_store'].similarity_search.assert_called_once_with(
            query="What is the test about?",
            document_id=None,
            top_k=5
        )
        mock_dependencies['query_processor'].generate_answer.assert_called_once_with(
            "What is the test about?", mock_chunks
        )
    
    def test_query_with_document_id_filter(self, simple_rag, mock_dependencies):
        """Test query with document ID filtering"""
        # Setup system state
        simple_rag.document_ids.add("SPECIFIC_DOC")
        
        # Setup mock responses
        mock_chunks = [Document(page_content="Specific content", metadata={"document_id": "SPECIFIC_DOC"})]
        mock_search_results = [(mock_chunks[0], 0.90)]
        mock_dependencies['vector_store'].similarity_search.return_value = mock_search_results
        mock_dependencies['query_processor'].generate_answer.return_value = "Specific answer"
        
        # Test query with document ID
        response = simple_rag.query("What is specific?", document_id="SPECIFIC_DOC")
        
        assert response.answer == "Specific answer"
        assert response.document_ids == ["SPECIFIC_DOC"]
        
        mock_dependencies['vector_store'].similarity_search.assert_called_once_with(
            query="What is specific?",
            document_id="SPECIFIC_DOC",
            top_k=5
        )
    
    def test_query_empty_question(self, simple_rag, mock_dependencies):
        """Test query with empty question"""
        with pytest.raises(QueryError, match="Question cannot be empty"):
            simple_rag.query("")
        
        with pytest.raises(QueryError, match="Question cannot be empty"):
            simple_rag.query("   ")
    
    def test_query_invalid_document_id(self, simple_rag, mock_dependencies):
        """Test query with invalid document ID"""
        simple_rag.document_ids.add("VALID_DOC")
        
        with pytest.raises(QueryError, match="Document ID 'INVALID_DOC' not found"):
            simple_rag.query("Test question", document_id="INVALID_DOC")
    
    def test_query_no_documents_ingested(self, simple_rag, mock_dependencies):
        """Test query when no documents have been ingested"""
        with pytest.raises(QueryError, match="No documents have been ingested yet"):
            simple_rag.query("Test question")
    
    def test_query_no_results_found(self, simple_rag, mock_dependencies):
        """Test query when no relevant results are found"""
        # Setup system state
        simple_rag.document_ids.add("TEST_DOC")
        
        # Setup mock to return no results
        mock_dependencies['vector_store'].similarity_search.return_value = []
        
        # Test query
        response = simple_rag.query("Irrelevant question")
        
        assert "couldn't find any relevant information" in response.answer
        assert len(response.source_chunks) == 0
        assert len(response.document_ids) == 0
        assert len(response.confidence_scores) == 0
        assert response.query_time > 0
    
    def test_query_unexpected_error(self, simple_rag, mock_dependencies):
        """Test query with unexpected error"""
        # Setup system state
        simple_rag.document_ids.add("TEST_DOC")
        
        # Setup mock to raise unexpected error
        mock_dependencies['vector_store'].similarity_search.side_effect = ValueError("Unexpected error")
        
        # Test query
        with pytest.raises(QueryError, match="Failed to process query"):
            simple_rag.query("Test question")
    
    def test_get_document_ids(self, simple_rag):
        """Test getting document IDs"""
        # Initially empty
        assert simple_rag.get_document_ids() == []
        
        # Add some document IDs
        simple_rag.document_ids.update(["DOC1", "DOC2", "DOC3"])
        
        ids = simple_rag.get_document_ids()
        assert len(ids) == 3
        assert set(ids) == {"DOC1", "DOC2", "DOC3"}
    
    def test_get_status(self, simple_rag, mock_dependencies):
        """Test getting system status"""
        # Setup mock vector store stats
        mock_dependencies['vector_store'].get_stats.return_value = {"total_documents": 5}
        
        # Add some document IDs
        simple_rag.document_ids.update(["DOC1", "DOC2"])
        
        status = simple_rag.get_status()
        
        assert status["total_documents"] == 2
        assert set(status["document_ids"]) == {"DOC1", "DOC2"}
        assert status["embedding_model"] == "test-model"
        assert status["use_local_embeddings"] is True
        assert status["chunk_size"] == 1000
        assert status["chunk_overlap"] == 200
        assert status["vector_store_documents"] == 5
        assert status["system_ready"] is True
    
    def test_get_status_empty_system(self, simple_rag, mock_dependencies):
        """Test getting status of empty system"""
        status = simple_rag.get_status()
        
        assert status["total_documents"] == 0
        assert status["document_ids"] == []
        assert status["system_ready"] is False
    
    def test_clear_documents(self, simple_rag, mock_dependencies):
        """Test clearing all documents"""
        # Add some document IDs
        simple_rag.document_ids.update(["DOC1", "DOC2"])
        
        # Clear documents
        simple_rag.clear_documents()
        
        assert len(simple_rag.document_ids) == 0
        mock_dependencies['vector_store'].clear.assert_called_once()
    
    def test_save_index(self, simple_rag, mock_dependencies):
        """Test saving index to disk"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Add some document IDs
            simple_rag.document_ids.update(["DOC1", "DOC2"])
            
            # Test save
            simple_rag.save_index(temp_dir)
            
            # Verify vector store save was called
            mock_dependencies['vector_store'].save_local.assert_called_once_with(temp_dir)
            
            # Verify document IDs file was created
            ids_file = os.path.join(temp_dir, "document_ids.pkl")
            assert os.path.exists(ids_file)
    
    def test_save_index_error(self, simple_rag, mock_dependencies):
        """Test save index with error"""
        # Setup mock to raise error
        mock_dependencies['vector_store'].save_local.side_effect = Exception("Save failed")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(SimpleRAGError, match="Failed to save index"):
                simple_rag.save_index(temp_dir)
    
    def test_load_index(self, simple_rag, mock_dependencies):
        """Test loading index from disk"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock document IDs file
            import pickle
            test_ids = {"DOC1", "DOC2"}
            ids_file = os.path.join(temp_dir, "document_ids.pkl")
            with open(ids_file, 'wb') as f:
                pickle.dump(test_ids, f)
            
            # Test load
            simple_rag.load_index(temp_dir)
            
            # Verify vector store load was called
            mock_dependencies['vector_store'].load_local.assert_called_once_with(temp_dir)
            
            # Verify document IDs were loaded
            assert simple_rag.document_ids == test_ids
    
    def test_load_index_no_ids_file(self, simple_rag, mock_dependencies):
        """Test loading index when document IDs file doesn't exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test load without IDs file
            simple_rag.load_index(temp_dir)
            
            # Verify vector store load was called
            mock_dependencies['vector_store'].load_local.assert_called_once_with(temp_dir)
            
            # Verify document IDs were reset to empty set
            assert simple_rag.document_ids == set()
    
    def test_load_index_error(self, simple_rag, mock_dependencies):
        """Test load index with error"""
        # Setup mock to raise error
        mock_dependencies['vector_store'].load_local.side_effect = Exception("Load failed")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(SimpleRAGError, match="Failed to load index"):
                simple_rag.load_index(temp_dir)


class TestSimpleRAGIntegration:
    """Integration tests for SimpleRAG with real components"""
    
    def test_initialization_with_real_components(self):
        """Test that SimpleRAG can be initialized with real components"""
        try:
            from utils import GPT
            mock_llm = GPT(model_name="gpt-4o-mini", temperature=0.1, max_tokens=1500, provider="openrouter", key=None)
            rag = SimpleRAG(
                llm=mock_llm,
                embedding_model="all-MiniLM-L6-v2",
                use_local_embeddings=True
            )
            
            # Verify components are created
            assert rag.document_processor is not None
            assert rag.vector_store is not None
            assert rag.query_processor is not None
            assert isinstance(rag.document_ids, set)
            
            # Verify status works
            status = rag.get_status()
            assert isinstance(status, dict)
            assert "total_documents" in status
            assert "system_ready" in status
            
        except ImportError:
            # Skip if dependencies not available
            pytest.skip("Required dependencies not available for integration test")
    
    def test_error_handling_with_real_components(self):
        """Test error handling with real components"""
        try:
            from utils import GPT
            mock_llm = GPT(model_name="gpt-4o-mini", temperature=0.1, max_tokens=1500, provider="openrouter", key=None)
            rag = SimpleRAG(llm=mock_llm)
            
            # Test query without documents
            with pytest.raises(QueryError):
                rag.query("Test question")
            
            # Test invalid document ID
            rag.document_ids.add("VALID_DOC")
            with pytest.raises(QueryError):
                rag.query("Test question", document_id="INVALID_DOC")
            
        except ImportError:
            # Skip if dependencies not available
            pytest.skip("Required dependencies not available for integration test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])