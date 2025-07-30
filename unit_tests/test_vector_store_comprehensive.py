"""
Comprehensive unit tests for VectorStore class with FAISS integration
Tests focus on local embeddings and avoid OpenAI API calls
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document

from simple_rag import VectorStore, SimpleRAGError, QueryError, LocalEmbeddings


class TestVectorStore:
    """Test suite for VectorStore class"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.test_documents = [
            Document(
                page_content="Apple Inc. reported strong quarterly earnings with revenue growth of 15%.",
                metadata={"document_id": "APPLE_2024", "chunk_index": 0, "source_path": "apple_2024.pdf"}
            ),
            Document(
                page_content="Microsoft announced new AI features in their cloud platform Azure.",
                metadata={"document_id": "MSFT_2024", "chunk_index": 0, "source_path": "msft_2024.pdf"}
            ),
            Document(
                page_content="Apple's iPhone sales exceeded expectations in the last quarter.",
                metadata={"document_id": "APPLE_2024", "chunk_index": 1, "source_path": "apple_2024.pdf"}
            )
        ]
    
    def test_vector_store_initialization_local(self):
        """Test VectorStore initialization with local embeddings"""
        with patch('simple_rag.SENTENCE_TRANSFORMERS_AVAILABLE', True):
            with patch('simple_rag.LocalEmbeddings') as mock_embeddings:
                mock_embeddings.return_value = Mock()
                
                vector_store = VectorStore(embedding_model="all-MiniLM-L6-v2", use_local=True)
                
                assert vector_store.embedding_model == "all-MiniLM-L6-v2"
                assert vector_store.use_local == True
                assert vector_store.vector_store is None
                mock_embeddings.assert_called_once_with("all-MiniLM-L6-v2")
    
    def test_vector_store_initialization_fallback(self):
        """Test VectorStore initialization fallback when local unavailable"""
        with patch('simple_rag.SENTENCE_TRANSFORMERS_AVAILABLE', False):
            # Skip OpenAI embedding tests as requested
            pass
    
    def test_add_documents_empty_list(self):
        """Test adding empty document list"""
        with patch('simple_rag.SENTENCE_TRANSFORMERS_AVAILABLE', True):
            with patch('simple_rag.LocalEmbeddings'):
                vector_store = VectorStore()
                
                # Should not raise error, just log warning
                vector_store.add_documents([])
                assert vector_store.vector_store is None
    
    def test_add_documents_first_batch(self):
        """Test adding documents to empty vector store"""
        with patch('simple_rag.SENTENCE_TRANSFORMERS_AVAILABLE', True):
            with patch('simple_rag.LocalEmbeddings') as mock_embeddings:
                with patch('simple_rag.FAISS') as mock_faiss:
                    mock_embeddings.return_value = Mock()
                    mock_vector_store = Mock()
                    mock_vector_store.docstore._dict = {"doc1": "content1", "doc2": "content2"}
                    mock_faiss.from_documents.return_value = mock_vector_store
                    
                    vector_store = VectorStore()
                    vector_store.add_documents(self.test_documents[:2])
                    
                    mock_faiss.from_documents.assert_called_once_with(
                        self.test_documents[:2], 
                        vector_store.embeddings
                    )
                    assert vector_store.vector_store == mock_vector_store
    
    def test_add_documents_to_existing(self):
        """Test adding documents to existing vector store"""
        with patch('simple_rag.SENTENCE_TRANSFORMERS_AVAILABLE', True):
            with patch('simple_rag.LocalEmbeddings') as mock_embeddings:
                with patch('simple_rag.FAISS') as mock_faiss:
                    mock_embeddings.return_value = Mock()
                    mock_vector_store = Mock()
                    mock_vector_store.docstore._dict = {"doc1": "content1", "doc2": "content2"}
                    mock_faiss.from_documents.return_value = mock_vector_store
                    
                    vector_store = VectorStore()
                    # Add first batch
                    vector_store.add_documents(self.test_documents[:1])
                    
                    # Add second batch
                    vector_store.add_documents(self.test_documents[1:])
                    
                    # Should call add_documents on existing vector store
                    mock_vector_store.add_documents.assert_called_once_with(self.test_documents[1:])
    
    def test_add_documents_error_handling(self):
        """Test error handling during document addition"""
        with patch('simple_rag.SENTENCE_TRANSFORMERS_AVAILABLE', True):
            with patch('simple_rag.LocalEmbeddings') as mock_embeddings:
                with patch('simple_rag.FAISS') as mock_faiss:
                    mock_embeddings.return_value = Mock()
                    mock_faiss.from_documents.side_effect = Exception("FAISS error")
                    
                    vector_store = VectorStore()
                    
                    with pytest.raises(SimpleRAGError, match="Failed to add documents to vector store"):
                        vector_store.add_documents(self.test_documents[:1])
    
    def test_similarity_search_empty_store(self):
        """Test similarity search on empty vector store"""
        with patch('simple_rag.SENTENCE_TRANSFORMERS_AVAILABLE', True):
            with patch('simple_rag.LocalEmbeddings'):
                vector_store = VectorStore()
                
                results = vector_store.similarity_search("test query")
                assert results == []
    
    def test_similarity_search_global(self):
        """Test global similarity search across all documents"""
        with patch('simple_rag.SENTENCE_TRANSFORMERS_AVAILABLE', True):
            with patch('simple_rag.LocalEmbeddings') as mock_embeddings:
                mock_embeddings.return_value = Mock()
                mock_vector_store = Mock()
                mock_results = [
                    (self.test_documents[0], 0.85),
                    (self.test_documents[2], 0.75)
                ]
                mock_vector_store.similarity_search_with_score.return_value = mock_results
                
                vector_store = VectorStore()
                vector_store.vector_store = mock_vector_store
                
                results = vector_store.similarity_search("Apple earnings", top_k=2)
                
                assert len(results) == 2
                assert results[0][0] == self.test_documents[0]
                assert results[0][1] == 0.85
                mock_vector_store.similarity_search_with_score.assert_called_once_with("Apple earnings", k=2)
    
    def test_similarity_search_document_specific(self):
        """Test document-specific similarity search with metadata filtering"""
        with patch('simple_rag.SENTENCE_TRANSFORMERS_AVAILABLE', True):
            with patch('simple_rag.LocalEmbeddings') as mock_embeddings:
                mock_embeddings.return_value = Mock()
                mock_vector_store = Mock()
                # Mock results include documents from different document_ids
                mock_results = [
                    (self.test_documents[0], 0.85),  # APPLE_2024
                    (self.test_documents[1], 0.80),  # MSFT_2024
                    (self.test_documents[2], 0.75)   # APPLE_2024
                ]
                mock_vector_store.similarity_search_with_score.return_value = mock_results
                
                vector_store = VectorStore()
                vector_store.vector_store = mock_vector_store
                
                results = vector_store.similarity_search("earnings", document_id="APPLE_2024", top_k=2)
                
                # Should only return documents with APPLE_2024 document_id
                assert len(results) == 2
                assert results[0][0] == self.test_documents[0]
                assert results[1][0] == self.test_documents[2]
                assert all(doc.metadata.get("document_id") == "APPLE_2024" for doc, _ in results)
                
                # Should search with higher k to account for filtering
                mock_vector_store.similarity_search_with_score.assert_called_once_with("earnings", k=20)
    
    def test_similarity_search_document_specific_no_matches(self):
        """Test document-specific search when no documents match the document_id"""
        with patch('simple_rag.SENTENCE_TRANSFORMERS_AVAILABLE', True):
            with patch('simple_rag.LocalEmbeddings') as mock_embeddings:
                mock_embeddings.return_value = Mock()
                mock_vector_store = Mock()
                # Mock results with no matching document_id
                mock_results = [
                    (self.test_documents[1], 0.80),  # MSFT_2024
                ]
                mock_vector_store.similarity_search_with_score.return_value = mock_results
                
                vector_store = VectorStore()
                vector_store.vector_store = mock_vector_store
                
                results = vector_store.similarity_search("test", document_id="NONEXISTENT", top_k=5)
                
                assert len(results) == 0
    
    def test_similarity_search_error_handling(self):
        """Test error handling during similarity search"""
        with patch('simple_rag.SENTENCE_TRANSFORMERS_AVAILABLE', True):
            with patch('simple_rag.LocalEmbeddings') as mock_embeddings:
                mock_embeddings.return_value = Mock()
                mock_vector_store = Mock()
                mock_vector_store.similarity_search_with_score.side_effect = Exception("Search error")
                
                vector_store = VectorStore()
                vector_store.vector_store = mock_vector_store
                
                with pytest.raises(QueryError, match="Failed to perform similarity search"):
                    vector_store.similarity_search("test query")
    
    def test_get_stats_empty_store(self):
        """Test getting statistics from empty vector store"""
        with patch('simple_rag.SENTENCE_TRANSFORMERS_AVAILABLE', True):
            with patch('simple_rag.LocalEmbeddings'):
                vector_store = VectorStore(embedding_model="test-model")
                
                stats = vector_store.get_stats()
                
                assert stats["total_documents"] == 0
                assert stats["embedding_model"] == "test-model"
    
    def test_get_stats_with_documents(self):
        """Test getting statistics from populated vector store"""
        with patch('simple_rag.SENTENCE_TRANSFORMERS_AVAILABLE', True):
            with patch('simple_rag.LocalEmbeddings') as mock_embeddings:
                mock_embeddings.return_value = Mock()
                mock_vector_store = Mock()
                mock_vector_store.docstore._dict = {"doc1": "content1", "doc2": "content2", "doc3": "content3"}
                
                vector_store = VectorStore(embedding_model="test-model")
                vector_store.vector_store = mock_vector_store
                
                stats = vector_store.get_stats()
                
                assert stats["total_documents"] == 3
                assert stats["embedding_model"] == "test-model"
    
    def test_clear_vector_store(self):
        """Test clearing the vector store"""
        with patch('simple_rag.SENTENCE_TRANSFORMERS_AVAILABLE', True):
            with patch('simple_rag.LocalEmbeddings') as mock_embeddings:
                mock_embeddings.return_value = Mock()
                mock_vector_store = Mock()
                
                vector_store = VectorStore()
                vector_store.vector_store = mock_vector_store
                
                vector_store.clear()
                
                assert vector_store.vector_store is None
    
    def test_save_local_with_data(self):
        """Test saving vector store to local disk"""
        with patch('simple_rag.SENTENCE_TRANSFORMERS_AVAILABLE', True):
            with patch('simple_rag.LocalEmbeddings') as mock_embeddings:
                mock_embeddings.return_value = Mock()
                mock_vector_store = Mock()
                
                vector_store = VectorStore()
                vector_store.vector_store = mock_vector_store
                
                vector_store.save_local("/test/path")
                
                mock_vector_store.save_local.assert_called_once_with("/test/path")
    
    def test_save_local_empty_store(self):
        """Test saving empty vector store (should not crash)"""
        with patch('simple_rag.SENTENCE_TRANSFORMERS_AVAILABLE', True):
            with patch('simple_rag.LocalEmbeddings'):
                vector_store = VectorStore()
                
                # Should not raise error
                vector_store.save_local("/test/path")
    
    def test_load_local_success(self):
        """Test loading vector store from local disk"""
        with patch('simple_rag.SENTENCE_TRANSFORMERS_AVAILABLE', True):
            with patch('simple_rag.LocalEmbeddings') as mock_embeddings:
                with patch('simple_rag.FAISS') as mock_faiss:
                    mock_embeddings_instance = Mock()
                    mock_embeddings.return_value = mock_embeddings_instance
                    mock_vector_store = Mock()
                    mock_faiss.load_local.return_value = mock_vector_store
                    
                    vector_store = VectorStore()
                    vector_store.load_local("/test/path")
                    
                    mock_faiss.load_local.assert_called_once_with(
                        "/test/path", 
                        mock_embeddings_instance,
                        allow_dangerous_deserialization=True
                    )
                    assert vector_store.vector_store == mock_vector_store
    
    def test_load_local_error_handling(self):
        """Test error handling when loading vector store fails"""
        with patch('simple_rag.SENTENCE_TRANSFORMERS_AVAILABLE', True):
            with patch('simple_rag.LocalEmbeddings') as mock_embeddings:
                with patch('simple_rag.FAISS') as mock_faiss:
                    mock_embeddings.return_value = Mock()
                    mock_faiss.load_local.side_effect = Exception("Load error")
                    
                    vector_store = VectorStore()
                    
                    with pytest.raises(SimpleRAGError, match="Failed to load vector store"):
                        vector_store.load_local("/test/path")


class TestLocalEmbeddings:
    """Test suite for LocalEmbeddings class"""
    
    def test_local_embeddings_initialization(self):
        """Test LocalEmbeddings initialization"""
        with patch('simple_rag.SENTENCE_TRANSFORMERS_AVAILABLE', True):
            with patch('simple_rag.SentenceTransformer') as mock_st:
                mock_model = Mock()
                mock_st.return_value = mock_model
                
                from simple_rag import LocalEmbeddings
                embeddings = LocalEmbeddings("test-model")
                
                assert embeddings.model_name == "test-model"
                assert embeddings.model == mock_model
                mock_st.assert_called_once_with("test-model")
    
    def test_local_embeddings_unavailable(self):
        """Test LocalEmbeddings when sentence-transformers is unavailable"""
        with patch('simple_rag.SENTENCE_TRANSFORMERS_AVAILABLE', False):
            from simple_rag import LocalEmbeddings
            
            with pytest.raises(ImportError, match="sentence-transformers is required"):
                LocalEmbeddings()
    
    def test_embed_documents(self):
        """Test embedding multiple documents"""
        with patch('simple_rag.SENTENCE_TRANSFORMERS_AVAILABLE', True):
            with patch('simple_rag.SentenceTransformer') as mock_st:
                mock_model = Mock()
                mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
                mock_model.encode.return_value = Mock()
                mock_model.encode.return_value.tolist.return_value = mock_embeddings
                mock_st.return_value = mock_model
                
                from simple_rag import LocalEmbeddings
                embeddings = LocalEmbeddings()
                
                texts = ["text1", "text2"]
                result = embeddings.embed_documents(texts)
                
                assert result == mock_embeddings
                mock_model.encode.assert_called_once_with(texts, convert_to_tensor=False)
    
    def test_embed_query(self):
        """Test embedding single query"""
        with patch('simple_rag.SENTENCE_TRANSFORMERS_AVAILABLE', True):
            with patch('simple_rag.SentenceTransformer') as mock_st:
                # Clear the cache to avoid interference
                with patch('simple_rag.LocalEmbeddings._model_cache', {}):
                    mock_model = Mock()
                    # Create a mock that properly supports indexing
                    mock_embedding = Mock()
                    mock_embedding.tolist.return_value = [0.1, 0.2, 0.3]
                    
                    # Make the encode result indexable - use actual list
                    mock_model.encode.return_value = [mock_embedding]
                    mock_st.return_value = mock_model
                    
                    embeddings = LocalEmbeddings()
                    
                    result = embeddings.embed_query("test query")
                    
                    assert result == [0.1, 0.2, 0.3]
                    mock_model.encode.assert_called_once_with(["test query"], convert_to_tensor=False)
    
    def test_callable_interface(self):
        """Test that LocalEmbeddings is callable (for LangChain compatibility)"""
        with patch('simple_rag.SENTENCE_TRANSFORMERS_AVAILABLE', True):
            with patch('simple_rag.SentenceTransformer') as mock_st:
                # Clear the cache to avoid interference
                with patch('simple_rag.LocalEmbeddings._model_cache', {}):
                    mock_model = Mock()
                    # Create a mock that properly supports indexing
                    mock_embedding = Mock()
                    mock_embedding.tolist.return_value = [0.1, 0.2, 0.3]
                    
                    # Make the encode result indexable - use actual list
                    mock_model.encode.return_value = [mock_embedding]
                    mock_st.return_value = mock_model
                    
                    embeddings = LocalEmbeddings()
                    
                    result = embeddings("test query")
                    
                    assert result == [0.1, 0.2, 0.3]


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])