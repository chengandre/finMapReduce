"""
Unit tests for QueryProcessor class

Tests cover answer generation, context formatting, response processing,
and error handling scenarios.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document

from simple_rag import QueryProcessor, QueryError


class TestQueryProcessor:
    """Test suite for QueryProcessor class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.query_processor = QueryProcessor(
            llm_model="gpt-4o-mini",
            temperature=0.1
        )
        
        # Sample document chunks for testing
        self.sample_chunks = [
            Document(
                page_content="Apple Inc. reported revenue of $394.3 billion in fiscal year 2022, representing a 8% increase from the previous year.",
                metadata={
                    "document_id": "APPLE_2022",
                    "chunk_index": 0,
                    "source_name": "apple_2022_10k",
                    "total_chunks": 10
                }
            ),
            Document(
                page_content="The company's iPhone segment generated $205.5 billion in revenue, accounting for 52% of total revenue.",
                metadata={
                    "document_id": "APPLE_2022",
                    "chunk_index": 1,
                    "source_name": "apple_2022_10k",
                    "total_chunks": 10
                }
            ),
            Document(
                page_content="Microsoft Corporation reported total revenue of $198.3 billion for fiscal year 2022.",
                metadata={
                    "document_id": "MSFT_2022",
                    "chunk_index": 0,
                    "source_name": "msft_2022_10k",
                    "total_chunks": 8
                }
            )
        ]
    
    def test_initialization(self):
        """Test QueryProcessor initialization"""
        qp = QueryProcessor(llm_model="gpt-3.5-turbo", temperature=0.2)
        assert qp.llm.model_name == "gpt-3.5-turbo"
        assert qp.llm.temperature == 0.2
        assert qp.llm.max_tokens == 1500
    
    def test_format_context_with_chunks(self):
        """Test context formatting with multiple chunks"""
        context = self.query_processor._format_context(self.sample_chunks)
        
        # Check that all chunks are included
        assert "[Chunk 1]" in context
        assert "[Chunk 2]" in context
        assert "[Chunk 3]" in context
        
        # Check metadata is included
        assert "Document ID: APPLE_2022" in context
        assert "Document ID: MSFT_2022" in context
        assert "Chunk Index: 0" in context
        assert "Chunk Index: 1" in context
        assert "Source: apple_2022_10k" in context
        assert "Source: msft_2022_10k" in context
        
        # Check content is included
        assert "Apple Inc. reported revenue" in context
        assert "iPhone segment generated" in context
        assert "Microsoft Corporation reported" in context
        
        # Check structure
        assert "---" in context  # Separator between chunks
    
    def test_format_context_empty_chunks(self):
        """Test context formatting with empty chunks list"""
        context = self.query_processor._format_context([])
        assert context == ""
    
    def test_format_context_missing_metadata(self):
        """Test context formatting with missing metadata"""
        chunk_no_metadata = Document(
            page_content="Test content without metadata",
            metadata={}
        )
        
        context = self.query_processor._format_context([chunk_no_metadata])
        
        assert "Document ID: Unknown" in context
        assert "Chunk Index: Unknown" in context
        assert "Source: Unknown" in context
        assert "Test content without metadata" in context
    
    @patch('simple_rag.load_prompt')
    def test_create_answer_prompt(self, mock_load_prompt):
        """Test prompt template creation from YAML file"""
        # Mock the loaded prompt
        mock_prompt = Mock()
        mock_prompt.input_variables = ["context", "question"]
        mock_prompt.template = """<mode>test</mode><template>{"reasoning": "test"}</template>"""
        mock_load_prompt.return_value = mock_prompt
        
        prompt = self.query_processor._create_answer_prompt()
        
        # Check that load_prompt was called with correct file
        mock_load_prompt.assert_called_once_with("prompts/rag_prompt.yml")
        
        # Check that prompt has correct input variables
        assert "context" in prompt.input_variables
        assert "question" in prompt.input_variables
    
    @patch('simple_rag.GPT')
    def test_generate_answer_success(self, mock_gpt_class):
        """Test successful answer generation with structured response"""
        # Mock GPT response
        mock_response = {
            'json': {
                "reasoning": "Based on the provided context, Apple reported revenue of $394.3 billion.",
                "evidence": ["Apple Inc. reported revenue of $394.3 billion in fiscal year 2022"],
                "answer": "Apple's total revenue in fiscal year 2022 was $394.3 billion.",
                "confidence": "high",
                "sources": ["APPLE_2022"]
            },
            'raw_response': Mock(content="Mock raw response")
        }
        
        mock_gpt_instance = Mock()
        mock_gpt_instance.return_value = mock_response
        mock_gpt_class.return_value = mock_gpt_instance
        
        # Create QueryProcessor with mocked GPT
        qp = QueryProcessor()
        qp.llm = mock_gpt_instance
        
        # Test answer generation
        question = "What was Apple's total revenue in 2022?"
        answer = qp.generate_answer(question, self.sample_chunks)
        
        # Verify the response
        assert "Apple's total revenue in fiscal year 2022 was $394.3 billion" in answer
        assert "**Confidence:** High" in answer
        assert "**Sources:** APPLE_2022" in answer
        assert "**Supporting Evidence:**" in answer
        
        # Verify GPT was called
        mock_gpt_instance.assert_called_once()
    
    @patch('simple_rag.GPT')
    def test_generate_answer_fallback(self, mock_gpt_class):
        """Test answer generation with fallback when JSON parsing fails"""
        # Mock GPT response without JSON
        mock_response = {
            'raw_response': Mock(content="Apple reported $394.3 billion in revenue for 2022.")
        }
        
        mock_gpt_instance = Mock()
        mock_gpt_instance.return_value = mock_response
        mock_gpt_class.return_value = mock_gpt_instance
        
        # Create QueryProcessor with mocked GPT
        qp = QueryProcessor()
        qp.llm = mock_gpt_instance
        
        # Test answer generation
        question = "What was Apple's revenue?"
        answer = qp.generate_answer(question, self.sample_chunks)
        
        # Verify fallback response
        assert "Apple reported $394.3 billion in revenue for 2022." in answer
        assert "**Sources:** APPLE_2022, MSFT_2022" in answer
    
    def test_generate_answer_empty_chunks(self):
        """Test answer generation with empty chunks"""
        answer = self.query_processor.generate_answer("Test question", [])
        assert answer == "I couldn't find any relevant information to answer your question."
    
    @patch('simple_rag.GPT')
    def test_generate_answer_error_handling(self, mock_gpt_class):
        """Test error handling during answer generation"""
        # Mock GPT to raise an exception
        mock_gpt_instance = Mock()
        mock_gpt_instance.side_effect = Exception("API Error")
        mock_gpt_class.return_value = mock_gpt_instance
        
        # Create QueryProcessor with mocked GPT
        qp = QueryProcessor()
        qp.llm = mock_gpt_instance
        
        # Test error handling
        answer = qp.generate_answer("Test question", self.sample_chunks)
        assert "I encountered an error while generating the answer: API Error" in answer
    
    def test_format_final_answer_complete(self):
        """Test formatting final answer with all components"""
        answer = "Apple's revenue was $394.3 billion."
        reasoning = "Based on financial data from the annual report."
        evidence = ["Apple Inc. reported revenue of $394.3 billion"]
        confidence = "high"
        sources = ["APPLE_2022"]
        
        formatted = self.query_processor._format_final_answer(
            answer, reasoning, evidence, confidence, sources
        )
        
        assert answer in formatted
        assert "**Confidence:** High" in formatted
        assert "**Sources:** APPLE_2022" in formatted
        assert "**Supporting Evidence:**" in formatted
        assert "Apple Inc. reported revenue of $394.3 billion" in formatted
    
    def test_format_final_answer_minimal(self):
        """Test formatting final answer with minimal components"""
        answer = "No sufficient information found."
        
        formatted = self.query_processor._format_final_answer(
            answer, "", [], "", []
        )
        
        assert answer in formatted
        assert "**Confidence:**" not in formatted
        assert "**Sources:**" not in formatted
        assert "**Supporting Evidence:**" not in formatted
    
    def test_format_final_answer_long_evidence(self):
        """Test formatting with long evidence (should be truncated)"""
        answer = "Test answer"
        evidence = [
            "This is a very long piece of evidence that should be truncated because it exceeds the 200 character limit that we have set for evidence display in the final answer to keep things concise and readable for the user.",
            "Short evidence",
            "Another piece of evidence",
            "Fourth evidence (should be excluded due to limit)"
        ]
        
        formatted = self.query_processor._format_final_answer(
            answer, "", evidence, "medium", ["TEST_DOC"]
        )
        
        # Should only include first 3 pieces of evidence
        assert "This is a very long piece of evidence" in formatted
        assert "Short evidence" in formatted
        assert "Another piece of evidence" in formatted
        assert "Fourth evidence" not in formatted  # Excluded due to limit
        
        # Long evidence should be truncated
        assert "..." in formatted
    
    def test_fallback_response_with_sources(self):
        """Test fallback response processing with source attribution"""
        raw_content = "Apple's revenue was significant in 2022."
        
        result = self.query_processor._fallback_response(raw_content, self.sample_chunks)
        
        assert raw_content in result
        assert "**Sources:** APPLE_2022, MSFT_2022" in result
    
    def test_fallback_response_no_sources(self):
        """Test fallback response with no document IDs"""
        chunks_no_ids = [
            Document(
                page_content="Test content",
                metadata={"chunk_index": 0}  # No document_id
            )
        ]
        
        raw_content = "Test response"
        result = self.query_processor._fallback_response(raw_content, chunks_no_ids)
        
        assert result == raw_content  # No source attribution added
    
    def test_process_response_json_success(self):
        """Test processing successful JSON response"""
        mock_response = {
            'json': {
                "reasoning": "Test reasoning",
                "evidence": ["Test evidence"],
                "answer": "Test answer",
                "confidence": "high",
                "sources": ["TEST_DOC"]
            }
        }
        
        result = self.query_processor._process_response(mock_response, self.sample_chunks)
        
        assert "Test answer" in result
        assert "**Confidence:** High" in result
        assert "**Sources:** TEST_DOC" in result
    
    def test_process_response_json_failure(self):
        """Test processing response when JSON parsing fails"""
        mock_response = {
            'raw_response': Mock(content="Fallback response content")
        }
        
        result = self.query_processor._process_response(mock_response, self.sample_chunks)
        
        assert "Fallback response content" in result
        assert "**Sources:** APPLE_2022, MSFT_2022" in result
    
    def test_process_response_invalid_format(self):
        """Test processing response with invalid format"""
        mock_response = "Invalid response format"
        
        result = self.query_processor._process_response(mock_response, self.sample_chunks)
        
        assert "Invalid response format" in result


class TestQueryProcessorIntegration:
    """Integration tests for QueryProcessor with real-like scenarios"""
    
    def setup_method(self):
        """Set up integration test fixtures"""
        self.query_processor = QueryProcessor()
        
        # More realistic document chunks
        self.financial_chunks = [
            Document(
                page_content="""
                CONSOLIDATED STATEMENTS OF OPERATIONS
                (In millions, except number of shares which are reflected in thousands and per share amounts)
                
                Net sales:
                Products: $365,817
                Services: $78,129
                Total net sales: $443,946
                
                Cost of sales:
                Products: $223,546
                Services: $22,075
                Total cost of sales: $245,621
                
                Gross margin: $198,325
                """,
                metadata={
                    "document_id": "APPLE_2023",
                    "chunk_index": 5,
                    "source_name": "apple_2023_10k",
                    "total_chunks": 45
                }
            ),
            Document(
                page_content="""
                The Company's business primarily runs on a geographic basis. The Americas segment includes North and South America. 
                The Europe segment includes European countries, as well as India, the Middle East and Africa. 
                The Greater China segment includes China mainland, Hong Kong and Taiwan. 
                The Japan segment includes Japan. The Rest of Asia Pacific segment includes Australia and those Asian countries not included in other segments.
                
                Net sales by reportable segment:
                Americas: $162,560 million
                Europe: $94,294 million  
                Greater China: $72,559 million
                Japan: $24,257 million
                Rest of Asia Pacific: $29,615 million
                """,
                metadata={
                    "document_id": "APPLE_2023",
                    "chunk_index": 12,
                    "source_name": "apple_2023_10k",
                    "total_chunks": 45
                }
            )
        ]
    
    @patch('simple_rag.GPT')
    def test_realistic_financial_query(self, mock_gpt_class):
        """Test with realistic financial query and response"""
        # Mock a realistic structured response
        mock_response = {
            'json': {
                "reasoning": "Based on the consolidated statements of operations, Apple's total net sales were $443,946 million, with products contributing $365,817 million and services contributing $78,129 million.",
                "evidence": [
                    "Total net sales: $443,946",
                    "Products: $365,817",
                    "Services: $78,129"
                ],
                "answer": "Apple's total net sales in 2023 were $443,946 million ($443.9 billion), with products accounting for $365,817 million and services for $78,129 million.",
                "confidence": "high",
                "sources": ["APPLE_2023"]
            }
        }
        
        mock_gpt_instance = Mock()
        mock_gpt_instance.return_value = mock_response
        mock_gpt_class.return_value = mock_gpt_instance
        
        qp = QueryProcessor()
        qp.llm = mock_gpt_instance
        
        question = "What were Apple's total net sales in 2023?"
        answer = qp.generate_answer(question, self.financial_chunks)
        
        # Verify comprehensive response
        assert "$443,946 million" in answer or "$443.9 billion" in answer
        assert "products" in answer.lower()
        assert "services" in answer.lower()
        assert "**Confidence:** High" in answer
        assert "**Sources:** APPLE_2023" in answer
        assert "**Supporting Evidence:**" in answer
    
    def test_context_formatting_realistic(self):
        """Test context formatting with realistic financial data"""
        context = self.query_processor._format_context(self.financial_chunks)
        
        # Check structure
        assert "[Chunk 1]" in context
        assert "[Chunk 2]" in context
        assert "Document ID: APPLE_2023" in context
        
        # Check financial data is preserved
        assert "CONSOLIDATED STATEMENTS OF OPERATIONS" in context
        assert "Total net sales: $443,946" in context
        assert "Americas: $162,560 million" in context
        assert "Greater China: $72,559 million" in context
        
        # Check metadata
        assert "Chunk Index: 5" in context
        assert "Chunk Index: 12" in context
        assert "apple_2023_10k" in context


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])