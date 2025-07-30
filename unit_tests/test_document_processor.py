"""
Test script for DocumentProcessor functionality
"""

import os
from simple_rag import DocumentProcessor, DocumentProcessingError

def test_document_processor():
    """Test the DocumentProcessor with a sample document"""
    
    # Initialize processor
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    
    # Test with a non-existent file to check error handling
    try:
        chunks = processor.process_document("nonexistent.pdf", "TEST_DOC")
        print("ERROR: Should have failed with non-existent file")
    except DocumentProcessingError as e:
        print(f"✓ Correctly handled non-existent file: {e}")
    
    # Look for any PDF files in the current directory to test with
    pdf_files = "3M_2018_10K"
    
    if pdf_files:
        test_file = pdf_files
        print(f"\nTesting with file: {test_file}")
        
        try:
            chunks = processor.process_document(test_file, "TEST_2024")
            
            print(f"✓ Successfully processed document:")
            print(f"  - Number of chunks: {len(chunks)}")
            
            if chunks:
                first_chunk = chunks[0]
                print(f"  - First chunk metadata: {first_chunk.metadata}")
                print(f"  - First chunk content preview: {first_chunk.page_content[:100]}...")
                
                # Verify metadata structure
                required_fields = ["document_id", "chunk_index", "source_path", "total_chunks"]
                for field in required_fields:
                    if field in first_chunk.metadata:
                        print(f"  ✓ {field}: {first_chunk.metadata[field]}")
                    else:
                        print(f"  ✗ Missing field: {field}")
        
        except Exception as e:
            print(f"✗ Error processing document: {e}")
    else:
        print("No PDF files found in current directory for testing")
        print("You can test manually by placing a PDF file in the directory")

if __name__ == "__main__":
    test_document_processor()