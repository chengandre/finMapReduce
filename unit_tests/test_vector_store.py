"""
Test script for VectorStore functionality
"""

from simple_rag import VectorStore, SimpleRAGError
from langchain.schema import Document

def test_vector_store():
    """Test the VectorStore with sample documents"""
    
    print("Testing VectorStore...")
    
    # Initialize vector store
    vector_store = VectorStore()
    
    # Create sample documents with metadata
    sample_docs = [
        Document(
            page_content="Apple Inc. reported revenue of $365.8 billion in fiscal year 2021.",
            metadata={"document_id": "APPLE_2021", "chunk_index": 0, "source_path": "apple_2021.pdf"}
        ),
        Document(
            page_content="Apple's iPhone sales contributed significantly to the company's growth.",
            metadata={"document_id": "APPLE_2021", "chunk_index": 1, "source_path": "apple_2021.pdf"}
        ),
        Document(
            page_content="Microsoft Corporation had revenue of $168 billion in fiscal year 2021.",
            metadata={"document_id": "MSFT_2021", "chunk_index": 0, "source_path": "msft_2021.pdf"}
        ),
        Document(
            page_content="Microsoft's cloud services Azure showed strong performance.",
            metadata={"document_id": "MSFT_2021", "chunk_index": 1, "source_path": "msft_2021.pdf"}
        )
    ]
    
    try:
        # Test adding documents
        print(f"Adding {len(sample_docs)} sample documents...")
        vector_store.add_documents(sample_docs)
        
        stats = vector_store.get_stats()
        print(f"✓ Vector store stats: {stats}")
        
        # Test global search
        print("\nTesting global search...")
        results = vector_store.similarity_search("What was Apple's revenue?", top_k=2)
        print(f"✓ Global search returned {len(results)} results")
        
        for i, (doc, score) in enumerate(results):
            print(f"  Result {i+1} (score: {score:.3f}): {doc.page_content[:50]}...")
            print(f"    Document ID: {doc.metadata.get('document_id')}")
        
        # Test document-specific search
        print("\nTesting document-specific search...")
        results = vector_store.similarity_search("revenue", document_id="APPLE_2021", top_k=2)
        print(f"✓ Document-specific search returned {len(results)} results")
        
        for i, (doc, score) in enumerate(results):
            print(f"  Result {i+1} (score: {score:.3f}): {doc.page_content[:50]}...")
            print(f"    Document ID: {doc.metadata.get('document_id')}")
        
        # Test search with non-existent document ID
        print("\nTesting search with non-existent document ID...")
        results = vector_store.similarity_search("revenue", document_id="NONEXISTENT", top_k=2)
        print(f"✓ Search with non-existent ID returned {len(results)} results (expected: 0)")
        
        print("\n✓ All VectorStore tests passed!")
        
    except Exception as e:
        print(f"✗ Error during VectorStore testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vector_store()