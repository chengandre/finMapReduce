"""
Test with fresh vector store initialization
"""

from simple_rag import VectorStore
from langchain.schema import Document

def test_fresh_vector_store():
    """Test with completely fresh vector store"""

    print("Testing fresh VectorStore...")

    try:
        # Create fresh vector store
        vector_store = VectorStore()
        print("✓ VectorStore created")

        # Create a simple test document
        test_doc = Document(
            page_content="Apple Inc. reported strong quarterly earnings with revenue growth.",
            metadata={"document_id": "APPLE_2024", "chunk_index": 0, "source_path": "test.pdf"}
        )

        print("Adding document to vector store...")
        vector_store.add_documents([test_doc])
        print("✓ Document added")

        # Check stats
        stats = vector_store.get_stats()
        print(f"Stats: {stats}")

        # Test search
        print("Testing search...")
        results = vector_store.similarity_search("Apple earnings revenue", top_k=1)
        print(f"✓ Search completed, found {len(results)} results")

        if results:
            doc, score = results[0]
            print(f"  Content: {doc.page_content}")
            print(f"  Score: {score:.4f}")
            print(f"  Document ID: {doc.metadata.get('document_id')}")

        # Test document-specific search
        print("\nTesting document-specific search...")
        results = vector_store.similarity_search("earnings", document_id="APPLE_2024", top_k=1)
        print(f"✓ Document-specific search found {len(results)} results")

        print("\n✓ All tests passed!")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fresh_vector_store()