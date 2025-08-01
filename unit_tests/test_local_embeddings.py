"""
Quick test for local embeddings
"""

from simple_rag import VectorStore
from langchain.schema import Document

def test_local_embeddings():
    """Test local embeddings initialization"""

    print("Testing local embeddings initialization...")

    try:
        # Initialize with local model
        vector_store = VectorStore(embedding_model="all-MiniLM-L6-v2")
        print("✓ VectorStore initialized with local embeddings")

        # Test with a simple document
        test_doc = Document(
            page_content="This is a test document for local embeddings.",
            metadata={"document_id": "TEST_DOC", "chunk_index": 0}
        )

        print("Adding test document...")
        vector_store.add_documents([test_doc])
        print("✓ Document added successfully")

        # Test search
        print("Testing search...")
        results = vector_store.similarity_search("test document", top_k=1)
        print(f"✓ Search returned {len(results)} results")

        if results:
            doc, score = results[0]
            print(f"  Content: {doc.page_content}")
            print(f"  Score: {score}")

        print("\n✓ Local embeddings test passed!")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_local_embeddings()