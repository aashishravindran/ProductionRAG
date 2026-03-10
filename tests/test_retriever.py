"""Tests for retrieval.retriever module."""
import pytest
from langchain_core.documents import Document


class TestGetRetriever:
    def test_returns_retriever(self, populated_store, fake_embeddings):
        from retrieval.retriever import get_retriever

        retriever = get_retriever(
            embedding_function=fake_embeddings,
            persist_directory=populated_store["persist_dir"],
            collection_name=populated_store["collection_name"],
        )
        assert hasattr(retriever, "invoke")

    def test_respects_top_k(self, populated_store, fake_embeddings):
        from retrieval.retriever import get_retriever

        retriever = get_retriever(
            embedding_function=fake_embeddings,
            persist_directory=populated_store["persist_dir"],
            collection_name=populated_store["collection_name"],
            top_k=1,
        )
        results = retriever.invoke("Python")
        assert len(results) == 1


class TestRetrieve:
    def test_returns_documents(self, populated_store, fake_embeddings):
        from retrieval.retriever import retrieve

        results = retrieve(
            query="Python experience",
            embedding_function=fake_embeddings,
            persist_directory=populated_store["persist_dir"],
            collection_name=populated_store["collection_name"],
        )
        assert len(results) > 0
        assert all(isinstance(doc, Document) for doc in results)

    def test_respects_top_k(self, populated_store, fake_embeddings):
        from retrieval.retriever import retrieve

        results = retrieve(
            query="Python",
            embedding_function=fake_embeddings,
            persist_directory=populated_store["persist_dir"],
            collection_name=populated_store["collection_name"],
            top_k=1,
        )
        assert len(results) == 1

    def test_nonexistent_store_raises(self, tmp_path, fake_embeddings):
        from retrieval.retriever import retrieve

        with pytest.raises(FileNotFoundError):
            retrieve(
                query="anything",
                embedding_function=fake_embeddings,
                persist_directory=tmp_path / "nonexistent",
                collection_name="missing",
            )
