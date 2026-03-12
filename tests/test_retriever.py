"""Tests for retrieval.retriever module (hybrid retrieve + get_retriever)."""
import pytest
from unittest.mock import patch, MagicMock
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
    @patch("retrieval.retriever.CrossEncoder")
    def test_returns_documents(self, mock_ce_cls, populated_store, fake_embeddings):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.8, 0.7]
        mock_ce_cls.return_value = mock_model

        from retrieval.retriever import retrieve

        results = retrieve(
            query="Python experience",
            embedding_function=fake_embeddings,
            persist_directory=populated_store["persist_dir"],
            collection_name=populated_store["collection_name"],
        )
        assert len(results) > 0
        assert all(isinstance(doc, Document) for doc in results)

    @patch("retrieval.retriever.CrossEncoder")
    def test_respects_top_k(self, mock_ce_cls, populated_store, fake_embeddings):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.8, 0.7]
        mock_ce_cls.return_value = mock_model

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
