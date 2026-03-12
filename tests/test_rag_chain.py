"""Tests for retrieval.rag_chain module."""
from unittest.mock import patch, MagicMock


class TestAsk:
    @patch("retrieval.retriever.CrossEncoder")
    @patch("retrieval.rag_chain.generate")
    def test_returns_answer_and_sources(self, mock_generate, mock_ce_cls, populated_store, fake_embeddings):
        mock_generate.return_value = "Mocked answer."
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.8, 0.7]
        mock_ce_cls.return_value = mock_model

        from retrieval.rag_chain import ask

        result = ask(
            query="What skills?",
            embedding_function=fake_embeddings,
            persist_directory=populated_store["persist_dir"],
            collection_name=populated_store["collection_name"],
        )

        assert "answer" in result
        assert "sources" in result
        assert result["answer"] == "Mocked answer."
        assert len(result["sources"]) > 0

    @patch("retrieval.retriever.CrossEncoder")
    @patch("retrieval.rag_chain.generate")
    def test_passes_retrieved_docs_to_generate(self, mock_generate, mock_ce_cls, populated_store, fake_embeddings):
        mock_generate.return_value = "Answer."
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.8, 0.7]
        mock_ce_cls.return_value = mock_model

        from retrieval.rag_chain import ask

        ask(
            query="Tell me about education",
            embedding_function=fake_embeddings,
            persist_directory=populated_store["persist_dir"],
            collection_name=populated_store["collection_name"],
        )

        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args
        context_docs = call_kwargs.kwargs.get("context_documents") or call_kwargs[1].get("context_documents")
        if context_docs is None:
            context_docs = call_kwargs[0][1]
        assert len(context_docs) > 0
