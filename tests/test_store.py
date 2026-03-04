"""Tests for ChromaDB vector store operations."""
import pytest
from langchain_core.documents import Document
from ingestion.store import create_vector_store, load_vector_store


class TestCreateVectorStore:
    def test_creates_store_and_persists(self, tmp_path, fake_embeddings):
        docs = [
            Document(
                page_content="Test content.",
                metadata={"source_file": "test.pdf", "page": 0, "chunk_index": 0},
            ),
        ]
        store = create_vector_store(
            documents=docs,
            embedding_function=fake_embeddings,
            persist_directory=tmp_path / "chroma_test",
            collection_name="test_collection",
        )
        results = store.similarity_search("test", k=1)
        assert len(results) == 1
        assert results[0].metadata["source_file"] == "test.pdf"

    def test_empty_documents_raises(self, tmp_path, fake_embeddings):
        with pytest.raises(ValueError):
            create_vector_store([], fake_embeddings, tmp_path / "empty")

    def test_metadata_survives_round_trip(self, tmp_path, fake_embeddings):
        docs = [
            Document(
                page_content="Some content about engineering.",
                metadata={
                    "source_file": "resume.pdf",
                    "page": 2,
                    "chunk_index": 7,
                    "document_type": "profile",
                    "document_name": "github_profile",
                },
            ),
        ]
        store = create_vector_store(docs, fake_embeddings, tmp_path / "meta_test")
        results = store.similarity_search("content", k=1)
        meta = results[0].metadata
        assert meta["source_file"] == "resume.pdf"
        assert meta["page"] == 2
        assert meta["chunk_index"] == 7
        assert meta["document_type"] == "profile"


class TestLoadVectorStore:
    def test_load_nonexistent_raises(self, tmp_path, fake_embeddings):
        with pytest.raises(FileNotFoundError):
            load_vector_store(fake_embeddings, tmp_path / "no_such_dir")

    def test_load_existing_store(self, tmp_path, fake_embeddings):
        docs = [Document(page_content="Hello world.", metadata={"page": 0})]
        persist_dir = tmp_path / "load_test"
        create_vector_store(docs, fake_embeddings, persist_dir, "test_coll")

        store = load_vector_store(fake_embeddings, persist_dir, "test_coll")
        results = store.similarity_search("hello", k=1)
        assert len(results) == 1
