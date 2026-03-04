"""Tests for document chunking."""
import pytest
from langchain_core.documents import Document
from ingestion.chunker import chunk_documents, create_splitter


class TestCreateSplitter:
    def test_default_parameters(self):
        splitter = create_splitter()
        assert splitter._chunk_size == 500
        assert splitter._chunk_overlap == 100

    def test_custom_parameters(self):
        splitter = create_splitter(chunk_size=200, chunk_overlap=50)
        assert splitter._chunk_size == 200
        assert splitter._chunk_overlap == 50


class TestChunkDocuments:
    def test_short_document_stays_single_chunk(self):
        docs = [Document(page_content="Short text.", metadata={"page": 0, "source": "x.pdf"})]
        chunks = chunk_documents(docs)
        assert len(chunks) == 1
        assert chunks[0].metadata["page"] == 0

    def test_long_document_splits(self):
        long_text = "word " * 200  # ~1000 chars
        docs = [Document(page_content=long_text, metadata={"page": 0, "source": "x.pdf"})]
        chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=100)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.metadata["page"] == 0

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            chunk_documents([])

    def test_metadata_preserved_through_split(self, sample_documents_with_names):
        chunks = chunk_documents(sample_documents_with_names)
        for chunk in chunks:
            assert "document_name" in chunk.metadata
            assert "source_file" in chunk.metadata
            assert "page" in chunk.metadata
