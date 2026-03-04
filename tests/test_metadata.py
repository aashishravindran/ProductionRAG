"""Tests for metadata enrichment."""
import pytest
from langchain_core.documents import Document
from ingestion.metadata import enrich_metadata, format_citation


class TestEnrichMetadata:
    def test_adds_chunk_index(self):
        chunks = [
            Document(page_content="a", metadata={"document_name": "x", "page": 0}),
            Document(page_content="b", metadata={"document_name": "x", "page": 0}),
        ]
        enriched = enrich_metadata(chunks, document_types={"x": "profile"})
        assert enriched[0].metadata["chunk_index"] == 0
        assert enriched[1].metadata["chunk_index"] == 1

    def test_adds_document_type(self):
        chunks = [
            Document(page_content="a", metadata={"document_name": "github_profile"}),
        ]
        enriched = enrich_metadata(chunks)
        assert enriched[0].metadata["document_type"] == "profile"

    def test_unknown_document_type(self):
        chunks = [
            Document(page_content="a", metadata={"document_name": "mystery_doc"}),
        ]
        enriched = enrich_metadata(chunks)
        assert enriched[0].metadata["document_type"] == "unknown"

    def test_mutates_in_place(self):
        chunks = [Document(page_content="a", metadata={"document_name": "x"})]
        result = enrich_metadata(chunks, document_types={"x": "test"})
        assert result is chunks


class TestFormatCitation:
    def test_full_metadata(self):
        meta = {"source_file": "resume.pdf", "page": 1, "chunk_index": 5}
        assert format_citation(meta) == "resume.pdf, page 2, chunk 5"

    def test_missing_fields(self):
        citation = format_citation({})
        assert "unknown" in citation

    def test_zero_indexed_page_to_one_indexed(self):
        meta = {"source_file": "doc.pdf", "page": 0, "chunk_index": 0}
        assert "page 1" in format_citation(meta)
