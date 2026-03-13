"""Tests for metadata enrichment."""
import pytest
from langchain_core.documents import Document
from ingestion.metadata import enrich_metadata, format_citation, _detect_resume_section


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
            Document(page_content="a", metadata={"document_name": "github_projects_detailed", "page": 0}),
        ]
        enriched = enrich_metadata(chunks)
        assert enriched[0].metadata["document_type"] == "projects"

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


class TestProjectMetadata:
    def test_adds_project_name_from_page(self):
        """Projects PDF page 2 = first project (ProductionRAG)."""
        chunks = [
            Document(
                page_content="ProductionRAG content",
                metadata={"document_name": "github_projects_detailed", "page": 2},
            ),
        ]
        enriched = enrich_metadata(chunks)
        assert enriched[0].metadata["project_name"] == "ProductionRAG"
        assert enriched[0].metadata["project_category"] == "GenAI / RAG / Production AI"
        assert enriched[0].metadata["project_language"] == "Python"

    def test_cover_page_has_no_project_metadata(self):
        """Page 0 (cover) should not get project-specific metadata."""
        chunks = [
            Document(
                page_content="Cover page",
                metadata={"document_name": "github_projects_detailed", "page": 0},
            ),
        ]
        enriched = enrich_metadata(chunks)
        assert "project_name" not in enriched[0].metadata

    def test_multiple_chunks_same_project_page(self):
        """Multiple chunks from the same project page share project metadata."""
        chunks = [
            Document(page_content="chunk 1", metadata={"document_name": "github_projects_detailed", "page": 3}),
            Document(page_content="chunk 2", metadata={"document_name": "github_projects_detailed", "page": 3}),
        ]
        enriched = enrich_metadata(chunks)
        assert enriched[0].metadata["project_name"] == enriched[1].metadata["project_name"]
        assert enriched[0].metadata["project_name"] == "agentic-fitness-app"


class TestResumeMetadata:
    def test_detects_experience_section(self):
        chunks = [
            Document(
                page_content="Experience\nSoftware Development Engineer II - AWS",
                metadata={"document_name": "resume", "page": 0},
            ),
        ]
        enriched = enrich_metadata(chunks)
        assert enriched[0].metadata["resume_section"] == "Experience"

    def test_detects_education_section(self):
        chunks = [
            Document(
                page_content="Education\nStony Brook University, Master of Science",
                metadata={"document_name": "resume", "page": 0},
            ),
        ]
        enriched = enrich_metadata(chunks)
        assert enriched[0].metadata["resume_section"] == "Education"

    def test_detects_skills_section(self):
        chunks = [
            Document(
                page_content="Skills\nPython, Java, DynamoDB, Lambda",
                metadata={"document_name": "resume", "page": 0},
            ),
        ]
        enriched = enrich_metadata(chunks)
        assert enriched[0].metadata["resume_section"] == "Skills"

    def test_no_section_header_returns_general(self):
        chunks = [
            Document(
                page_content="Built a scalable dataplane using DynamoDB",
                metadata={"document_name": "resume", "page": 0},
            ),
        ]
        enriched = enrich_metadata(chunks)
        assert enriched[0].metadata["resume_section"] == "General"

    def test_section_propagates_to_following_chunks(self):
        """Chunks after a section header inherit that section."""
        chunks = [
            Document(
                page_content="Experience\nSDE II at AWS",
                metadata={"document_name": "resume", "page": 0},
            ),
            Document(
                page_content="Built a scalable dataplane using DynamoDB",
                metadata={"document_name": "resume", "page": 0},
            ),
            Document(
                page_content="Reduced onboarding time by 80%",
                metadata={"document_name": "resume", "page": 0},
            ),
        ]
        enriched = enrich_metadata(chunks)
        assert enriched[0].metadata["resume_section"] == "Experience"
        assert enriched[1].metadata["resume_section"] == "Experience"
        assert enriched[2].metadata["resume_section"] == "Experience"

    def test_section_changes_on_new_header(self):
        chunks = [
            Document(
                page_content="Experience\nSDE II at AWS",
                metadata={"document_name": "resume", "page": 0},
            ),
            Document(
                page_content="Education\nStony Brook University",
                metadata={"document_name": "resume", "page": 0},
            ),
            Document(
                page_content="GPA 3.8, graduated 2020",
                metadata={"document_name": "resume", "page": 0},
            ),
        ]
        enriched = enrich_metadata(chunks)
        assert enriched[0].metadata["resume_section"] == "Experience"
        assert enriched[1].metadata["resume_section"] == "Education"
        assert enriched[2].metadata["resume_section"] == "Education"


class TestDetectResumeSection:
    def test_case_insensitive(self):
        assert _detect_resume_section("EXPERIENCE\nSDE II at AWS") == "Experience"

    def test_last_section_wins(self):
        text = "Professional Summary\nBlah blah\nExperience\nSDE at AWS"
        assert _detect_resume_section(text) == "Experience"

    def test_empty_text(self):
        assert _detect_resume_section("") == "General"


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

    def test_citation_includes_project_name(self):
        meta = {
            "source_file": "github_projects_detailed.pdf",
            "page": 2,
            "chunk_index": 3,
            "project_name": "ProductionRAG",
        }
        citation = format_citation(meta)
        assert "[ProductionRAG]" in citation

    def test_citation_includes_resume_section(self):
        meta = {
            "source_file": "Resume.pdf",
            "page": 0,
            "chunk_index": 5,
            "resume_section": "Experience",
        }
        citation = format_citation(meta)
        assert "[Experience]" in citation

    def test_citation_skips_general_section(self):
        meta = {
            "source_file": "Resume.pdf",
            "page": 0,
            "chunk_index": 5,
            "resume_section": "General",
        }
        citation = format_citation(meta)
        assert "[" not in citation
