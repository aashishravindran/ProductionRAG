"""Tests for retrieval.query_analyzer -- embedding-based routing + keyword matching."""
import numpy as np
import pytest
from langchain_core.embeddings import Embeddings

from retrieval.query_analyzer import (
    _detect_resume_section,
    _detect_project_name,
    _cosine_similarity,
    _classify_doc_types,
    analyze_query,
)
from retrieval.retriever import _build_metadata_filter


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        assert _cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 1.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _detect_resume_section
# ---------------------------------------------------------------------------

class TestDetectResumeSection:
    def test_experience_query(self):
        assert _detect_resume_section("What is my work experience?") == "Experience"

    def test_skills_query(self):
        assert _detect_resume_section("What are my skills?") == "Skills"

    def test_education_query(self):
        assert _detect_resume_section("Tell me about my education and degree") == "Education"

    def test_certifications_query(self):
        assert _detect_resume_section("Do I have any certifications?") == "Certifications"

    def test_summary_query(self):
        assert _detect_resume_section("Give me a professional background summary") == "Professional Summary"

    def test_no_match(self):
        assert _detect_resume_section("What color is the sky?") is None

    def test_case_insensitive(self):
        assert _detect_resume_section("SKILLS AND TOOLS") == "Skills"


# ---------------------------------------------------------------------------
# _detect_project_name
# ---------------------------------------------------------------------------

class TestDetectProjectName:
    def test_exact_match(self):
        assert _detect_project_name("Tell me about ProductionRAG") == "ProductionRAG"

    def test_case_insensitive(self):
        assert _detect_project_name("what is productionrag?") == "ProductionRAG"

    def test_hyphenated_name(self):
        assert _detect_project_name("Describe agentic-fitness-app") == "agentic-fitness-app"

    def test_no_match(self):
        assert _detect_project_name("What projects do you have?") is None

    def test_multiple_projects_returns_first(self):
        result = _detect_project_name("Compare ProductionRAG and YelpCamp")
        assert result == "ProductionRAG"


# ---------------------------------------------------------------------------
# _classify_doc_types (with mock embeddings)
# ---------------------------------------------------------------------------

class _DirectionalEmbeddings(Embeddings):
    """Embeddings that map text to distinct vectors based on keywords.

    Resume-related text -> [1, 0, 0]
    Projects-related text -> [0, 1, 0]
    Research-related text -> [0, 0, 1]
    Neutral text -> [0.33, 0.33, 0.33]
    """

    def _vector_for(self, text: str) -> list[float]:
        t = text.lower()
        if any(kw in t for kw in ["experience", "skills", "education", "resume", "career", "job", "work", "certif", "professional", "technologies"]):
            return [1.0, 0.1, 0.1]
        if any(kw in t for kw in ["project", "repo", "github", "architecture", "tech stack", "feature", "application", "design", "built"]):
            return [0.1, 1.0, 0.1]
        if any(kw in t for kw in ["research", "paper", "academic", "methodology", "experiment", "study", "findings", "analysis", "genai", "data"]):
            return [0.1, 0.1, 1.0]
        return [0.33, 0.33, 0.33]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector_for(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector_for(text)


class TestClassifyDocTypes:
    def test_resume_query(self):
        emb = _DirectionalEmbeddings()
        result = _classify_doc_types("What is your work experience?", emb)
        assert "resume" in result

    def test_project_query(self):
        emb = _DirectionalEmbeddings()
        result = _classify_doc_types("Tell me about your GitHub repos", emb)
        assert "projects" in result

    def test_research_query(self):
        emb = _DirectionalEmbeddings()
        result = _classify_doc_types("Describe the research paper findings", emb)
        assert "research" in result


# ---------------------------------------------------------------------------
# analyze_query (integration with keyword matching)
# ---------------------------------------------------------------------------

class TestAnalyzeQuery:
    def test_resume_skills_query(self):
        emb = _DirectionalEmbeddings()
        result = analyze_query("What are my skills?", embeddings=emb)
        assert "resume" in result["doc_types"]
        assert result["resume_section"] == "Skills"

    def test_project_name_detected(self):
        emb = _DirectionalEmbeddings()
        result = analyze_query("Tell me about ProductionRAG", embeddings=emb)
        assert result["project_name"] == "ProductionRAG"
        assert "projects" in result["doc_types"]

    def test_project_name_forces_projects_type(self):
        """Even if embedding says 'resume', project_name ensures 'projects' is included."""
        emb = _DirectionalEmbeddings()
        result = analyze_query("What experience led to ProductionRAG?", embeddings=emb)
        assert result["project_name"] == "ProductionRAG"
        assert "projects" in result["doc_types"]

    def test_resume_section_forces_resume_type(self):
        emb = _DirectionalEmbeddings()
        result = analyze_query("Tell me about education background", embeddings=emb)
        assert result["resume_section"] == "Education"
        assert "resume" in result["doc_types"]

    def test_no_section_no_project(self):
        emb = _DirectionalEmbeddings()
        result = analyze_query("Tell me about your GitHub repos", embeddings=emb)
        assert result["resume_section"] is None
        assert result["project_name"] is None


# ---------------------------------------------------------------------------
# _build_metadata_filter (unchanged, verify still works)
# ---------------------------------------------------------------------------

class TestBuildMetadataFilter:
    def test_doc_types_only(self):
        analysis = {"doc_types": ["resume"], "resume_section": None, "project_name": None}
        result = _build_metadata_filter(analysis)
        assert result == {"document_type": {"$in": ["resume"]}}

    def test_multiple_doc_types(self):
        analysis = {"doc_types": ["resume", "projects"], "resume_section": None, "project_name": None}
        result = _build_metadata_filter(analysis)
        assert result == {"document_type": {"$in": ["resume", "projects"]}}

    def test_doc_types_with_resume_section(self):
        analysis = {"doc_types": ["resume"], "resume_section": "Skills", "project_name": None}
        result = _build_metadata_filter(analysis)
        assert result == {
            "$and": [
                {"document_type": {"$in": ["resume"]}},
                {"resume_section": "Skills"},
            ]
        }

    def test_doc_types_with_project_name(self):
        analysis = {"doc_types": ["projects"], "resume_section": None, "project_name": "ProductionRAG"}
        result = _build_metadata_filter(analysis)
        assert result == {
            "$and": [
                {"document_type": {"$in": ["projects"]}},
                {"project_name": "ProductionRAG"},
            ]
        }

    def test_no_filters_returns_none(self):
        analysis = {"doc_types": None, "resume_section": None, "project_name": None}
        result = _build_metadata_filter(analysis)
        assert result is None

    def test_empty_analysis_returns_none(self):
        result = _build_metadata_filter({})
        assert result is None
