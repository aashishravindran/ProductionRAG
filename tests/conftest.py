"""Shared test fixtures for the ingestion pipeline."""
import pytest
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


@pytest.fixture
def sample_documents():
    """Sample LangChain Documents mimicking PyPDFLoader output."""
    return [
        Document(
            page_content="John Doe is a software engineer with 5 years experience in Python and machine learning.",
            metadata={"source": "/fake/path/resume.pdf", "page": 0},
        ),
        Document(
            page_content="Skills include: Python, JavaScript, Docker, Kubernetes, AWS. "
            "Projects include a RAG pipeline and a recommendation engine.",
            metadata={"source": "/fake/path/resume.pdf", "page": 1},
        ),
    ]


@pytest.fixture
def sample_documents_with_names(sample_documents):
    """Sample documents with document_name already set (post-loader)."""
    for doc in sample_documents:
        doc.metadata["document_name"] = "test_profile"
        doc.metadata["source_file"] = "resume.pdf"
    return sample_documents


@pytest.fixture
def sample_pdf_path(tmp_path):
    """Create a minimal valid PDF for integration tests."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(text="Test content for page 1.")
    pdf.add_page()
    pdf.cell(text="Test content for page 2.")
    output_path = tmp_path / "test_doc.pdf"
    pdf.output(str(output_path))
    return output_path


class FakeEmbeddings(Embeddings):
    """Deterministic fake embeddings for testing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 384 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 384


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings()


@pytest.fixture(autouse=True)
def _clear_caches():
    """Clear cached models between tests so mocks take effect."""
    from retrieval.retriever import _get_cross_encoder
    import retrieval.query_analyzer as qa
    _get_cross_encoder.cache_clear()
    qa._cached_prototypes = None
    yield
    _get_cross_encoder.cache_clear()
    qa._cached_prototypes = None


@pytest.fixture
def populated_store(tmp_path, fake_embeddings):
    """A ChromaDB store pre-loaded with sample resume chunks."""
    from ingestion.store import create_vector_store

    docs = [
        Document(
            page_content="John Doe has 5 years of Python experience and built ML pipelines.",
            metadata={"source_file": "resume.pdf", "page": 0, "chunk_index": 0},
        ),
        Document(
            page_content="Skills: Docker, Kubernetes, AWS, CI/CD, MLOps.",
            metadata={"source_file": "resume.pdf", "page": 1, "chunk_index": 1},
        ),
        Document(
            page_content="Education: MSc Computer Science from Stanford University.",
            metadata={"source_file": "resume.pdf", "page": 2, "chunk_index": 2},
        ),
    ]
    persist_dir = tmp_path / "test_chroma"
    create_vector_store(docs, fake_embeddings, persist_dir, "test_resume")
    return {"persist_dir": persist_dir, "collection_name": "test_resume"}
