"""Data ingestion pipeline for ProductionRAG."""
from .pipeline import run_ingestion
from .loader import load_pdf, load_all_pdfs
from .chunker import chunk_documents
from .metadata import enrich_metadata, format_citation
from .questions import enrich_with_questions
from .store import create_vector_store, load_vector_store

__all__ = [
    "run_ingestion",
    "load_pdf",
    "load_all_pdfs",
    "chunk_documents",
    "enrich_metadata",
    "format_citation",
    "enrich_with_questions",
    "create_vector_store",
    "load_vector_store",
]
