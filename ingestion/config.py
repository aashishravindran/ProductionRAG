"""Ingestion pipeline configuration constants."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data_sourcing" / "data"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "chroma_db"
CHROMA_COLLECTION_NAME = "resume_documents"

SOURCE_PDFS = {
    "github_profile": DATA_DIR / "github_profile.pdf",
    "linkedin_profile": DATA_DIR / "linkedin_profile.pdf",
    "Gen_ai_divide": DATA_DIR / "Gen_ai_divide.pdf",
}

DOCUMENT_TYPES = {
    "github_profile": "profile",
    "linkedin_profile": "profile",
    "Gen_ai_divide": "research",
}

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
