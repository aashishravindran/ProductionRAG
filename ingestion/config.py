"""Ingestion pipeline configuration constants."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data_sourcing" / "data"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "chroma_db"
CHROMA_COLLECTION_NAME = "resume_documents"

SOURCE_PDFS = {
    "github_projects_detailed": DATA_DIR / "github_projects_detailed.pdf",
    "resume": DATA_DIR / "Resume.pdf",
    "Gen_ai_divide": DATA_DIR / "Gen_ai_divide.pdf",
}

DOCUMENT_TYPES = {
    "github_projects_detailed": "projects",
    "resume": "resume",
    "Gen_ai_divide": "research",
}

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Per-document-type chunk sizes:
# - projects: each project is a full page with rich sections, use large chunks
#   to keep project descriptions, tech stacks, and features together.
# - resume: smaller chunks so individual bullet points stay isolated and
#   retrieve cleanly without being diluted by surrounding bullets.
# - research: small chunks so specific statistics and facts (e.g. "95% of
#   orgs get zero return") are isolated rather than buried in a 1.5-page
#   block, keeping each chunk's embedding tightly focused on one concept.
CHUNK_SIZES_BY_TYPE = {
    "projects": {"chunk_size": 1000, "chunk_overlap": 200},
    "resume": {"chunk_size": 250, "chunk_overlap": 50},
    "research": {"chunk_size": 600, "chunk_overlap": 150},
}
