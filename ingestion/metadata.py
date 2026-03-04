"""Enrich chunk metadata with document type and chunk indexing."""
from langchain_core.documents import Document

from .config import DOCUMENT_TYPES


def enrich_metadata(
    chunks: list[Document],
    document_types: dict[str, str] | None = None,
) -> list[Document]:
    """Add document_type and chunk_index to each chunk's metadata.

    Final metadata after enrichment:
        source, source_file, page, document_name, document_type, chunk_index
    """
    type_map = document_types or DOCUMENT_TYPES

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        doc_name = chunk.metadata.get("document_name", "")
        chunk.metadata["document_type"] = type_map.get(doc_name, "unknown")

    return chunks


def format_citation(metadata: dict) -> str:
    """Format chunk metadata into a human-readable citation.

    Example: "github_profile.pdf, page 2, chunk 5"
    """
    source_file = metadata.get("source_file", "unknown")
    page = metadata.get("page", "?")
    chunk_index = metadata.get("chunk_index", "?")
    display_page = page + 1 if isinstance(page, int) else page
    return f"{source_file}, page {display_page}, chunk {chunk_index}"
