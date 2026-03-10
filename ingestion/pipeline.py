"""Ingestion pipeline orchestrator."""
from pathlib import Path

from langchain_core.embeddings import Embeddings

from .config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    CHUNK_SIZES_BY_TYPE,
    DOCUMENT_TYPES,
    SOURCE_PDFS,
)
from .loader import load_all_pdfs
from .chunker import chunk_documents
from .metadata import enrich_metadata
from .questions import enrich_with_questions
from .store import create_vector_store


def run_ingestion(
    pdf_paths: dict[str, Path] | None = None,
    embedding_function: Embeddings | None = None,
    persist_directory: Path | None = None,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> dict:
    """Execute the full ingestion pipeline: load -> chunk -> enrich -> store.

    Args:
        pdf_paths: Override source PDFs. Defaults to config.SOURCE_PDFS.
        embedding_function: LangChain Embeddings instance. If None, uses
            HuggingFace sentence-transformers all-MiniLM-L6-v2.
        persist_directory: ChromaDB persist path. Defaults to config.CHROMA_PERSIST_DIR.
        collection_name: ChromaDB collection name.

    Returns:
        Dict with documents_loaded, chunks_created, collection_name, persist_directory.
    """
    sources = pdf_paths or SOURCE_PDFS
    persist_dir = persist_directory or CHROMA_PERSIST_DIR

    if embedding_function is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    # Load -> Chunk (per document type) -> Enrich -> Store
    documents = load_all_pdfs(sources)

    # Group documents by type so each group gets appropriate chunk sizes
    docs_by_type: dict[str, list] = {}
    for doc in documents:
        doc_name = doc.metadata.get("document_name", "")
        doc_type = DOCUMENT_TYPES.get(doc_name, "research")
        docs_by_type.setdefault(doc_type, []).append(doc)

    chunks = []
    for doc_type, docs in docs_by_type.items():
        type_config = CHUNK_SIZES_BY_TYPE.get(doc_type, {})
        chunks.extend(chunk_documents(docs, **type_config))

    enriched_chunks = enrich_metadata(chunks)

    # Generate hypothetical questions per chunk to improve retrieval
    print("Generating hypothetical questions for each chunk...")
    enriched_chunks = enrich_with_questions(enriched_chunks)

    create_vector_store(
        documents=enriched_chunks,
        embedding_function=embedding_function,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )

    return {
        "documents_loaded": len(documents),
        "chunks_created": len(enriched_chunks),
        "collection_name": collection_name,
        "persist_directory": str(persist_dir),
    }


if __name__ == "__main__":
    print("Running ingestion pipeline...")
    stats = run_ingestion()
    print(f"Loaded {stats['documents_loaded']} pages")
    print(f"Created {stats['chunks_created']} chunks")
    print(f"Stored in {stats['persist_directory']}")
    print("Done.")
