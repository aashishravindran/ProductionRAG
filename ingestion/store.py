"""ChromaDB vector store operations."""
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def create_vector_store(
    documents: list[Document],
    embedding_function: Embeddings,
    persist_directory: str | Path,
    collection_name: str = "resume_documents",
) -> Chroma:
    """Create a ChromaDB vector store from documents and persist to disk.

    Raises:
        ValueError: If documents list is empty.
    """
    if not documents:
        raise ValueError("Cannot create vector store from empty document list")

    return Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=str(Path(persist_directory)),
        collection_name=collection_name,
    )


def load_vector_store(
    embedding_function: Embeddings,
    persist_directory: str | Path,
    collection_name: str = "resume_documents",
) -> Chroma:
    """Load an existing ChromaDB vector store from disk.

    Raises:
        FileNotFoundError: If persist_directory does not exist.
    """
    persist_dir = Path(persist_directory)
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found: {persist_dir}. Run the ingestion pipeline first."
        )

    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embedding_function,
        collection_name=collection_name,
    )
