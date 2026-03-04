"""Chunk documents using RecursiveCharacterTextSplitter."""
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_SIZE, CHUNK_OVERLAP


def create_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """Create a configured text splitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )


def chunk_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into chunks, preserving metadata from source documents.

    Each output chunk inherits all metadata from its parent document.

    Raises:
        ValueError: If documents list is empty.
    """
    if not documents:
        raise ValueError("Cannot chunk an empty document list")

    splitter = create_splitter(chunk_size, chunk_overlap)
    return splitter.split_documents(documents)
