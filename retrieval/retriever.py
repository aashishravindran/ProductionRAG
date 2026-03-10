"""Retrieve relevant document chunks from ChromaDB."""
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ingestion.store import load_vector_store
from ingestion.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
from .config import RETRIEVAL_TOP_K


def get_retriever(
    embedding_function: Embeddings | None = None,
    persist_directory=None,
    collection_name: str = CHROMA_COLLECTION_NAME,
    top_k: int = RETRIEVAL_TOP_K,
):
    """Load ChromaDB and return a LangChain retriever.

    Returns a VectorStoreRetriever with similarity search, top_k results.
    """
    if embedding_function is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    persist_dir = persist_directory or CHROMA_PERSIST_DIR

    store = load_vector_store(
        embedding_function=embedding_function,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )
    return store.as_retriever(search_kwargs={"k": top_k})


def retrieve(
    query: str,
    embedding_function: Embeddings | None = None,
    persist_directory=None,
    collection_name: str = CHROMA_COLLECTION_NAME,
    top_k: int = RETRIEVAL_TOP_K,
) -> list[Document]:
    """Retrieve top-k relevant chunks for a query."""
    retriever = get_retriever(
        embedding_function=embedding_function,
        persist_directory=persist_directory,
        collection_name=collection_name,
        top_k=top_k,
    )
    return retriever.invoke(query)
