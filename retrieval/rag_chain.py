"""RAG chain: retrieve context then generate an answer."""
import time

from langchain_core.embeddings import Embeddings

from ingestion.config import CHROMA_COLLECTION_NAME
from .config import RETRIEVAL_TOP_K, OLLAMA_MODEL, OLLAMA_BASE_URL
from .retriever import retrieve
from .generator import generate


def ask(
    query: str,
    embedding_function: Embeddings | None = None,
    persist_directory=None,
    collection_name: str = CHROMA_COLLECTION_NAME,
    top_k: int = RETRIEVAL_TOP_K,
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    skip_analysis: bool = False,
) -> dict:
    """Run the full RAG pipeline: analyze query, retrieve relevant chunks, generate answer.

    Returns dict with keys: answer, sources, response_time_ms.
    """
    t_start = time.perf_counter()

    documents = retrieve(
        query=query,
        embedding_function=embedding_function,
        persist_directory=persist_directory,
        collection_name=collection_name,
        top_k=top_k,
        skip_analysis=skip_analysis,
    )

    answer = generate(
        query=query,
        context_documents=documents,
        model=model,
        base_url=base_url,
    )

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    return {
        "answer": answer,
        "sources": documents,
        "response_time_ms": round(elapsed_ms, 1),
    }


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "What are your key skills?"
    result = ask(query)
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\n--- Retrieved {len(result['sources'])} chunks in {result['response_time_ms']}ms ---")
    for doc in result["sources"]:
        src = doc.metadata.get("source_file", "?")
        print(f"  - {src} (page {doc.metadata.get('page', '?')})")
