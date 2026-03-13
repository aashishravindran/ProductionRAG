"""Hybrid retrieval: BM25 + vector search with RRF fusion and cross-encoder reranking."""
from functools import lru_cache

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ingestion.store import load_vector_store
from ingestion.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
from .config import (
    RETRIEVAL_TOP_K,
    BM25_TOP_K,
    VECTOR_TOP_K,
    RRF_K,
    RERANKER_MODEL,
)


# ---------------------------------------------------------------------------
# BM25 retriever
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "a an and are as at be by for from has have he her his how i in is it its"
    " me my no not of on or our she that the their them they this to was we"
    " what when where which who will with you your".split()
)


def _tokenize(text: str) -> list[str]:
    """Whitespace + lowercase tokenizer with stopword removal for BM25."""
    return [w for w in text.lower().split() if w not in _STOPWORDS]


def bm25_search(
    query: str,
    documents: list[Document],
    top_k: int = BM25_TOP_K,
) -> list[Document]:
    """Rank documents by BM25 keyword relevance."""
    if not documents:
        return []
    corpus = [_tokenize(doc.metadata.get("original_content", doc.page_content)) for doc in documents]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [documents[idx] for idx, _ in ranked[:top_k]]


# ---------------------------------------------------------------------------
# Vector retriever
# ---------------------------------------------------------------------------

def vector_search(
    query: str,
    embedding_function: Embeddings,
    persist_directory=None,
    collection_name: str = CHROMA_COLLECTION_NAME,
    top_k: int = VECTOR_TOP_K,
    metadata_filter: dict | None = None,
) -> list[Document]:
    """Retrieve documents by vector similarity from ChromaDB.

    Args:
        metadata_filter: ChromaDB where clause, e.g.
            {"document_type": {"$in": ["resume", "projects"]}}
    """
    persist_dir = persist_directory or CHROMA_PERSIST_DIR
    store = load_vector_store(
        embedding_function=embedding_function,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )
    kwargs = {"k": top_k}
    if metadata_filter:
        kwargs["filter"] = metadata_filter
    return store.similarity_search(query, **kwargs)


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _doc_id(doc: Document) -> str:
    """Create a stable ID for deduplication across result lists."""
    return f"{doc.metadata.get('source_file', '')}:{doc.metadata.get('chunk_index', '')}"


def reciprocal_rank_fusion(
    *result_lists: list[Document],
    k: int = RRF_K,
) -> list[Document]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF score for a document = sum(1 / (k + rank_i)) across all lists
    where rank_i is the 1-based rank in list i.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for results in result_lists:
        for rank, doc in enumerate(results, 1):
            did = _doc_id(doc)
            scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank)
            doc_map[did] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[did] for did, _ in ranked]


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_cross_encoder(model_name: str) -> CrossEncoder:
    """Load and cache a CrossEncoder model (loaded once, reused across calls)."""
    return CrossEncoder(model_name)


def rerank(
    query: str,
    documents: list[Document],
    top_k: int = RETRIEVAL_TOP_K,
    model_name: str = RERANKER_MODEL,
) -> list[Document]:
    """Re-score documents using a cross-encoder model and return top_k."""
    if not documents:
        return []

    model = _get_cross_encoder(model_name)
    contents = [doc.metadata.get("original_content", doc.page_content) for doc in documents]
    pairs = [[query, content] for content in contents]
    scores = model.predict(pairs)

    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _load_all_chunks(
    embedding_function: Embeddings,
    persist_directory=None,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> list[Document]:
    """Load all documents from ChromaDB for BM25 indexing."""
    persist_dir = persist_directory or CHROMA_PERSIST_DIR
    store = load_vector_store(
        embedding_function=embedding_function,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )
    collection = store._collection
    result = collection.get(include=["documents", "metadatas"])
    return [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(result["documents"], result["metadatas"])
    ]


def _build_metadata_filter(analysis: dict) -> dict | None:
    """Convert query analysis into a ChromaDB where clause."""
    conditions = []

    doc_types = analysis.get("doc_types")
    if doc_types:
        conditions.append({"document_type": {"$in": doc_types}})

    resume_section = analysis.get("resume_section")
    if resume_section:
        conditions.append({"resume_section": resume_section})

    project_name = analysis.get("project_name")
    if project_name:
        conditions.append({"project_name": project_name})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def retrieve(
    query: str,
    embedding_function: Embeddings | None = None,
    persist_directory=None,
    collection_name: str = CHROMA_COLLECTION_NAME,
    top_k: int = RETRIEVAL_TOP_K,
    skip_analysis: bool = False,
) -> list[Document]:
    """Hybrid retrieve: query analysis, BM25 + vector search, RRF fusion, cross-encoder rerank.

    Args:
        skip_analysis: If True, skip LLM query analysis and search all docs.
            Useful for tests or when filters are not needed.
    """
    if embedding_function is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    # Step 1: Analyze query to determine doc type filters
    analysis = {}
    metadata_filter = None
    if not skip_analysis:
        from .query_analyzer import analyze_query

        analysis = analyze_query(query)
        metadata_filter = _build_metadata_filter(analysis)

    # Step 2: Load all chunks for BM25 (filtered by doc type)
    all_chunks = _load_all_chunks(
        embedding_function=embedding_function,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    # Apply doc_type filter to BM25 candidate pool
    doc_types = analysis.get("doc_types")
    if doc_types:
        all_chunks = [
            c for c in all_chunks
            if c.metadata.get("document_type") in doc_types
        ]

    # Step 3: Run both retrievers
    bm25_results = bm25_search(query, all_chunks)
    vector_results = vector_search(
        query,
        embedding_function=embedding_function,
        persist_directory=persist_directory,
        collection_name=collection_name,
        metadata_filter=metadata_filter,
    )

    # Step 4: Fuse with RRF
    fused = reciprocal_rank_fusion(bm25_results, vector_results)

    # Step 5: Rerank with cross-encoder
    return rerank(query, fused, top_k=top_k)


def get_retriever(
    embedding_function: Embeddings | None = None,
    persist_directory=None,
    collection_name: str = CHROMA_COLLECTION_NAME,
    top_k: int = RETRIEVAL_TOP_K,
):
    """Load ChromaDB and return a LangChain retriever (vector-only, for compatibility)."""
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
