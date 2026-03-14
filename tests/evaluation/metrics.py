"""Evaluation metrics for the RAG Golden Dataset test suite.

All metrics are self-contained — no LLM judge required.
The sentence-transformer model is loaded once per process and cached.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Shared embedding model (loaded once, cached for the entire test session)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_st_model():
    """Lazy-load all-MiniLM-L6-v2 (already on disk from the main pipeline)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-10 else 0.0


# ---------------------------------------------------------------------------
# Metric 1 — Context Recall
# ---------------------------------------------------------------------------

def context_recall(ground_truth_context: str, docs: Sequence[Document]) -> float:
    """Max cosine similarity between the ground-truth context and any retrieved chunk.

    Score range: [0.0, 1.0].  A score >= 0.55 means the retriever surfaced
    a chunk that semantically contains the answer evidence.

    Uses original_content metadata when present (strips HyDE-prepended questions
    so the comparison is between raw document text and the ground-truth context,
    not between HyDE questions + document text and the ground-truth context).
    """
    if not docs:
        return 0.0
    model = _get_st_model()
    gt_emb = model.encode(ground_truth_context, convert_to_numpy=True)
    texts = [
        d.metadata.get("original_content") or d.page_content for d in docs
    ]
    chunk_embs = model.encode(texts, convert_to_numpy=True, batch_size=32)
    return float(max(_cosine(gt_emb, c) for c in chunk_embs))


# ---------------------------------------------------------------------------
# Metric 2 — Answer Semantic Similarity
# ---------------------------------------------------------------------------

def answer_similarity(generated: str, ground_truth: str) -> float:
    """Cosine similarity between the generated answer and the ground-truth answer.

    Tolerates paraphrasing — only penalises answers that are semantically off-topic.
    Score range: [0.0, 1.0].  Threshold: >= 0.50.
    """
    model = _get_st_model()
    embs = model.encode([generated, ground_truth], convert_to_numpy=True)
    return _cosine(embs[0], embs[1])


# ---------------------------------------------------------------------------
# Metric 3 — ROUGE-L F1 (token-level Longest Common Subsequence)
# ---------------------------------------------------------------------------

def rouge_l(hypothesis: str, reference: str) -> float:
    """ROUGE-L F1 at the token level — measures lexical overlap.

    Implemented from scratch to avoid the rouge-score package dependency.
    Score range: [0.0, 1.0].  Threshold: >= 0.12 (generous for generative output).
    """
    hyp = hypothesis.lower().split()
    ref = reference.lower().split()
    if not hyp or not ref:
        return 0.0

    m, n = len(ref), len(hyp)
    # DP for LCS length using two rows to save memory
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            curr[j] = (
                prev[j - 1] + 1
                if ref[i - 1] == hyp[j - 1]
                else max(prev[j], curr[j - 1])
            )
        prev = curr

    lcs = prev[n]
    precision = lcs / n
    recall = lcs / m
    denom = precision + recall
    return 2 * precision * recall / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Metric 4 — Faithfulness (answer grounded in retrieved context)
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "a an the is are was were in of to and or for with on at by from "
    "that this it as be has have had do does did will would not no but "
    "if than then so yet both either each also its which who what how "
    "when where why just can could should would may might must".split()
)


def faithfulness(answer: str, docs: Sequence[Document]) -> float:
    """Fraction of significant answer tokens found somewhere in the retrieved context.

    'Significant' means length > 3 and not in the stopword list.
    Score range: [0.0, 1.0].  Threshold: >= 0.40.
    A low score suggests the LLM hallucinated content absent from the context.
    """
    if not docs:
        return 0.0
    context_text = " ".join(d.page_content for d in docs).lower()
    tokens = [
        t.strip(".,;:!?\"'()[]{}—–")
        for t in answer.split()
    ]
    significant = [
        t.lower() for t in tokens
        if len(t) > 3 and t.lower() not in _STOPWORDS
    ]
    if not significant:
        return 1.0  # Nothing to verify — conservatively pass
    found = sum(1 for t in significant if t in context_text)
    return found / len(significant)
