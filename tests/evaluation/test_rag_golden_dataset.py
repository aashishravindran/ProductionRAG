"""Golden Dataset evaluation tests for ProductionRAG.

Two complementary test suites that share the same 30-question dataset:

  TestContextRecall  (marker: retrieval)
    - Calls retrieval.retriever.retrieve() directly
    - Asserts that the ground-truth context is among the top-k chunks
    - Always runs — no Ollama required
    - Fast: ~1-3s per question once the models are warm

  TestEndToEnd  (marker: e2e)
    - Calls retrieval.rag_chain.ask() for the full pipeline
    - Asserts answer semantic similarity, ROUGE-L, faithfulness, and latency
    - Auto-skipped when Ollama is not reachable on localhost:11434

Usage
-----
  # Retrieval only (fast, no Ollama):
  pytest tests/evaluation/ -m retrieval -v

  # Full pipeline (requires Ollama):
  pytest tests/evaluation/ -m e2e -v

  # Everything:
  pytest tests/evaluation/ -v

  # With a short summary table at the end (always printed via conftest):
  pytest tests/evaluation/ -v -s
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from .metrics import (
    answer_similarity as compute_answer_similarity,
    context_recall as compute_context_recall,
    faithfulness as compute_faithfulness,
    rouge_l as compute_rouge_l,
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
GOLDEN_DATASET: List[Dict[str, Any]] = json.loads(_DATASET_PATH.read_text())

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

CONTEXT_RECALL_THRESHOLD = 0.55    # semantic similarity: GT context ↔ best chunk
ANSWER_SIMILARITY_THRESHOLD = 0.50  # semantic similarity: GT answer ↔ LLM answer
ROUGE_L_THRESHOLD = 0.12            # ROUGE-L F1 (generous for generative output)
FAITHFULNESS_THRESHOLD = 0.40       # fraction of key answer tokens in context

# Questions whose cross-document routing is intentionally unsolvable by
# single-document retrieval — marked xfail so failures are documented but
# don't break the CI gate.
_XFAIL_QUESTIONS: frozenset[str] = frozenset({
    # Hard Negative: query text routes to resume (mentions "RAG system in the
    # resume") but GT context comes from Gen_ai_divide.pdf — retriever cannot
    # satisfy both intents simultaneously.
    "The GenAI Divide paper states '5% deploy' for GenAI tools. Does this match the accuracy figure for the RAG system described in the resume, and what does each percentage actually refer to?",
})

# Per-difficulty latency budgets (milliseconds)
LATENCY_BUDGET_MS: Dict[str, float] = {
    "Easy": 20_000,
    "Medium": 25_000,
    "Hard": 35_000,
}

# ---------------------------------------------------------------------------
# Stable, human-readable test IDs
# ---------------------------------------------------------------------------

def _test_id(item: Dict[str, Any]) -> str:
    q = item["question"][:55].replace(" ", "_")
    for ch in "?'\".,;:!()[]{}—–/\\":
        q = q.replace(ch, "")
    return f"{item['difficulty']}__{q}"


_IDS = [_test_id(i) for i in GOLDEN_DATASET]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_or_create_entry(
    eval_results: Dict[str, Any], item: Dict[str, Any]
) -> Dict[str, Any]:
    """Return the existing result entry for this question, or create one."""
    key = item["question"]
    if key not in eval_results:
        eval_results[key] = {
            "question": item["question"],
            "document_source": item["document_source"],
            "difficulty": item["difficulty"],
            "reasoning_type": item["reasoning_type"],
            "stress_test": item.get("stress_test_note"),
            # Metrics — populated by the two test functions
            "context_recall": None,
            "answer_similarity": None,
            "rouge_l": None,
            "faithfulness": None,
            "latency_ms": None,
            "generated_answer": None,
        }
    return eval_results[key]


# ===========================================================================
# Suite 1 — Retrieval / Context-Recall  (always runs, no Ollama needed)
# ===========================================================================

class TestContextRecall:
    """Validates that hybrid retrieval surfaces the correct evidence chunk
    for every question in the Golden Dataset.

    Passes when the best-matching retrieved chunk has cosine similarity
    >= CONTEXT_RECALL_THRESHOLD (0.55) with the ground-truth context.
    """

    @pytest.mark.retrieval
    @pytest.mark.parametrize("item", GOLDEN_DATASET, ids=_IDS)
    def test_context_recall(
        self,
        item: Dict[str, Any],
        real_embeddings,
        eval_results: Dict[str, Any],
    ) -> None:
        if item["question"] in _XFAIL_QUESTIONS:
            pytest.xfail(
                "Known cross-document hard negative: query routes to one document "
                "type but GT context lives in another — single-pass retrieval "
                "cannot satisfy both intents."
            )

        from retrieval.retriever import retrieve

        docs = retrieve(item["question"], embedding_function=real_embeddings)
        score = compute_context_recall(item["ground_truth_context"], docs)

        entry = _get_or_create_entry(eval_results, item)
        entry["context_recall"] = round(score, 4)
        entry["retrieved_chunks"] = len(docs)

        top_chunk = docs[0].page_content[:300] if docs else "NO DOCS RETURNED"
        assert score >= CONTEXT_RECALL_THRESHOLD, (
            f"\n[{item['difficulty']} | {item['document_source']}] "
            f"Context recall {score:.3f} < threshold {CONTEXT_RECALL_THRESHOLD}\n"
            f"Question   : {item['question']}\n"
            f"GT context : {item['ground_truth_context'][:250]}\n"
            f"Top chunk  : {top_chunk}\n"
        )


# ===========================================================================
# Suite 2 — End-to-End Answer Quality  (requires Ollama on localhost:11434)
# ===========================================================================

class TestEndToEnd:
    """Validates the full RAG pipeline (retrieval + LLM generation) against
    the ground-truth answers in the Golden Dataset.

    Each test is auto-skipped when Ollama is not reachable.

    Metrics asserted per question:
      - answer_similarity  >= 0.50
      - rouge_l            >= 0.12
      - faithfulness       >= 0.40
      - latency_ms         <= per-difficulty budget
    """

    @pytest.mark.e2e
    @pytest.mark.parametrize("item", GOLDEN_DATASET, ids=_IDS)
    def test_answer_quality(
        self,
        item: Dict[str, Any],
        real_embeddings,
        ollama_available: bool,
        eval_results: Dict[str, Any],
    ) -> None:
        if not ollama_available:
            pytest.skip("Ollama not reachable on localhost:11434 — skipping E2E test")

        from retrieval.rag_chain import ask

        result = ask(item["question"], embedding_function=real_embeddings)
        generated: str = result["answer"]
        retrieved_docs = result["sources"]
        latency_ms: float = result["response_time_ms"]

        ans_sim = compute_answer_similarity(generated, item["answer"])
        rouge = compute_rouge_l(generated, item["answer"])
        faith = compute_faithfulness(generated, retrieved_docs)

        entry = _get_or_create_entry(eval_results, item)
        entry.update(
            {
                "answer_similarity": round(ans_sim, 4),
                "rouge_l": round(rouge, 4),
                "faithfulness": round(faith, 4),
                "latency_ms": round(latency_ms, 1),
                "generated_answer": generated,
            }
        )

        latency_budget = LATENCY_BUDGET_MS.get(item["difficulty"], 35_000)

        failures: List[str] = []

        if ans_sim < ANSWER_SIMILARITY_THRESHOLD:
            failures.append(
                f"answer_similarity {ans_sim:.3f} < {ANSWER_SIMILARITY_THRESHOLD}\n"
                f"  GT answer : {item['answer']}\n"
                f"  Generated : {generated[:300]}"
            )
        if rouge < ROUGE_L_THRESHOLD:
            failures.append(f"rouge_l {rouge:.3f} < {ROUGE_L_THRESHOLD}")
        if faith < FAITHFULNESS_THRESHOLD:
            failures.append(f"faithfulness {faith:.3f} < {FAITHFULNESS_THRESHOLD}")
        if latency_ms > latency_budget:
            failures.append(
                f"latency {latency_ms:.0f}ms > {latency_budget:.0f}ms budget"
            )

        if failures:
            header = (
                f"\n[{item['difficulty']} | {item['document_source']}] "
                f"{item['reasoning_type']}\n"
                f"Question: {item['question']}\n"
            )
            pytest.fail(header + "\n".join(f"  ✗ {f}" for f in failures))
