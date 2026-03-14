"""Fixtures for the Golden Dataset evaluation test suite.

Key design decisions:
- _clear_caches is overridden as a no-op so the CrossEncoder and
  sentence-transformer models stay warm across all 60 evaluation tests.
- real_embeddings is session-scoped to load the HuggingFace model once.
- eval_results is a session-scoped dict keyed by question; both the
  context-recall and E2E test functions update the same entry per question.
  At session teardown the fixture writes reports/latest_run.json and
  prints a summary table to the terminal.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import pytest

REPORTS_DIR = Path(__file__).parent / "reports"


# ---------------------------------------------------------------------------
# Override the parent conftest's autouse _clear_caches so the CrossEncoder
# is NOT evicted between every evaluation test (saves ~2s × 60 tests).
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clear_caches():
    """No-op override: keep model caches warm across evaluation tests."""
    yield


# ---------------------------------------------------------------------------
# Real embeddings (session-scoped — loaded once for all 60 tests)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def real_embeddings():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# Ollama availability (session-scoped — checked once)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def ollama_available() -> bool:
    import requests
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Eval results collector + report writer
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def eval_results() -> Dict[str, Any]:
    """Session-scoped dict keyed by question string.

    Each entry is a result dict.  Both test functions (context-recall and
    E2E) update the same entry for a given question.
    At teardown, writes reports/latest_run.json and prints a summary table.
    """
    results: Dict[str, Any] = {}
    yield results
    _write_reports(results)


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def _write_reports(results: Dict[str, Any]) -> None:
    if not results:
        return
    REPORTS_DIR.mkdir(exist_ok=True)
    records = list(results.values())

    report_path = REPORTS_DIR / "latest_run.json"
    with open(report_path, "w") as f:
        json.dump(records, f, indent=2)

    _print_summary(records)
    print(f"\n  Full report: {report_path}\n")


def _avg(items: list, key: str) -> float:
    vals = [i[key] for i in items if i.get(key) is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def _print_summary(records: list) -> None:
    by_source: dict = defaultdict(list)
    by_difficulty: dict = defaultdict(list)
    by_type: dict = defaultdict(list)
    for r in records:
        by_source[r.get("document_source", "unknown")].append(r)
        by_difficulty[r.get("difficulty", "unknown")].append(r)
        by_type[r.get("reasoning_type", "unknown")].append(r)

    W = 95
    sep = "─" * W

    print("\n\n" + "═" * W)
    print("  RAG GOLDEN DATASET — EVALUATION REPORT")
    print("═" * W)
    print(f"  Questions evaluated : {len(records)}")
    e2e_count = sum(1 for r in records if r.get("answer_similarity") is not None)
    print(f"  E2E (Ollama) tested : {e2e_count}")

    # --- Per document source ---
    print(f"\n  {'Document':<44} {'N':>3}  {'CtxRecall':>10}  {'AnsSim':>8}  {'ROUGE-L':>8}  {'Faithful':>9}")
    print(f"  {sep}")
    for src in sorted(by_source):
        items = by_source[src]
        print(
            f"  {src:<44} {len(items):>3}  "
            f"{_avg(items, 'context_recall'):>10.3f}  "
            f"{_avg(items, 'answer_similarity'):>8.3f}  "
            f"{_avg(items, 'rouge_l'):>8.3f}  "
            f"{_avg(items, 'faithfulness'):>9.3f}"
        )
    print(f"  {sep}")
    print(
        f"  {'OVERALL':<44} {len(records):>3}  "
        f"{_avg(records, 'context_recall'):>10.3f}  "
        f"{_avg(records, 'answer_similarity'):>8.3f}  "
        f"{_avg(records, 'rouge_l'):>8.3f}  "
        f"{_avg(records, 'faithfulness'):>9.3f}"
    )

    # --- Per difficulty ---
    print(f"\n  {'Difficulty':<18} {'N':>3}  {'CtxRecall':>10}  {'AnsSim':>8}  {'ROUGE-L':>8}  {'Faithful':>9}  {'Latency(ms)':>12}")
    print(f"  {sep}")
    for diff in ("Easy", "Medium", "Hard"):
        items = by_difficulty.get(diff, [])
        if not items:
            continue
        print(
            f"  {diff:<18} {len(items):>3}  "
            f"{_avg(items, 'context_recall'):>10.3f}  "
            f"{_avg(items, 'answer_similarity'):>8.3f}  "
            f"{_avg(items, 'rouge_l'):>8.3f}  "
            f"{_avg(items, 'faithfulness'):>9.3f}  "
            f"{_avg(items, 'latency_ms'):>12.0f}"
        )

    # --- Per reasoning type ---
    print(f"\n  {'Reasoning Type':<18} {'N':>3}  {'CtxRecall':>10}  {'AnsSim':>8}  {'ROUGE-L':>8}  {'Faithful':>9}")
    print(f"  {sep}")
    for rtype in ("Direct", "Multi-hop", "Synthesized"):
        items = by_type.get(rtype, [])
        if not items:
            continue
        print(
            f"  {rtype:<18} {len(items):>3}  "
            f"{_avg(items, 'context_recall'):>10.3f}  "
            f"{_avg(items, 'answer_similarity'):>8.3f}  "
            f"{_avg(items, 'rouge_l'):>8.3f}  "
            f"{_avg(items, 'faithfulness'):>9.3f}"
        )

    print("═" * W + "\n")
