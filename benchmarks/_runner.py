"""
Shared retrieval evaluation harness.
No ragnav imports — pure metrics so any retriever can be plugged in.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievalScore:
    method: str
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr_at_10: float
    n_queries: int

    def row(self) -> str:
        return (
            f"{self.method:<30} "
            f"{self.recall_at_1:.3f}  "
            f"{self.recall_at_3:.3f}  "
            f"{self.recall_at_5:.3f}  "
            f"{self.mrr_at_10:.3f}  "
            f"n={self.n_queries}"
        )


def recall_at_k(ranked_ids: list[str], gold_id: str, k: int) -> float:
    return 1.0 if gold_id in ranked_ids[:k] else 0.0


def mrr_at_k(ranked_ids: list[str], gold_id: str, k: int) -> float:
    for i, rid in enumerate(ranked_ids[:k], 1):
        if rid == gold_id:
            return 1.0 / i
    return 0.0


def recall_at_k_any(ranked_ids: list[str], gold_ids: frozenset[str], k: int) -> float:
    head = ranked_ids[:k]
    return 1.0 if any(g in head for g in gold_ids) else 0.0


def mrr_at_k_any(ranked_ids: list[str], gold_ids: frozenset[str], k: int) -> float:
    for i, rid in enumerate(ranked_ids[:k], 1):
        if rid in gold_ids:
            return 1.0 / i
    return 0.0


def score_any_gold(method: str, results: list[tuple[list[str], frozenset[str]]]) -> RetrievalScore:
    """
    results: list of (ranked_block_ids, gold_block_ids)
    A query counts as correct if any gold id appears in the ranked list (set semantics).
    """
    r1 = statistics.mean(recall_at_k_any(r, g, 1) for r, g in results)
    r3 = statistics.mean(recall_at_k_any(r, g, 3) for r, g in results)
    r5 = statistics.mean(recall_at_k_any(r, g, 5) for r, g in results)
    mrr = statistics.mean(mrr_at_k_any(r, g, 10) for r, g in results)
    return RetrievalScore(method, r1, r3, r5, mrr, len(results))


def span_recall_at_k_blocks(blocks: list[Any], gold_texts: list[str], k: int) -> float:
    """
    True if any gold answer string appears inside the concatenation of the first ``k``
    blocks' text (case-insensitive). Suited to clause QA where the span may cross blocks.
    """
    combined = " ".join((getattr(b, "text", "") or "") for b in blocks[:k]).lower()
    return 1.0 if any(t.lower() in combined for t in gold_texts if t.strip()) else 0.0


def span_mrr_at_k_blocks(blocks: list[Any], gold_texts: list[str], k: int) -> float:
    """First prefix length (1..k) whose concatenation contains a gold span → 1/rank."""
    acc = ""
    for i, b in enumerate(blocks[:k], 1):
        acc = (acc + " " + (getattr(b, "text", "") or "")).lower()
        if any(t.lower() in acc for t in gold_texts if t.strip()):
            return 1.0 / i
    return 0.0


def score_span_recall(method: str, rows: list[tuple[list[Any], list[str]]]) -> RetrievalScore:
    """Mean span-based recall / MRR over (ordered retrieved blocks, gold answer strings)."""
    r1 = statistics.mean(span_recall_at_k_blocks(b, t, 1) for b, t in rows)
    r3 = statistics.mean(span_recall_at_k_blocks(b, t, 3) for b, t in rows)
    r5 = statistics.mean(span_recall_at_k_blocks(b, t, 5) for b, t in rows)
    mrr = statistics.mean(span_mrr_at_k_blocks(b, t, 10) for b, t in rows)
    return RetrievalScore(method, r1, r3, r5, mrr, len(rows))


def score(method: str, results: list[tuple[list[str], str]]) -> RetrievalScore:
    """
    results: list of (ranked_block_ids, gold_block_id)
    gold_block_id is the id of the block that contains the answer.
    """
    r1 = statistics.mean(recall_at_k(r, g, 1) for r, g in results)
    r3 = statistics.mean(recall_at_k(r, g, 3) for r, g in results)
    r5 = statistics.mean(recall_at_k(r, g, 5) for r, g in results)
    mrr = statistics.mean(mrr_at_k(r, g, 10) for r, g in results)
    return RetrievalScore(method, r1, r3, r5, mrr, len(results))


def print_table(scores: list[RetrievalScore]) -> None:
    header = f"{'Method':<30} {'R@1':>5}  {'R@3':>5}  {'R@5':>5}  {'MRR@10':>7}"
    print(header)
    print("─" * len(header))
    for s in scores:
        print(s.row())
