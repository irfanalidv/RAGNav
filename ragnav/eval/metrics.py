from __future__ import annotations

from dataclasses import dataclass

from ..models import Block
from .cases import EvalCase


@dataclass(frozen=True)
class RetrievalMetrics:
    n_cases: int
    block_hit_rate: float
    page_hit_rate: float
    substring_hit_rate: float
    avg_blocks_returned: float


def _hit_block_ids(case: EvalCase, blocks: list[Block]) -> bool:
    if not case.expected_block_ids:
        return True
    got = {b.block_id for b in blocks}
    return case.expected_block_ids.issubset(got)


def _hit_pages(case: EvalCase, blocks: list[Block]) -> bool:
    if not case.expected_pages:
        return True
    got_pages = set()
    for b in blocks:
        p = b.anchors.get("page")
        if isinstance(p, int):
            got_pages.add(p)
    return case.expected_pages.issubset(got_pages)


def _hit_substrings(case: EvalCase, blocks: list[Block]) -> bool:
    if not case.expected_text_substrings:
        return True
    text = "\n\n".join(b.text or "" for b in blocks)
    return all(s in text for s in case.expected_text_substrings)


def score_retrieval(cases: list[EvalCase], retrieved: list[list[Block]]) -> RetrievalMetrics:
    if len(cases) != len(retrieved):
        raise ValueError("cases/retrieved length mismatch")

    n = len(cases)
    if n == 0:
        return RetrievalMetrics(
            n_cases=0,
            block_hit_rate=0.0,
            page_hit_rate=0.0,
            substring_hit_rate=0.0,
            avg_blocks_returned=0.0,
        )

    block_hits = 0
    page_hits = 0
    substring_hits = 0
    total_blocks = 0

    for c, blocks in zip(cases, retrieved):
        total_blocks += len(blocks)
        if _hit_block_ids(c, blocks):
            block_hits += 1
        if _hit_pages(c, blocks):
            page_hits += 1
        if _hit_substrings(c, blocks):
            substring_hits += 1

    return RetrievalMetrics(
        n_cases=n,
        block_hit_rate=block_hits / n,
        page_hit_rate=page_hits / n,
        substring_hit_rate=substring_hits / n,
        avg_blocks_returned=total_blocks / n,
    )

