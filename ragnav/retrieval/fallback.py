"""
Query variation fallback: retries retrieval with LLM-generated rephrasings when the first
pass is uncertain, so the right passage is less often missed just because wording diverged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from ..llm.base import LLMClient
from ..models import ConfidenceLevel, RetrievalResult

if TYPE_CHECKING:
    from .retriever import RAGNavRetriever

logger = logging.getLogger(__name__)


@dataclass
class FallbackConfig:
    max_attempts: int = 3
    retry_on: set[ConfidenceLevel] = field(
        default_factory=lambda: {ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM}
    )
    variation_prompt: str = (
        "Generate {n} alternative phrasings of this search query. "
        "Each must preserve the original intent but use different words.\n"
        "Query: {query}\n"
        "Return ONLY the variations, one per line, no numbering."
    )


@dataclass
class FallbackResult:
    final_result: RetrievalResult
    attempts: int
    queries_tried: list[str]
    winning_query: str
    improved: bool


class QueryFallback:
    """
    Wraps ``RAGNavRetriever`` so a weak first-pass score triggers LLM-written query variants.

    Production retrieval often fails quietly when the user’s wording does not match the corpus;
    this layer retries with rephrased queries instead of returning low-confidence hits once.
    """

    def __init__(
        self,
        retriever: RAGNavRetriever,
        llm: LLMClient,
        config: Optional[FallbackConfig] = None,
    ):
        self._retriever = retriever
        self._llm = llm
        self._config = config or FallbackConfig()

    def retrieve(self, query: str, **retrieve_kwargs) -> FallbackResult:
        cfg = self._config
        initial = self._retriever.retrieve(query, **retrieve_kwargs)
        best = initial
        winning_query = query
        queries_tried = [query]

        if best.confidence not in cfg.retry_on or cfg.max_attempts <= 1:
            return FallbackResult(best, 1, queries_tried, winning_query, improved=False)

        for attempt in range(2, cfg.max_attempts + 1):
            try:
                variations = self._generate_variations(query, n=1)
            except Exception as exc:
                logger.warning("variation generation failed on attempt %d: %s", attempt, exc)
                break
            if not variations:
                break
            variation = variations[0]
            queries_tried.append(variation)
            candidate = self._retriever.retrieve(variation, **retrieve_kwargs)
            if candidate.top_score > best.top_score:
                best = candidate
                winning_query = variation
                if best.confidence not in cfg.retry_on:
                    return FallbackResult(
                        best,
                        attempt,
                        queries_tried,
                        winning_query,
                        improved=best.top_score > initial.top_score,
                    )

        improved = best.top_score > initial.top_score
        return FallbackResult(best, len(queries_tried), queries_tried, winning_query, improved=improved)

    def _generate_variations(self, query: str, *, n: int = 2) -> list[str]:
        prompt = self._config.variation_prompt.format(query=query, n=n)
        raw = self._llm.chat(messages=[{"role": "user", "content": prompt}], temperature=0.7)
        lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
        return lines[:n]
