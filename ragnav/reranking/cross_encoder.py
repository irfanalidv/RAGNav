"""
Cross-encoder reranking: rescores retrieved blocks by actual query-passage relevance.
Runs after BM25+vector fusion to push truly relevant blocks to the top.
Requires sentence-transformers (same dep as embeddings extra).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..exceptions import RAGNavEmbeddingError
from ..models import Block


@dataclass
class CrossEncoderReranker:
    """
    Lazy-loads a cross-encoder and scores (query, passage) pairs.

    Unlike bi-encoder cosine similarity, a cross-attention model is trained to predict
    relevance directly, which better matches ranking for QA and retrieval benchmarks.
    """

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_length: int = 512
    _model: Optional[object] = field(default=None, init=False, repr=False)

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, max_length=self.max_length)
        except ImportError as e:
            raise RAGNavEmbeddingError(
                "sentence-transformers is required for cross-encoder reranking.\n"
                "Install with: pip install ragnav[embeddings]"
            ) from e

    def rerank_scored(self, query: str, blocks: list[Block]) -> list[tuple[Block, float]]:
        """Return all ``blocks`` ordered by cross-encoder score (highest first)."""
        if not blocks:
            return []
        self._load()
        assert self._model is not None
        pairs = [(query, (b.text or "")[: self.max_length]) for b in blocks]
        raw = self._model.predict(pairs)
        ranked = sorted(
            [(b, float(s)) for b, s in zip(blocks, raw)],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked

    def rerank(self, query: str, blocks: list[Block], *, top_k: int) -> list[Block]:
        """
        Cross-encoder scores each block's text against the query; returns the top_k blocks
        by that relevance score (not by embedding cosine similarity).
        """
        return [b for b, _ in self.rerank_scored(query, blocks)[:top_k]]
