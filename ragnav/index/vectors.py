from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ..exceptions import RAGNavEmbeddingError
from ..models import Block

if TYPE_CHECKING:
    from ..cache.sqlite_cache import EmbeddingCache
    from ..observability import Trace

_LOG = logging.getLogger(__name__)

_ST_MODELS: dict[str, Any] = {}


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(mat, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return mat / denom


def _get_sentence_transformer(model_name: str) -> Any:
    if model_name not in _ST_MODELS:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RAGNavEmbeddingError(
                "sentence-transformers is required for embedding-based retrieval with the default local model.\n"
                "Install with: pip install ragnav[embeddings]"
            ) from e
        _ST_MODELS[model_name] = SentenceTransformer(model_name)
    return _ST_MODELS[model_name]


def _encode_texts_sentence_transformer(texts: list[str], model_name: str) -> np.ndarray:
    model = _get_sentence_transformer(model_name)
    out = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(out, dtype=np.float32)


@dataclass
class VectorIndex:
    """
    Dense retrieval over block texts: either sentence-transformers (default) or caller-supplied embeddings (e.g. LLM).
    """

    blocks: list[Block] = field(default_factory=list)
    block_doc_ids: list[str] = field(default_factory=list)
    vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 1), dtype=np.float32))
    model_name: Optional[str] = None
    _backend: str = field(default="external", repr=False)

    @property
    def has_fitted(self) -> bool:
        return self.vectors.size > 0 and self.vectors.shape[0] > 0

    @property
    def uses_sentence_transformers(self) -> bool:
        return self._backend == "sentence_transformers"

    def build(
        self,
        texts: list[str],
        ids: list[str],
        *,
        blocks: Optional[list[Block]] = None,
        embedding_cache: Optional[EmbeddingCache] = None,
        embed_batch_size: int = 64,
        trace: Optional[Trace] = None,
    ) -> None:
        """Embed ``texts`` with sentence-transformers and store rows aligned with ``ids`` / ``blocks``."""
        if len(texts) != len(ids):
            raise RAGNavEmbeddingError(
                "VectorIndex.build: texts and ids must have the same length. "
                "Pass one id per text chunk."
            )
        if blocks is not None and len(blocks) != len(texts):
            raise RAGNavEmbeddingError(
                "VectorIndex.build: blocks must be the same length as texts when provided."
            )
        if not texts:
            self.blocks = blocks or []
            self.block_doc_ids = [b.doc_id for b in self.blocks]
            self.vectors = np.zeros((0, 1), dtype=np.float32)
            self._backend = "sentence_transformers"
            return

        mname = self.model_name or "all-MiniLM-L6-v2"
        self.model_name = mname
        embeddings: list[Optional[list[float]]] = [None] * len(texts)

        for i in range(0, len(texts), embed_batch_size):
            chunk = texts[i : i + embed_batch_size]
            cached: dict[int, list[float]] = {}
            if embedding_cache is not None:
                cached = embedding_cache.get_many(model=mname, texts=chunk)
                if trace:
                    trace.incr("embed_cache_hits", len(cached))

            missing_texts: list[str] = []
            missing_pos: list[int] = []
            for j, t in enumerate(chunk):
                if j in cached:
                    embeddings[i + j] = cached[j]
                else:
                    missing_texts.append(t)
                    missing_pos.append(i + j)

            if missing_texts:
                if trace:
                    with trace.span("st_embed_ms"):
                        mat = _encode_texts_sentence_transformer(missing_texts, mname)
                else:
                    mat = _encode_texts_sentence_transformer(missing_texts, mname)
                for row, pos in zip(mat, missing_pos):
                    embeddings[pos] = row.tolist()
                if embedding_cache is not None:
                    embedding_cache.set_many(
                        model=mname,
                        texts=missing_texts,
                        embeddings=[row.tolist() for row in mat],
                    )
                if trace:
                    trace.incr("st_embed_texts", len(missing_texts))

        if any(e is None for e in embeddings):
            raise RAGNavEmbeddingError(
                "Embedding build failed: some texts have no vector after encoding. "
                "Retry with a smaller batch or check disk/memory."
            )
        mat = np.asarray(embeddings, dtype=np.float32)
        if mat.ndim != 2:
            raise RAGNavEmbeddingError(
                "Embeddings must form a 2D matrix (one row per text). Check sentence-transformers output."
            )
        self.vectors = _l2_normalize(mat)
        self.blocks = blocks if blocks is not None else []
        self.block_doc_ids = [b.doc_id for b in self.blocks] if self.blocks else []
        self._backend = "sentence_transformers"

    def query(self, query_text: str, *, top_k: int = 20) -> list[tuple[str, float]]:
        """Return ``(block_id, cosine similarity)`` for the top ``top_k`` ids (sentence-transformers backend only)."""
        if not self.has_fitted:
            return []
        if not self.uses_sentence_transformers or not self.model_name:
            raise RAGNavEmbeddingError(
                "VectorIndex.query(text) only applies when the index was built with sentence-transformers. "
                "Use search() with a query embedding from your LLM instead."
            )
        if not self.blocks or len(self.blocks) != self.vectors.shape[0]:
            raise RAGNavEmbeddingError(
                "VectorIndex.query requires blocks aligned with each row; rebuild the index with blocks=..."
            )
        q = _encode_texts_sentence_transformer([query_text], self.model_name)
        q = _l2_normalize(q)
        sims = (self.vectors @ q.T).reshape(-1)
        idxs = np.argsort(-sims)[:top_k]
        return [(self.blocks[int(i)].block_id, float(sims[int(i)])) for i in idxs]

    @classmethod
    def from_embeddings(cls, blocks: list[Block], embeddings: list[list[float]]) -> VectorIndex:
        """Build from precomputed vectors (e.g. Mistral / FakeLLM ``embed`` output)."""
        if len(blocks) != len(embeddings):
            raise RAGNavEmbeddingError(
                "blocks and embeddings length mismatch. Pass one embedding list per block when building the index."
            )
        mat = np.asarray(embeddings, dtype=np.float32)
        if mat.ndim != 2:
            raise RAGNavEmbeddingError(
                "embeddings must be a 2D array (one row per block). Check your embedding provider output shape."
            )
        doc_ids = [b.doc_id for b in blocks]
        return cls(
            blocks=blocks,
            block_doc_ids=doc_ids,
            vectors=_l2_normalize(mat),
            model_name=None,
            _backend="external",
        )

    def search(
        self,
        query_vec: list[float],
        *,
        k: int = 20,
        allowed_doc_ids: Optional[set[str]] = None,
    ) -> list[tuple[Block, float]]:
        """Search with an externally supplied query vector (same dimension as stored rows)."""
        if not self.has_fitted:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        q = _l2_normalize(q)
        sims = (self.vectors @ q.T).reshape(-1)
        if allowed_doc_ids is not None:
            mask = np.fromiter((d in allowed_doc_ids for d in self.block_doc_ids), dtype=bool)
            filtered = sims.copy()
            filtered[~mask] = -1e9
            idxs = np.argsort(-filtered)[:k]
            return [(self.blocks[int(i)], float(filtered[int(i)])) for i in idxs]
        idxs = np.argsort(-sims)[:k]
        return [(self.blocks[int(i)], float(sims[int(i)])) for i in idxs]

    def search_by_text(
        self,
        query: str,
        *,
        k: int = 20,
        allowed_doc_ids: Optional[set[str]] = None,
    ) -> list[tuple[Block, float]]:
        """Search using the same sentence-transformers model as index build (cosine on L2-normalized rows)."""
        if not self.has_fitted:
            return []
        if not self.uses_sentence_transformers or not self.model_name:
            raise RAGNavEmbeddingError(
                "search_by_text requires a sentence-transformers index. "
                "Build with VectorIndex.build(..., blocks=...) or install ragnav[embeddings]."
            )
        q = _encode_texts_sentence_transformer([query], self.model_name)
        q = _l2_normalize(q)
        sims = (self.vectors @ q.T).reshape(-1)
        if allowed_doc_ids is not None:
            mask = np.fromiter((d in allowed_doc_ids for d in self.block_doc_ids), dtype=bool)
            filtered = sims.copy()
            filtered[~mask] = -1e9
            idxs = np.argsort(-filtered)[:k]
            return [(self.blocks[int(i)], float(filtered[int(i)])) for i in idxs]
        idxs = np.argsort(-sims)[:k]
        return [(self.blocks[int(i)], float(sims[int(i)])) for i in idxs]

    def search_for_query(
        self,
        query: str,
        *,
        query_embedding: Optional[list[float]],
        k: int,
        allowed_doc_ids: Optional[set[str]] = None,
    ) -> list[tuple[Block, float]]:
        """Dispatch to text or vector search depending on how the index was built."""
        if self.uses_sentence_transformers:
            return self.search_by_text(query, k=k, allowed_doc_ids=allowed_doc_ids)
        if query_embedding is None:
            raise RAGNavEmbeddingError(
                "This index uses external embeddings; pass query_embedding from your LLM's embed() call."
            )
        return self.search(query_embedding, k=k, allowed_doc_ids=allowed_doc_ids)
