from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

from ..cache.sqlite_cache import EmbeddingCache
from ..exceptions import RAGNavEmbeddingError, RAGNavLLMError
from ..graph import Edge
from ..index.bm25 import BM25Index
from ..index.vectors import VectorIndex
from ..llm.base import LLMClient
from ..models import Block, Document
from ..observability import Trace
from ._helpers import _edges_in, _edges_out

_LOG = logging.getLogger(__name__)


@dataclass
class RAGNavIndex:
    documents: dict[str, Document]
    blocks: list[Block]
    blocks_by_id: dict[str, Block]
    blocks_by_parent: dict[str, list[Block]]
    bm25: BM25Index
    vectors: VectorIndex | None
    edges: list[Edge] = field(default_factory=list)
    edges_out: dict[str, list[Edge]] = field(default_factory=dict)
    edges_in: dict[str, list[Edge]] = field(default_factory=dict)

    @property
    def has_vectors(self) -> bool:
        return self.vectors is not None and self.vectors.has_fitted

    @classmethod
    def build(
        cls,
        *,
        documents: Iterable[Document],
        blocks: list[Block],
        llm: LLMClient | None = None,
        embed_batch_size: int = 64,
        build_vectors: bool = True,
        edges: list[Edge] | None = None,
        embedding_cache: EmbeddingCache | None = None,
        trace: Trace | None = None,
        vector_model: str | None = None,
        use_sentence_transformers: bool = True,
    ) -> RAGNavIndex:
        # An LLM client is only needed to embed blocks when sentence-transformers is not used.
        # (A BM25-only index, build_vectors=False, needs no client at all.)
        if llm is None and build_vectors and not use_sentence_transformers:
            try:
                from ..llm.mistral import MistralClient  # optional dependency

                llm = MistralClient()
            except Exception as e:
                raise RAGNavLLMError(
                    "No LLM client provided. Pass `llm=...` (e.g. FakeLLMClient), "
                    "or use use_sentence_transformers=True with build_vectors=True for local "
                    "embeddings, "
                    "or install Mistral support with: pip install ragnav[mistral]"
                ) from e

        docs_map = {d.doc_id: d for d in documents}
        blocks_by_id = {b.block_id: b for b in blocks}
        blocks_by_parent: dict[str, list[Block]] = {}
        for b in blocks:
            if b.parent_id:
                blocks_by_parent.setdefault(b.parent_id, []).append(b)
        for kids in blocks_by_parent.values():
            kids.sort(key=lambda x: x.anchors.get("line_start", 0))

        bm25 = BM25Index.build(blocks)

        vectors: VectorIndex | None = None
        if build_vectors and blocks:
            st_model = vector_model or "all-MiniLM-L6-v2"
            if use_sentence_transformers:
                vi = VectorIndex(model_name=st_model)
                try:
                    vi.build(
                        [b.text for b in blocks],
                        [b.block_id for b in blocks],
                        blocks=blocks,
                        embedding_cache=embedding_cache,
                        embed_batch_size=embed_batch_size,
                        trace=trace,
                    )
                    vectors = vi
                except RAGNavEmbeddingError as e:
                    if not isinstance(e.__cause__, ImportError):
                        raise
                    if llm is None:
                        # Without a real client the only "fallback" would be hash-based fake
                        # embeddings, which add noise to every ranking. Stay BM25-only instead.
                        _LOG.warning(
                            "sentence-transformers not available and no llm= given; building a "
                            "BM25-only index. For local embeddings install: "
                            "pip install ragnav[embeddings]"
                        )
                    else:
                        _LOG.warning(
                            "sentence-transformers not available; falling back to LLM embeddings "
                            "for vectors. "
                            "For local embeddings install: pip install ragnav[embeddings]"
                        )

            if vectors is None and llm is not None:
                texts = [b.text for b in blocks]
                embeddings: list[list[float] | None] = [None] * len(texts)
                model_hint: str | None = None
                for i in range(0, len(texts), embed_batch_size):
                    chunk = texts[i : i + embed_batch_size]
                    cached: dict[int, list[float]] = {}
                    if embedding_cache is not None:
                        cached = embedding_cache.get_many(model=model_hint, texts=chunk)
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
                            with trace.span("embed_ms"):
                                new_embeds = llm.embed(inputs=missing_texts)
                        else:
                            new_embeds = llm.embed(inputs=missing_texts)
                        for pos, emb in zip(missing_pos, new_embeds, strict=False):
                            embeddings[pos] = emb
                        if embedding_cache is not None:
                            embedding_cache.set_many(
                                model=model_hint, texts=missing_texts, embeddings=new_embeds
                            )
                        if trace:
                            trace.incr("embed_texts", len(missing_texts))

                if any(e is None for e in embeddings):
                    raise RAGNavEmbeddingError(
                        "Embedding generation failed: some blocks have no embedding after calling "
                        "the LLM. "
                        "Retry with a smaller batch, check your embedding API quota, or pass "
                        "build_vectors=False."
                    )
                vectors = VectorIndex.from_embeddings(blocks, embeddings)  # type: ignore[arg-type]

        return cls(
            documents=docs_map,
            blocks=blocks,
            blocks_by_id=blocks_by_id,
            blocks_by_parent=blocks_by_parent,
            bm25=bm25,
            vectors=vectors,
            edges=edges or [],
            edges_out=_edges_out(edges or []),
            edges_in=_edges_in(edges or []),
        )
