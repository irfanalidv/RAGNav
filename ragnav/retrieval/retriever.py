from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional

from ..cache.sqlite_cache import RetrievalCache
from ..exceptions import RAGNavLLMError
from ..graph import EdgeType
from ..llm.base import LLMClient
from ..models import Block, ConfidenceLevel, RetrievalResult
from ..observability import Trace
from ..security.policy import ContentPolicy
from ._helpers import (
    _allowed_by_constraints,
    _confidence_normalize_top_two,
    _expand_graph,
    _expand_structure,
    _fuse_retrieval_rankings,
    _score_confidence,
    _with_trace,
)

if TYPE_CHECKING:
    from ..cost import CostTracker
    from ..reranking.cross_encoder import CrossEncoderReranker
from . import routing, tree_search
from .index import RAGNavIndex


@dataclass
class RAGNavRetriever:
    index: RAGNavIndex
    llm: Optional[LLMClient] = None
    reranker: Optional["CrossEncoderReranker"] = None
    cost_tracker: Optional["CostTracker"] = None

    def __post_init__(self):
        if self.llm is None:
            try:
                from ..llm.mistral import MistralClient  # optional dependency

                self.llm = MistralClient()
            except Exception as e:
                raise RAGNavLLMError(
                    "No LLM client provided. Pass `llm=...` (e.g. FakeLLMClient for offline), "
                    "or install Mistral support with: pip install -e \".[mistral]\""
                ) from e
        if self.cost_tracker is not None and self.llm is not None:
            if getattr(self.llm, "cost_tracker", None) is None:
                self.llm.cost_tracker = self.cost_tracker

    def retrieve(
        self,
        query: str,
        *,
        k_bm25: int = 30,
        k_vec: int = 30,
        k_final: int = 12,
        top_k: Optional[int] = None,
        w_bm25: float = 0.5,
        w_vec: float = 0.5,
        bm25_weight: Optional[float] = None,
        vector_weight: Optional[float] = None,
        expand_structure: bool = True,
        allowed_doc_ids: Optional[set[str]] = None,
        allowed_doc_pages: Optional[dict[str, set[int]]] = None,
        use_vectors: bool = True,
        required_doc_metadata: Optional[dict[str, Any]] = None,
        principal: Optional[str] = None,
        expand_graph: bool = False,
        graph_edge_types: Optional[set[EdgeType]] = None,
        graph_hops: int = 1,
        retrieval_cache: Optional[RetrievalCache] = None,
        content_policy: Optional[ContentPolicy] = None,
        trace: Optional[Trace] = None,
        fusion: Literal["rrf", "weighted"] = "rrf",
    ) -> RetrievalResult:
        """
        Hybrid retrieval with optional structure/graph expansion.

        When ``reranker`` is set, fusion keeps ``max(k_final * 5, 50)`` candidates before
        reranking to ``k_final`` — reranking works best with **≥50** first-stage candidates.
        """
        if bm25_weight is not None:
            w_bm25 = bm25_weight
        if vector_weight is not None:
            w_vec = vector_weight
        if top_k is not None:
            k_bm25 = top_k
            k_vec = top_k
            k_final = top_k

        cache_payload = {
            "query": query,
            "allowed_doc_ids": sorted(list(allowed_doc_ids)) if allowed_doc_ids else None,
            "allowed_doc_pages": (
                {k: sorted(list(v)) for k, v in allowed_doc_pages.items()} if allowed_doc_pages else None
            ),
            "required_doc_metadata": required_doc_metadata,
            "principal": principal,
            "k_bm25": k_bm25,
            "k_vec": k_vec,
            "k_final": k_final,
            "w_bm25": w_bm25,
            "w_vec": w_vec,
            "use_vectors": use_vectors,
            "expand_structure": expand_structure,
            "expand_graph": expand_graph,
            "graph_edge_types": sorted(list(graph_edge_types)) if graph_edge_types else None,
            "graph_hops": graph_hops,
            "fusion": fusion,
            "reranker_model": self.reranker.model_name if self.reranker is not None else None,
        }
        if retrieval_cache is not None:
            cached_ids = retrieval_cache.get(cache_payload)
            if cached_ids:
                blocks = [self.index.blocks_by_id[b] for b in cached_ids if b in self.index.blocks_by_id]
                return RetrievalResult(
                    query=query,
                    blocks=blocks,
                    confidence=ConfidenceLevel.LOW,
                    top_score=0.0,
                    trace={"cache_hit": True},
                )

        bm: list[tuple[Block, float]] = []
        if w_bm25 > 0:
            bm = self.index.bm25.search(query, k=k_bm25, allowed_doc_ids=allowed_doc_ids)
        ve: list[tuple[Block, float]] = []
        if use_vectors and self.index.has_vectors and k_vec > 0 and w_vec > 0:
            vidx = self.index.vectors
            assert vidx is not None
            qemb: Optional[list[float]] = None
            if not vidx.uses_sentence_transformers:
                if trace:
                    with trace.span("embed_query_ms"):
                        qemb = self.llm.embed(inputs=[query])[0]
                else:
                    qemb = self.llm.embed(inputs=[query])[0]
            ve = vidx.search_for_query(
                query, query_embedding=qemb, k=k_vec, allowed_doc_ids=allowed_doc_ids
            )

        merged = _fuse_retrieval_rankings(
            bm, ve, fusion=fusion, w_bm25=w_bm25, w_vec=w_vec
        )

        filtered = []
        for b, s in merged:
            if allowed_doc_ids is not None and b.doc_id not in allowed_doc_ids:
                continue
            if allowed_doc_pages is not None:
                pages = allowed_doc_pages.get(b.doc_id)
                if not pages:
                    continue
                p = b.anchors.get("page")
                if not isinstance(p, int) or p not in pages:
                    continue
            doc = self.index.documents.get(b.doc_id)
            if not _allowed_by_constraints(
                block=b,
                doc=doc,
                required_doc_metadata=required_doc_metadata,
                principal=principal,
            ):
                continue
            filtered.append((b, s))

        if not filtered:
            trace_empty: dict[str, Any] = {
                "query": query,
                "seed_block_ids": [],
                "fusion": fusion,
                "bm25_top": [(b.block_id, s) for b, s in bm[:10]],
                "vec_top": [(b.block_id, s) for b, s in ve[:10]],
                "expanded_block_ids": [],
                "expand_graph": expand_graph,
            }
            return RetrievalResult(
                query=query,
                blocks=[],
                confidence=ConfidenceLevel.LOW,
                top_score=0.0,
                trace=trace_empty,
            )

        rerank_pool = min(len(filtered), max(k_final * 5, 50))
        scored_for_confidence: list[tuple[Block, float]]
        if self.reranker is not None and rerank_pool > 0:
            pool_blocks = [b for b, _ in filtered[:rerank_pool]]
            scored_for_confidence = self.reranker.rerank_scored(query, pool_blocks)
            seeds = [b for b, _ in scored_for_confidence[:k_final]]
        else:
            scored_for_confidence = filtered
            seeds = [b for b, _ in filtered[:k_final]]

        top_norm, second_norm = _confidence_normalize_top_two(scored_for_confidence, window=50)
        conf = _score_confidence(top_norm, second_norm)
        top_score_out = float(top_norm)

        expanded: list[Block] = seeds
        if expand_structure:
            expanded = _expand_structure(self.index.blocks_by_id, self.index.blocks_by_parent, expanded)
        if expand_graph and self.index.edges:
            types = graph_edge_types or {"next", "reply_to", "same_thread", "quote_of"}
            expanded = _expand_graph(
                self.index.blocks_by_id,
                self.index.edges_out,
                self.index.edges_in,
                expanded,
                edge_types=set(types),
                hops=graph_hops,
            )

        policy_info: dict[str, Any] = {}
        if content_policy is not None:
            pr = content_policy.apply(query=query, blocks=expanded)
            expanded = pr.kept
            policy_info = {
                "dropped_block_ids": pr.dropped_block_ids,
                "sanitized_block_ids": pr.sanitized_block_ids,
            }

        if retrieval_cache is not None:
            retrieval_cache.set(cache_payload, [b.block_id for b in expanded])

        trace_payload: dict[str, Any] = {
            "query": query,
            "allowed_doc_ids": sorted(list(allowed_doc_ids)) if allowed_doc_ids else None,
            "allowed_doc_pages": (
                {k: sorted(list(v)) for k, v in allowed_doc_pages.items()} if allowed_doc_pages else None
            ),
            "required_doc_metadata": required_doc_metadata,
            "principal": principal,
            "seed_block_ids": [b.block_id for b in seeds],
            "fusion": fusion,
            "bm25_top": [(b.block_id, s) for b, s in bm[:10]],
            "vec_top": [(b.block_id, s) for b, s in ve[:10]],
            "expanded_block_ids": [b.block_id for b in expanded],
            "expand_graph": expand_graph,
        }
        if policy_info:
            trace_payload["content_policy"] = policy_info
        if trace is not None:
            trace_payload["timings_ms"] = trace.timings_ms
            trace_payload["counters"] = trace.counters

        if not expanded:
            conf = ConfidenceLevel.LOW
            top_score_out = 0.0
        trace_payload["confidence"] = conf.value
        trace_payload["top_score"] = top_score_out

        return RetrievalResult(
            query=query,
            blocks=expanded,
            confidence=conf,
            top_score=top_score_out,
            trace=trace_payload,
        )

    def retrieve_raw(
        self,
        query: str,
        *,
        allowed_doc_ids: Optional[set[str]] = None,
        allowed_doc_pages: Optional[dict[str, set[int]]] = None,
        k_bm25: int = 30,
        k_vec: int = 30,
        k_final: int = 8,
        w_bm25: float = 0.5,
        w_vec: float = 0.5,
        use_vectors: bool = True,
        expand_structure: bool = True,
        retrieval_cache: Optional[RetrievalCache] = None,
        content_policy: Optional[ContentPolicy] = None,
        trace: Optional[Trace] = None,
        max_blocks: int = 8,
    ) -> list[dict[str, Any]]:
        """
        Agent-oriented retrieval: JSON-serializable context objects with stable ids and anchors.
        """
        res = self.retrieve(
            query,
            allowed_doc_ids=allowed_doc_ids,
            allowed_doc_pages=allowed_doc_pages,
            k_bm25=k_bm25,
            k_vec=k_vec,
            k_final=k_final,
            w_bm25=w_bm25,
            w_vec=w_vec,
            use_vectors=use_vectors,
            expand_structure=expand_structure,
            retrieval_cache=retrieval_cache,
            content_policy=content_policy,
            trace=trace,
        )
        out: list[dict[str, Any]] = []
        for b in res.blocks[:max_blocks]:
            out.append(
                {
                    "doc_id": b.doc_id,
                    "block_id": b.block_id,
                    "title": " > ".join(b.heading_path) if b.heading_path else None,
                    "anchors": b.anchors,
                    "content": b.text,
                }
            )
        return out

    def route_documents_by_semantics(self, query: str, **kwargs: Any) -> list[tuple[str, float, int]]:
        return routing.route_documents_by_semantics(self.index, self.llm, query, **kwargs)

    def route_pages_by_semantics(self, query: str, **kwargs: Any) -> list[tuple[str, int, float, int]]:
        return routing.route_pages_by_semantics(self.index, self.llm, query, **kwargs)

    def retrieve_paper(
        self,
        query: str,
        *,
        allowed_doc_ids: Optional[set[str]] = None,
        top_pages: int = 4,
        follow_refs: bool = True,
        include_next: bool = True,
        graph_hops: int = 1,
        use_vectors: bool = True,
        k_final: int = 10,
        content_policy: Optional[ContentPolicy] = None,
        retrieval_cache: Optional[RetrievalCache] = None,
    ) -> RetrievalResult:
        """
        Paper-optimized retrieval:
        1) route to likely pages
        2) retrieve blocks within those pages
        3) optionally follow cross-references via `link_to` edges
        """
        ranked_pages = routing.route_pages_by_semantics(
            self.index, self.llm, query, allowed_doc_ids=allowed_doc_ids, top_pages=top_pages
        )
        if not ranked_pages:
            res = self.retrieve(query, allowed_doc_ids=allowed_doc_ids, use_vectors=use_vectors, k_final=k_final)
            return _with_trace(res, {"mode": "paper", "fallback": True})

        allowed_doc_pages: dict[str, set[int]] = {}
        for doc_id, page, _score, _n in ranked_pages:
            allowed_doc_pages.setdefault(doc_id, set()).add(int(page))

        edge_types: set[EdgeType] = set()
        if follow_refs:
            edge_types.add("link_to")
        if include_next:
            edge_types.add("next")

        res = self.retrieve(
            query,
            allowed_doc_ids=allowed_doc_ids,
            allowed_doc_pages=allowed_doc_pages,
            use_vectors=use_vectors,
            k_final=k_final,
            expand_structure=True,
            expand_graph=bool(edge_types),
            graph_edge_types=edge_types if edge_types else None,
            graph_hops=graph_hops,
            content_policy=content_policy,
            retrieval_cache=retrieval_cache,
        )
        return _with_trace(
            res,
            {
                "mode": "paper",
                "routed_pages": ranked_pages,
                "follow_refs": follow_refs,
            },
        )

    def tree_prompt_baseline(self, query: str, **kwargs: Any) -> dict[str, Any]:
        return tree_search.tree_prompt_baseline(self, query, **kwargs)

    def tree_search_llm(self, query: str, **kwargs: Any) -> RetrievalResult:
        return tree_search.tree_search_llm(self, query, **kwargs)

    def hybrid_tree_search_llm(self, query: str, **kwargs: Any) -> RetrievalResult:
        return tree_search.hybrid_tree_search_llm(self, query, **kwargs)

    def generate_doc_descriptions(self, **kwargs: Any) -> dict[str, str]:
        return routing.generate_doc_descriptions(self.index, self.llm, **kwargs)

    def route_documents_by_description(self, query: str, **kwargs: Any) -> list[str]:
        return routing.route_documents_by_description(self.index, self.llm, query, **kwargs)

    def route_documents_by_metadata(self, *, required: dict[str, Any]) -> list[str]:
        return routing.route_documents_by_metadata(self.index, required=required)
