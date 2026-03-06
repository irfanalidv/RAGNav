from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .cache.sqlite_cache import EmbeddingCache, RetrievalCache
from .index.bm25 import BM25Index
from .index.vectors import VectorIndex
from .graph import Edge, EdgeType
from .json_utils import extract_json
from .llm.base import LLMClient
from .models import Block, Document, RetrievalResult
from .observability import Trace
from .security.policy import ContentPolicy


def _minmax_norm(vals: list[float]) -> list[float]:
    if not vals:
        return vals
    lo = min(vals)
    hi = max(vals)
    if hi == lo:
        return [0.0 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


def _dedupe_keep_best(scored: Iterable[tuple[Block, float]]) -> list[tuple[Block, float]]:
    best: dict[str, tuple[Block, float]] = {}
    for b, s in scored:
        cur = best.get(b.block_id)
        if cur is None or s > cur[1]:
            best[b.block_id] = (b, s)
    return list(best.values())


def _parent_chain(by_id: dict[str, Block], block: Block, *, max_hops: int = 8) -> list[Block]:
    out: list[Block] = []
    cur = block
    hops = 0
    while cur.parent_id and hops < max_hops:
        parent = by_id.get(cur.parent_id)
        if not parent:
            break
        out.append(parent)
        cur = parent
        hops += 1
    return out


def _children(by_parent: dict[str, list[Block]], block: Block, *, max_children: int = 8) -> list[Block]:
    kids = by_parent.get(block.block_id, [])
    return kids[:max_children]


def _expand_structure(by_id: dict[str, Block], by_parent: dict[str, list[Block]], seeds: list[Block]) -> list[Block]:
    """
    Deterministic expansion to make context coherent:
    - parent chain (section headers)
    - a few children (local neighborhood)
    """
    out: list[Block] = []
    seen: set[str] = set()

    def add(b: Block):
        if b.block_id in seen:
            return
        seen.add(b.block_id)
        out.append(b)

    for seed in seeds:
        add(seed)
        for p in _parent_chain(by_id, seed):
            add(p)
        for c in _children(by_parent, seed):
            add(c)

    # Prefer higher-level context first when available
    out.sort(key=lambda b: (len(b.heading_path), b.block_id))
    return out


def _expand_graph(
    by_id: dict[str, Block],
    out_edges: dict[str, list[Edge]],
    in_edges: dict[str, list[Edge]],
    seeds: list[Block],
    *,
    edge_types: set[EdgeType],
    hops: int = 1,
    max_neighbors_per_node: int = 20,
) -> list[Block]:
    """
    Expand from seeds along typed edges (reply/next/link/etc.).
    """
    frontier = [b.block_id for b in seeds]
    seen: set[str] = set(frontier)

    for _ in range(max(1, hops)):
        nxt: list[str] = []
        for bid in frontier:
            # outgoing
            for e in out_edges.get(bid, []):
                if e.type not in edge_types:
                    continue
                if e.dst in seen:
                    continue
                if e.dst in by_id:
                    seen.add(e.dst)
                    nxt.append(e.dst)
                if len(nxt) >= max_neighbors_per_node:
                    break
            # incoming
            for e in in_edges.get(bid, []):
                if e.type not in edge_types:
                    continue
                if e.src in seen:
                    continue
                if e.src in by_id:
                    seen.add(e.src)
                    nxt.append(e.src)
                if len(nxt) >= max_neighbors_per_node:
                    break
        frontier = nxt
        if not frontier:
            break

    return [by_id[x] for x in seen if x in by_id]


def _allowed_by_constraints(
    *,
    block: Block,
    doc: Optional[Document],
    required_doc_metadata: Optional[dict[str, Any]],
    principal: Optional[str],
) -> bool:
    # doc metadata filter
    if required_doc_metadata and doc is not None:
        for k, v in required_doc_metadata.items():
            if doc.metadata.get(k) != v:
                return False

    # ACL filter (simple, explicit allow-list model)
    # Supported conventions:
    # - doc.metadata["acl"]: ["user:alice", "team:finance", ...]
    # - block.metadata["acl"]: same
    if principal:
        doc_acl = (doc.metadata.get("acl") if doc else None) or []
        blk_acl = block.metadata.get("acl") or []
        if doc_acl or blk_acl:
            allowed = set([str(x) for x in doc_acl] + [str(x) for x in blk_acl])
            if principal not in allowed:
                return False

    return True


def _edges_out(edges: list[Edge]) -> dict[str, list[Edge]]:
    out: dict[str, list[Edge]] = {}
    for e in edges:
        out.setdefault(e.src, []).append(e)
    return out


def _edges_in(edges: list[Edge]) -> dict[str, list[Edge]]:
    inc: dict[str, list[Edge]] = {}
    for e in edges:
        inc.setdefault(e.dst, []).append(e)
    return inc


def _safe_title(block: Block) -> str:
    if block.heading_path:
        return " > ".join(block.heading_path[-3:])
    for line in (block.text or "").splitlines():
        if line.strip():
            return line.strip()[:90]
    return block.block_id


def _with_trace(res: RetrievalResult, extra: dict[str, Any]) -> RetrievalResult:
    merged = {**(res.trace or {}), **(extra or {})}
    return RetrievalResult(query=res.query, blocks=res.blocks, trace=merged)


@dataclass
class RAGNavIndex:
    documents: dict[str, Document]
    blocks: list[Block]
    blocks_by_id: dict[str, Block]
    blocks_by_parent: dict[str, list[Block]]
    bm25: BM25Index
    vectors: Optional[VectorIndex]
    edges: list[Edge] = field(default_factory=list)
    edges_out: dict[str, list[Edge]] = field(default_factory=dict)
    edges_in: dict[str, list[Edge]] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        documents: Iterable[Document],
        blocks: list[Block],
        llm: Optional[LLMClient] = None,
        embed_batch_size: int = 64,
        build_vectors: bool = True,
        edges: Optional[list[Edge]] = None,
        embedding_cache: Optional[EmbeddingCache] = None,
        trace: Optional[Trace] = None,
    ) -> "RAGNavIndex":
        if llm is None:
            # Lazy default: try Mistral if installed, otherwise provide a clear error.
            try:
                from .llm.mistral import MistralClient  # optional dependency

                llm = MistralClient()
            except Exception as e:
                raise RuntimeError(
                    "No LLM client provided. Pass `llm=...` (e.g. FakeLLMClient for offline), "
                    "or install Mistral support with: pip install -e \".[mistral]\""
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

        vectors: Optional[VectorIndex] = None
        if build_vectors:
            texts = [b.text for b in blocks]
            embeddings: list[Optional[list[float]]] = [None] * len(texts)
            model_hint: Optional[str] = None
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
                    for pos, emb in zip(missing_pos, new_embeds):
                        embeddings[pos] = emb
                    if embedding_cache is not None:
                        embedding_cache.set_many(model=model_hint, texts=missing_texts, embeddings=new_embeds)
                    if trace:
                        trace.incr("embed_texts", len(missing_texts))

            if any(e is None for e in embeddings):
                raise RuntimeError("Embedding generation failed: missing embeddings for some blocks")
            vectors = VectorIndex.build(blocks, embeddings)  # type: ignore[arg-type]
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


@dataclass
class RAGNavRetriever:
    index: RAGNavIndex
    llm: Optional[LLMClient] = None

    def __post_init__(self):
        if self.llm is None:
            try:
                from .llm.mistral import MistralClient  # optional dependency

                self.llm = MistralClient()
            except Exception as e:
                raise RuntimeError(
                    "No LLM client provided. Pass `llm=...` (e.g. FakeLLMClient for offline), "
                    "or install Mistral support with: pip install -e \".[mistral]\""
                ) from e

    def retrieve(
        self,
        query: str,
        *,
        k_bm25: int = 30,
        k_vec: int = 30,
        k_final: int = 12,
        w_bm25: float = 0.45,
        w_vec: float = 0.55,
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
    ) -> RetrievalResult:
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
        }
        if retrieval_cache is not None:
            cached_ids = retrieval_cache.get(cache_payload)
            if cached_ids:
                blocks = [self.index.blocks_by_id[b] for b in cached_ids if b in self.index.blocks_by_id]
                return RetrievalResult(query=query, blocks=blocks, trace={"cache_hit": True})

        bm = self.index.bm25.search(query, k=k_bm25, allowed_doc_ids=allowed_doc_ids)
        ve: list[tuple[Block, float]] = []
        if use_vectors and self.index.vectors is not None and k_vec > 0 and w_vec > 0:
            if trace:
                with trace.span("embed_query_ms"):
                    qvec = self.llm.embed(inputs=[query])[0]
            else:
                qvec = self.llm.embed(inputs=[query])[0]
            ve = self.index.vectors.search(qvec, k=k_vec, allowed_doc_ids=allowed_doc_ids)

        bm_scores = _minmax_norm([s for _, s in bm])
        ve_scores = _minmax_norm([s for _, s in ve])

        bm_scored = [(b, w_bm25 * s) for (b, _), s in zip(bm, bm_scores)]
        ve_scored = [(b, w_vec * s) for (b, _), s in zip(ve, ve_scores)]

        merged = _dedupe_keep_best(bm_scored + ve_scored)
        merged.sort(key=lambda x: x[1], reverse=True)

        # Apply doc/ACL constraints at block level before choosing seeds
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
        seeds = [b for b, _ in filtered[:k_final]]

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
            "bm25_top": [(b.block_id, s) for (b, _), s in zip(bm[:10], bm_scores[:10])],
            "vec_top": [(b.block_id, s) for (b, _), s in zip(ve[:10], ve_scores[:10])],
            "expanded_block_ids": [b.block_id for b in expanded],
            "expand_graph": expand_graph,
        }
        if policy_info:
            trace_payload["content_policy"] = policy_info
        if trace is not None:
            trace_payload["timings_ms"] = trace.timings_ms
            trace_payload["counters"] = trace.counters

        return RetrievalResult(query=query, blocks=expanded, trace=trace_payload)

    def retrieve_raw(
        self,
        query: str,
        *,
        allowed_doc_ids: Optional[set[str]] = None,
        allowed_doc_pages: Optional[dict[str, set[int]]] = None,
        k_bm25: int = 30,
        k_vec: int = 30,
        k_final: int = 8,
        w_bm25: float = 0.45,
        w_vec: float = 0.55,
        use_vectors: bool = True,
        expand_structure: bool = True,
        retrieval_cache: Optional[RetrievalCache] = None,
        content_policy: Optional[ContentPolicy] = None,
        trace: Optional[Trace] = None,
        max_blocks: int = 8,
    ) -> list[dict[str, Any]]:
        """
        Agent-friendly retrieval primitive (similar spirit to PageIndex cookbook retrieval prompt).

        Returns a JSON-serializable list of "raw context" objects with stable identifiers
        and anchors for citation/debugging.
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

    def route_documents_by_semantics(
        self,
        query: str,
        *,
        k_bm25: int = 40,
        k_vec: int = 40,
        w_bm25: float = 0.45,
        w_vec: float = 0.55,
        max_blocks_per_doc: int = 10,
        top_docs: int = 3,
    ) -> list[tuple[str, float, int]]:
        """
        Mirrors the doc routing idea in PageIndex `tutorials/doc-search/semantics.md`:

          DocScore = (1 / sqrt(N + 1)) * sum_{n=1..N} ChunkScore(n)

        Here, "chunks" are RAGNav blocks. We compute candidate block scores via hybrid retrieval,
        then aggregate them per doc with the same diminishing-returns normalization.

        Returns: list of (doc_id, doc_score, N_used) sorted by score desc.
        """
        import math

        # Candidate generation across ALL docs
        bm = self.index.bm25.search(query, k=k_bm25)
        ve: list[tuple[Block, float]] = []
        if self.index.vectors is not None and k_vec > 0 and w_vec > 0:
            qvec = self.llm.embed(inputs=[query])[0]
            ve = self.index.vectors.search(qvec, k=k_vec)

        bm_scores = _minmax_norm([s for _, s in bm])
        ve_scores = _minmax_norm([s for _, s in ve])

        bm_scored = [(b, w_bm25 * s) for (b, _), s in zip(bm, bm_scores)]
        ve_scored = [(b, w_vec * s) for (b, _), s in zip(ve, ve_scores)]

        merged = _dedupe_keep_best(bm_scored + ve_scored)
        merged.sort(key=lambda x: x[1], reverse=True)

        # Per-doc keep top scores
        by_doc: dict[str, list[float]] = {}
        for b, s in merged:
            by_doc.setdefault(b.doc_id, []).append(float(s))

        ranked: list[tuple[str, float, int]] = []
        for doc_id, scores in by_doc.items():
            top_scores = sorted(scores, reverse=True)[:max_blocks_per_doc]
            n = len(top_scores)
            doc_score = (sum(top_scores) / math.sqrt(n + 1)) if n > 0 else 0.0
            ranked.append((doc_id, float(doc_score), n))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_docs]

    def route_pages_by_semantics(
        self,
        query: str,
        *,
        allowed_doc_ids: Optional[set[str]] = None,
        k_bm25: int = 60,
        k_vec: int = 60,
        w_bm25: float = 0.45,
        w_vec: float = 0.55,
        max_blocks_per_page: int = 10,
        top_pages: int = 4,
    ) -> list[tuple[str, int, float, int]]:
        """
        Paper-friendly routing: rank (doc_id, page) pairs using the same diminishing-returns
        aggregation idea as DocScore, but applied to page numbers.

        Returns: list of (doc_id, page, page_score, N_used) sorted desc.
        """
        import math

        bm = self.index.bm25.search(query, k=k_bm25, allowed_doc_ids=allowed_doc_ids)
        ve: list[tuple[Block, float]] = []
        if self.index.vectors is not None and k_vec > 0 and w_vec > 0:
            qvec = self.llm.embed(inputs=[query])[0]
            ve = self.index.vectors.search(qvec, k=k_vec, allowed_doc_ids=allowed_doc_ids)

        bm_scores = _minmax_norm([s for _, s in bm])
        ve_scores = _minmax_norm([s for _, s in ve])
        bm_scored = [(b, w_bm25 * s) for (b, _), s in zip(bm, bm_scores)]
        ve_scored = [(b, w_vec * s) for (b, _), s in zip(ve, ve_scores)]

        merged = _dedupe_keep_best(bm_scored + ve_scored)
        merged.sort(key=lambda x: x[1], reverse=True)

        by_page: dict[tuple[str, int], list[float]] = {}
        for b, s in merged:
            p = b.anchors.get("page")
            if not isinstance(p, int):
                continue
            by_page.setdefault((b.doc_id, p), []).append(float(s))

        ranked: list[tuple[str, int, float, int]] = []
        for (doc_id, page), scores in by_page.items():
            top_scores = sorted(scores, reverse=True)[:max_blocks_per_page]
            n = len(top_scores)
            page_score = (sum(top_scores) / math.sqrt(n + 1)) if n > 0 else 0.0
            ranked.append((doc_id, int(page), float(page_score), n))

        ranked.sort(key=lambda x: x[2], reverse=True)
        return ranked[:top_pages]

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
        2) retrieve blocks *within those pages*
        3) optionally follow cross-references via `link_to` edges (Figure/Table/Appendix/Section)
        """
        ranked_pages = self.route_pages_by_semantics(query, allowed_doc_ids=allowed_doc_ids, top_pages=top_pages)
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

    def tree_prompt_baseline(
        self,
        query: str,
        *,
        max_nodes: int = 80,
        temperature: float = 0,
    ) -> dict[str, Any]:
        """
        PageIndex-style baseline: give the model a (trimmed) tree and ask for relevant node ids.
        This is not RAGNav's default retrieval, but it enables apples-to-apples comparisons
        against "LLM navigates a hierarchy" approaches.
        """
        # Compress blocks to (id, title-ish) using heading_path tail
        nodes = []
        for b in self.index.blocks[:max_nodes]:
            title = b.heading_path[-1] if b.heading_path else b.text.splitlines()[0][:80]
            nodes.append({"node_id": b.block_id, "title": title, "anchors": b.anchors})

        prompt = f"""
You are given a question and a tree structure of a document.
Your task is to find all nodes that are likely to contain the answer to the question.

Question: {query}

Document tree structure (node_id + title):
{nodes}

Reply in the following JSON format:
{{
  "thinking": "<brief reasoning>",
  "node_list": ["node_id_1", "node_id_2", "..."]
}}
Directly return the final JSON structure. Do not output anything else.
""".strip()

        raw = self.llm.chat(messages=[{"role": "user", "content": prompt}], temperature=temperature)
        return {"raw": raw, "prompt_nodes": nodes}

    def tree_search_llm(
        self,
        query: str,
        *,
        allowed_doc_ids: Optional[set[str]] = None,
        max_nodes: int = 220,
        max_steps: int = 3,
        nodes_per_step: int = 8,
        expand_structure: bool = True,
        expand_graph: bool = False,
        graph_edge_types: Optional[set[EdgeType]] = None,
        graph_hops: int = 1,
        temperature: float = 0,
    ) -> RetrievalResult:
        """
        Reasoning-based navigation over a (trimmed) structure index.

        This mirrors the PageIndex blog loop at a minimal, library-friendly level:
        - show a structural index
        - LLM picks node_ids to read
        - retrieve those nodes and optionally expand deterministically
        - repeat for a few steps or stop when the LLM says it's enough

        If the LLM output isn't parseable, it falls back to standard hybrid retrieval.
        """
        # 1) Build a compact "index" view of nodes.
        nodes: list[dict[str, Any]] = []
        for b in self.index.blocks:
            if allowed_doc_ids is not None and b.doc_id not in allowed_doc_ids:
                continue
            nodes.append(
                {
                    "node_id": b.block_id,
                    "doc_id": b.doc_id,
                    "title": _safe_title(b),
                    "parent_id": b.parent_id,
                }
            )
            if len(nodes) >= max_nodes:
                break

        # 2) Iterative navigation loop.
        picked_ids: list[str] = []
        done = False
        for step in range(max(1, max_steps)):
            remaining = [n for n in nodes if n["node_id"] not in set(picked_ids)]
            prompt = f"""
You are given a question and a document structure index (node_id + title + parent_id).
Select up to {nodes_per_step} node_id values that are most likely to contain the answer.

Question: {query}

Already selected node_ids: {picked_ids}

Index (trimmed): {remaining}

Reply ONLY as JSON in one of the following forms:
1) {{"done": false, "node_list": ["id1", "id2"]}}
2) {{"done": true, "node_list": []}}

Rules:
- Only choose node_ids that appear in the provided Index.
- Prefer higher-level/summary nodes before deep details if unsure.
""".strip()

            raw = self.llm.chat(messages=[{"role": "user", "content": prompt}], temperature=temperature)
            try:
                parsed = extract_json(raw)
            except Exception:
                parsed = None

            node_list: list[str] = []
            if isinstance(parsed, dict):
                done = bool(parsed.get("done", False))
                node_list_any = parsed.get("node_list", [])
                if isinstance(node_list_any, list):
                    node_list = [str(x) for x in node_list_any if isinstance(x, (str, int))]
            if done:
                break

            allowed_ids = {n["node_id"] for n in nodes}
            node_list = [nid for nid in node_list if nid in allowed_ids]
            if not node_list:
                # No useful selection → fall back to hybrid retrieval.
                fallback = self.retrieve(
                    query,
                    allowed_doc_ids=allowed_doc_ids,
                    expand_structure=expand_structure,
                    expand_graph=expand_graph,
                    graph_edge_types=graph_edge_types,
                    graph_hops=graph_hops,
                )
                return _with_trace(fallback, {"mode": "tree_search_llm", "fallback": True, "raw": raw})

            for nid in node_list:
                if nid not in picked_ids:
                    picked_ids.append(nid)

        # 3) Materialize blocks and apply the same coherence expansion as the default retriever.
        seeds = [self.index.blocks_by_id[nid] for nid in picked_ids if nid in self.index.blocks_by_id]
        expanded = seeds
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

        return RetrievalResult(
            query=query,
            blocks=expanded,
            trace={
                "mode": "tree_search_llm",
                "picked_node_ids": picked_ids,
                "max_steps": max_steps,
                "max_nodes": max_nodes,
            },
        )

    def hybrid_tree_search_llm(
        self,
        query: str,
        *,
        allowed_doc_ids: Optional[set[str]] = None,
        k_candidates: int = 80,
        use_vectors: bool = True,
        k_bm25: int = 40,
        k_vec: int = 40,
        w_bm25: float = 0.45,
        w_vec: float = 0.55,
        pick_k: int = 10,
        expand_structure: bool = True,
        temperature: float = 0,
    ) -> RetrievalResult:
        """
        Hybrid "tree search" variant:
        - candidate generation via BM25 + (optional) vectors
        - LLM picks the best node_ids among candidates (lighter than full-tree prompts)
        - deterministic structure expansion for coherence

        If the LLM output isn't parseable, it falls back to standard hybrid retrieval.
        """
        bm = self.index.bm25.search(query, k=k_bm25, allowed_doc_ids=allowed_doc_ids)
        ve: list[tuple[Block, float]] = []
        if use_vectors and self.index.vectors is not None and k_vec > 0 and w_vec > 0:
            qvec = self.llm.embed(inputs=[query])[0]
            ve = self.index.vectors.search(qvec, k=k_vec, allowed_doc_ids=allowed_doc_ids)

        bm_scores = _minmax_norm([s for _, s in bm])
        ve_scores = _minmax_norm([s for _, s in ve])
        bm_scored = [(b, w_bm25 * s) for (b, _), s in zip(bm, bm_scores)]
        ve_scored = [(b, w_vec * s) for (b, _), s in zip(ve, ve_scores)]
        merged = _dedupe_keep_best(bm_scored + ve_scored)
        merged.sort(key=lambda x: x[1], reverse=True)

        candidates: list[dict[str, Any]] = []
        for b, s in merged[:k_candidates]:
            candidates.append(
                {
                    "node_id": b.block_id,
                    "doc_id": b.doc_id,
                    "title": _safe_title(b),
                    "score_hint": round(float(s), 4),
                }
            )

        prompt = f"""
You are given a question and a list of candidate nodes (node_id + title).
Pick up to {pick_k} node_ids that are most likely to contain the answer.

Question: {query}

Candidates: {candidates}

Reply ONLY JSON:
{{
  "thinking": "<brief reasoning>",
  "node_list": ["id1", "id2"]
}}

Rules:
- Only pick node_ids from the provided Candidates list.
""".strip()
        raw = self.llm.chat(messages=[{"role": "user", "content": prompt}], temperature=temperature)
        try:
            parsed = extract_json(raw)
        except Exception:
            parsed = None

        node_list: list[str] = []
        if isinstance(parsed, dict) and isinstance(parsed.get("node_list"), list):
            node_list = [str(x) for x in parsed["node_list"] if isinstance(x, (str, int))]

        allowed_ids = {c["node_id"] for c in candidates}
        node_list = [nid for nid in node_list if nid in allowed_ids][:pick_k]
        if not node_list:
            fallback = self.retrieve(
                query,
                allowed_doc_ids=allowed_doc_ids,
                k_bm25=k_bm25,
                k_vec=k_vec,
                w_bm25=w_bm25,
                w_vec=w_vec,
                expand_structure=expand_structure,
                use_vectors=use_vectors,
            )
            return _with_trace(fallback, {"mode": "hybrid_tree_search_llm", "fallback": True, "raw": raw})

        seeds = [self.index.blocks_by_id[nid] for nid in node_list if nid in self.index.blocks_by_id]
        expanded = seeds
        if expand_structure:
            expanded = _expand_structure(self.index.blocks_by_id, self.index.blocks_by_parent, expanded)

        return RetrievalResult(
            query=query,
            blocks=expanded,
            trace={
                "mode": "hybrid_tree_search_llm",
                "candidate_count": len(candidates),
                "picked_node_ids": node_list,
            },
        )

    def generate_doc_descriptions(
        self,
        *,
        max_titles: int = 50,
        temperature: float = 0,
    ) -> dict[str, str]:
        """
        Mirrors PageIndex "Search by Description" workflow:
        generate a one-sentence description per document.
        """
        outlines: dict[str, list[str]] = {}
        for b in self.index.blocks:
            title = b.heading_path[-1] if b.heading_path else None
            if not title:
                continue
            outlines.setdefault(b.doc_id, [])
            if len(outlines[b.doc_id]) < max_titles:
                outlines[b.doc_id].append(title)

        descriptions: dict[str, str] = {}
        for doc_id, titles in outlines.items():
            doc = self.index.documents.get(doc_id)
            doc_name = (doc.source if doc and doc.source else doc_id)
            prompt = f"""
You are given an outline (titles/headings) of a document.
Generate a single-sentence description that makes it easy to distinguish this document from others.
Keep it concrete and specific.

doc_name: {doc_name}
doc_id: {doc_id}
outline_titles: {titles}

Return ONLY the one-sentence description.
""".strip()
            descriptions[doc_id] = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )

        return descriptions

    def route_documents_by_description(
        self,
        query: str,
        *,
        descriptions: dict[str, str],
        top_docs: int = 3,
        temperature: float = 0,
    ) -> list[str]:
        """
        Mirrors PageIndex `tutorials/doc-search/description.md`:
        LLM selects doc_ids based on descriptions.
        """
        docs_payload = []
        for doc_id, desc in descriptions.items():
            doc = self.index.documents.get(doc_id)
            docs_payload.append(
                {
                    "doc_id": doc_id,
                    "doc_name": doc.source if doc else None,
                    "doc_description": desc,
                }
            )

        prompt = f"""
You are given a user query and a list of documents with their IDs and descriptions.
Select the documents that may contain information relevant to answering the query.

Query: {query}

Documents: {docs_payload}

Response format:
{{
  "thinking": "<brief reasoning>",
  "answer": ["doc_id1", "doc_id2"]
}}
Return [] if none are relevant.
Return ONLY the JSON structure.
""".strip()

        raw = self.llm.chat(messages=[{"role": "user", "content": prompt}], temperature=temperature)
        try:
            parsed = extract_json(raw)
            picked = parsed.get("answer", []) if isinstance(parsed, dict) else []
            picked = [d for d in picked if isinstance(d, str) and d in descriptions]
            return picked[:top_docs]
        except Exception:
            # Fallback: use semantic routing if LLM output isn't parseable
            routed = self.route_documents_by_semantics(query, top_docs=top_docs)
            return [doc_id for doc_id, _, _ in routed]

    def route_documents_by_metadata(
        self,
        *,
        required: dict[str, Any],
    ) -> list[str]:
        """
        Simple deterministic metadata filter (the building block for PageIndex-like 'query-to-SQL').
        """
        out: list[str] = []
        for doc_id, doc in self.index.documents.items():
            ok = True
            for k, v in required.items():
                if doc.metadata.get(k) != v:
                    ok = False
                    break
            if ok:
                out.append(doc_id)
        return out

