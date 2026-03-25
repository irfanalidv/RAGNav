from __future__ import annotations

from typing import Any, Iterable, Optional

from ..graph import Edge, EdgeType
from ..models import Block, ConfidenceLevel, Document, RetrievalResult


def _minmax_norm(vals: list[float]) -> list[float]:
    if not vals:
        return vals
    lo = min(vals)
    hi = max(vals)
    if hi == lo:
        return [0.0 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


def _rrf_fuse(
    ranked_lists: list[list[tuple[str, float]]],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.

    ``k=60`` follows Cormack et al. (2009): a larger ``k`` dampens the influence of
    top-ranked outliers so one noisy #1 hit cannot dominate the fused score.

    Each inner list is ``(block_id, score)``; only rank position matters for fusion,
    not the magnitude of ``score``.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (block_id, _) in enumerate(ranked, 1):
            scores[block_id] = scores.get(block_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _fuse_retrieval_rankings(
    bm: list[tuple[Block, float]],
    ve: list[tuple[Block, float]],
    *,
    fusion: str,
    w_bm25: float,
    w_vec: float,
) -> list[tuple[Block, float]]:
    """
    Combine BM25 and dense hit lists for hybrid retrieval.

    ``fusion="rrf"`` uses reciprocal rank fusion when at least one non-empty
    ranked list is present. ``fusion="weighted"`` uses min–max normalized scores
    and the caller's BM25 / vector weights (legacy path).
    """
    by_id: dict[str, Block] = {}
    for b, _ in bm:
        by_id[b.block_id] = b
    for b, _ in ve:
        by_id[b.block_id] = b

    if fusion == "weighted":
        bm_scores = _minmax_norm([s for _, s in bm])
        ve_scores = _minmax_norm([s for _, s in ve])
        bm_scored = [(b, w_bm25 * s) for (b, _), s in zip(bm, bm_scores)]
        ve_scored = [(b, w_vec * s) for (b, _), s in zip(ve, ve_scores)]
        merged = _dedupe_keep_best(bm_scored + ve_scored)
        merged.sort(key=lambda x: x[1], reverse=True)
        return merged

    ranked_lists: list[list[tuple[str, float]]] = []
    if bm and w_bm25 > 0:
        ranked_lists.append([(b.block_id, s) for b, s in bm])
    if ve and w_vec > 0:
        ranked_lists.append([(b.block_id, s) for b, s in ve])
    if not ranked_lists:
        return []
    fused = _rrf_fuse(ranked_lists)
    out: list[tuple[Block, float]] = []
    for bid, rrf_s in fused:
        b = by_id.get(bid)
        if b is not None:
            out.append((b, rrf_s))
    return out


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
    if required_doc_metadata and doc is not None:
        for k, v in required_doc_metadata.items():
            if doc.metadata.get(k) != v:
                return False

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


def _confidence_normalize_top_two(
    scored: list[tuple[Block, float]],
    *,
    window: int = 50,
) -> tuple[float, float]:
    """
    Map the top two raw fusion or reranker scores into [0, 1] via min–max over the
    first ``window`` candidates so fixed thresholds in ``_score_confidence`` apply.
    """
    if not scored:
        return 0.0, 0.0
    if len(scored) == 1:
        return 1.0, 0.0
    w = min(len(scored), max(window, 2))
    slice_scores = [s for _, s in scored[:w]]
    lo, hi = min(slice_scores), max(slice_scores)
    raw_top = float(scored[0][1])
    raw_second = float(scored[1][1]) if len(scored) > 1 else 0.0
    if hi <= lo:
        # Degenerate window (all tied scores): avoid mapping every hit to LOW.
        return (0.62, 0.61)
    return (
        (raw_top - lo) / (hi - lo),
        (raw_second - lo) / (hi - lo),
    )


def _score_confidence(top_score: float, second_score: float) -> ConfidenceLevel:
    """
    Map normalized top score and margin to HIGH / MEDIUM / LOW.

    Expects ``top_score`` and ``second_score`` already scaled to roughly ``[0, 1]``
    (see ``_confidence_normalize_top_two``), not raw BM25 or RRF sums.
    """
    if top_score < 0.60:
        return ConfidenceLevel.LOW
    gap = top_score - second_score
    if top_score > 0.85 and gap > 0.15:
        return ConfidenceLevel.HIGH
    return ConfidenceLevel.MEDIUM


def _safe_title(block: Block) -> str:
    if block.heading_path:
        return " > ".join(block.heading_path[-3:])
    for line in (block.text or "").splitlines():
        if line.strip():
            return line.strip()[:90]
    return block.block_id


def _with_trace(res: RetrievalResult, extra: dict[str, Any]) -> RetrievalResult:
    merged = {**(res.trace or {}), **(extra or {})}
    return RetrievalResult(
        query=res.query,
        blocks=res.blocks,
        confidence=res.confidence,
        top_score=res.top_score,
        trace=merged,
    )
