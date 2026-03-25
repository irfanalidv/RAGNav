from __future__ import annotations

from typing import Any, Optional

from ..graph import EdgeType
from ..json_utils import extract_json
from ..models import Block, ConfidenceLevel, RetrievalResult
from ._helpers import _dedupe_keep_best, _expand_graph, _expand_structure, _minmax_norm, _safe_title, _with_trace


def tree_prompt_baseline(retriever: Any, query: str, *, max_nodes: int = 80, temperature: float = 0) -> dict[str, Any]:
    """
    Baseline: give the model a trimmed list of nodes (ids + titles) and ask for relevant node ids.
    """
    nodes = []
    for b in retriever.index.blocks[:max_nodes]:
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

    raw = retriever.llm.chat(messages=[{"role": "user", "content": prompt}], temperature=temperature)
    return {"raw": raw, "prompt_nodes": nodes}


def tree_search_llm(
    retriever: Any,
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
    Iterative LLM navigation: structural index → picked node ids → optional expansion (repeat).
    """
    nodes: list[dict[str, Any]] = []
    for b in retriever.index.blocks:
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

    picked_ids: list[str] = []
    done = False
    for _ in range(max(1, max_steps)):
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

        raw = retriever.llm.chat(messages=[{"role": "user", "content": prompt}], temperature=temperature)
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
            fallback = retriever.retrieve(
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

    seeds = [retriever.index.blocks_by_id[nid] for nid in picked_ids if nid in retriever.index.blocks_by_id]
    expanded = seeds
    if expand_structure:
        expanded = _expand_structure(retriever.index.blocks_by_id, retriever.index.blocks_by_parent, expanded)
    if expand_graph and retriever.index.edges:
        types = graph_edge_types or {"next", "reply_to", "same_thread", "quote_of"}
        expanded = _expand_graph(
            retriever.index.blocks_by_id,
            retriever.index.edges_out,
            retriever.index.edges_in,
            expanded,
            edge_types=set(types),
            hops=graph_hops,
        )

    return RetrievalResult(
        query=query,
        blocks=expanded,
        confidence=ConfidenceLevel.LOW,
        top_score=0.0,
        trace={
            "mode": "tree_search_llm",
            "picked_node_ids": picked_ids,
            "max_steps": max_steps,
            "max_nodes": max_nodes,
        },
    )


def hybrid_tree_search_llm(
    retriever: Any,
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
    Hybrid variant: BM25/vector candidates, LLM picks node ids, then structure expansion.
    """
    bm = retriever.index.bm25.search(query, k=k_bm25, allowed_doc_ids=allowed_doc_ids)
    ve: list[tuple[Block, float]] = []
    if use_vectors and retriever.index.has_vectors and k_vec > 0 and w_vec > 0:
        vidx = retriever.index.vectors
        assert vidx is not None
        qemb = None if vidx.uses_sentence_transformers else retriever.llm.embed(inputs=[query])[0]
        ve = vidx.search_for_query(
            query, query_embedding=qemb, k=k_vec, allowed_doc_ids=allowed_doc_ids
        )

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
    raw = retriever.llm.chat(messages=[{"role": "user", "content": prompt}], temperature=temperature)
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
        fallback = retriever.retrieve(
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

    seeds = [retriever.index.blocks_by_id[nid] for nid in node_list if nid in retriever.index.blocks_by_id]
    expanded = seeds
    if expand_structure:
        expanded = _expand_structure(retriever.index.blocks_by_id, retriever.index.blocks_by_parent, expanded)

    return RetrievalResult(
        query=query,
        blocks=expanded,
        confidence=ConfidenceLevel.LOW,
        top_score=0.0,
        trace={
            "mode": "hybrid_tree_search_llm",
            "candidate_count": len(candidates),
            "picked_node_ids": node_list,
        },
    )
