from __future__ import annotations

import math
from typing import Any, Optional

from ..json_utils import extract_json
from ..llm.base import LLMClient
from ..models import Block
from ._helpers import _dedupe_keep_best, _minmax_norm
from .index import RAGNavIndex


def route_documents_by_semantics(
    index: RAGNavIndex,
    llm: LLMClient,
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
    Rank documents by hybrid block scores aggregated per document.

    DocScore uses (1 / sqrt(N + 1)) * sum of top block scores per document (diminishing returns).
    """
    bm = index.bm25.search(query, k=k_bm25)
    ve: list[tuple[Block, float]] = []
    if index.has_vectors and k_vec > 0 and w_vec > 0:
        vidx = index.vectors
        assert vidx is not None
        qemb = None if vidx.uses_sentence_transformers else llm.embed(inputs=[query])[0]
        ve = vidx.search_for_query(query, query_embedding=qemb, k=k_vec)

    bm_scores = _minmax_norm([s for _, s in bm])
    ve_scores = _minmax_norm([s for _, s in ve])

    bm_scored = [(b, w_bm25 * s) for (b, _), s in zip(bm, bm_scores)]
    ve_scored = [(b, w_vec * s) for (b, _), s in zip(ve, ve_scores)]

    merged = _dedupe_keep_best(bm_scored + ve_scored)
    merged.sort(key=lambda x: x[1], reverse=True)

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
    index: RAGNavIndex,
    llm: LLMClient,
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
    Rank (doc_id, page) pairs using the same diminishing-returns aggregation as document routing.
    """
    bm = index.bm25.search(query, k=k_bm25, allowed_doc_ids=allowed_doc_ids)
    ve: list[tuple[Block, float]] = []
    if index.has_vectors and k_vec > 0 and w_vec > 0:
        vidx = index.vectors
        assert vidx is not None
        qemb = None if vidx.uses_sentence_transformers else llm.embed(inputs=[query])[0]
        ve = vidx.search_for_query(
            query, query_embedding=qemb, k=k_vec, allowed_doc_ids=allowed_doc_ids
        )

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


def generate_doc_descriptions(
    index: RAGNavIndex,
    llm: LLMClient,
    *,
    max_titles: int = 50,
    temperature: float = 0,
) -> dict[str, str]:
    """
    Ask the LLM for a one-sentence description per document from outline titles (headings).
    """
    outlines: dict[str, list[str]] = {}
    for b in index.blocks:
        title = b.heading_path[-1] if b.heading_path else None
        if not title:
            continue
        outlines.setdefault(b.doc_id, [])
        if len(outlines[b.doc_id]) < max_titles:
            outlines[b.doc_id].append(title)

    descriptions: dict[str, str] = {}
    for doc_id, titles in outlines.items():
        doc = index.documents.get(doc_id)
        doc_name = doc.source if doc and doc.source else doc_id
        prompt = f"""
You are given an outline (titles/headings) of a document.
Generate a single-sentence description that makes it easy to distinguish this document from others.
Keep it concrete and specific.

doc_name: {doc_name}
doc_id: {doc_id}
outline_titles: {titles}

Return ONLY the one-sentence description.
""".strip()
        descriptions[doc_id] = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )

    return descriptions


def route_documents_by_description(
    index: RAGNavIndex,
    llm: LLMClient,
    query: str,
    *,
    descriptions: dict[str, str],
    top_docs: int = 3,
    temperature: float = 0,
) -> list[str]:
    """Use the LLM to pick doc_ids from short descriptions; fall back to semantic routing if JSON fails."""
    docs_payload = []
    for doc_id, desc in descriptions.items():
        doc = index.documents.get(doc_id)
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

    raw = llm.chat(messages=[{"role": "user", "content": prompt}], temperature=temperature)
    try:
        parsed = extract_json(raw)
        picked = parsed.get("answer", []) if isinstance(parsed, dict) else []
        picked = [d for d in picked if isinstance(d, str) and d in descriptions]
        return picked[:top_docs]
    except Exception:
        routed = route_documents_by_semantics(index, llm, query, top_docs=top_docs)
        return [doc_id for doc_id, _, _ in routed]


def route_documents_by_metadata(
    index: RAGNavIndex,
    *,
    required: dict[str, Any],
) -> list[str]:
    """Return doc_ids whose document metadata matches all key/value pairs in ``required``."""
    out: list[str] = []
    for doc_id, doc in index.documents.items():
        ok = True
        for k, v in required.items():
            if doc.metadata.get(k) != v:
                ok = False
                break
        if ok:
            out.append(doc_id)
    return out
