from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from ..models import Block
from ..retrieval import RAGNavRetriever
from .graph import EntityGraph


@dataclass(frozen=True)
class EntityGraphRetrieverConfig:
    hops: int = 2
    max_entities: int = 6
    max_relations_per_entity: int = 12
    max_evidence_blocks: int = 10
    include_relation_evidence: bool = True


_Q_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-+/]*")


def _query_terms(query: str) -> set[str]:
    return {t.lower() for t in _Q_TOKEN.findall(query or "") if len(t) >= 2}


class EntityGraphRetriever:
    """
    Entity-graph retrieval with provenance:
    - match query terms to entity names
    - traverse relations for a few hops
    - return the supporting evidence blocks (block_id -> Block)
    """

    def __init__(self, *, graph: EntityGraph, blocks_by_id: dict[str, Block]):
        self.graph = graph
        self.blocks_by_id = blocks_by_id

    def match_entities(self, query: str, *, cfg: EntityGraphRetrieverConfig) -> list[str]:
        q = (query or "").lower()
        terms = _query_terms(query)
        scored: list[tuple[str, float]] = []
        for eid, ent in self.graph.entities.items():
            name_terms = _query_terms(ent.name)
            overlap = len(terms.intersection(name_terms))
            if overlap == 0:
                continue
            # Prefer exact substring matches and longer names.
            name_low = (ent.name or "").lower()
            bonus = 2.0 if name_low and name_low in q else 0.0
            scored.append((eid, float(overlap) + bonus + (len(name_low) / 100.0)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [eid for eid, _ in scored[: cfg.max_entities]]

    def retrieve(self, query: str, *, cfg: EntityGraphRetrieverConfig = EntityGraphRetrieverConfig()) -> dict[str, Any]:
        seeds = self.match_entities(query, cfg=cfg)
        visited_entities: set[str] = set(seeds)
        evidence_ids: list[str] = []
        visited_edges: list[dict[str, Any]] = []

        frontier = list(seeds)
        for _ in range(max(1, cfg.hops)):
            nxt: list[str] = []
            for eid in frontier:
                rels = self.graph.out_relations(eid)[: cfg.max_relations_per_entity]
                for r in rels:
                    visited_edges.append({"src": r.src, "dst": r.dst, "type": r.type})
                    if cfg.include_relation_evidence:
                        for bid in r.evidence_block_ids:
                            if bid not in evidence_ids:
                                evidence_ids.append(bid)
                            if len(evidence_ids) >= cfg.max_evidence_blocks:
                                break
                    if r.dst not in visited_entities:
                        visited_entities.add(r.dst)
                        nxt.append(r.dst)
            frontier = nxt
            if not frontier or len(evidence_ids) >= cfg.max_evidence_blocks:
                break

        blocks = [self.blocks_by_id[bid] for bid in evidence_ids if bid in self.blocks_by_id]
        return {
            "query": query,
            "seed_entity_ids": seeds,
            "visited_entity_ids": sorted(list(visited_entities)),
            "visited_relations": visited_edges,
            "evidence_block_ids": evidence_ids,
            "blocks": blocks,
        }

    @staticmethod
    def hybrid_fallback(
        query: str,
        *,
        rag_retriever: RAGNavRetriever,
        allowed_doc_ids: Optional[set[str]] = None,
        k_final: int = 10,
    ):
        return rag_retriever.retrieve(query, allowed_doc_ids=allowed_doc_ids, k_final=k_final)

