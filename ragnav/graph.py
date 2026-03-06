from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from .models import Block, Document


EdgeType = Literal[
    "parent",        # hierarchy (e.g., heading -> paragraph)
    "next",          # sequence (e.g., message -> next message)
    "reply_to",      # conversational threading
    "quote_of",      # quoting/referencing another block
    "link_to",       # hyperlink / citation
    "contains",      # container relationship (e.g., doc -> block)
    "same_thread",   # grouping edge for threads
]


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    type: EdgeType
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BlockGraph:
    """
    Canonical graph representation across messy sources:
    - Documents
    - Blocks (typed, anchored)
    - Typed edges (reply, next, link, etc.)
    """

    documents: dict[str, Document] = field(default_factory=dict)
    blocks: dict[str, Block] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    _out: dict[str, list[Edge]] = field(default_factory=dict, init=False, repr=False)
    _in: dict[str, list[Edge]] = field(default_factory=dict, init=False, repr=False)

    def add_document(self, doc: Document) -> None:
        self.documents[doc.doc_id] = doc

    def add_block(self, block: Block) -> None:
        self.blocks[block.block_id] = block

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)
        self._out.setdefault(edge.src, []).append(edge)
        self._in.setdefault(edge.dst, []).append(edge)

    def build_indexes(self) -> None:
        self._out.clear()
        self._in.clear()
        for e in self.edges:
            self._out.setdefault(e.src, []).append(e)
            self._in.setdefault(e.dst, []).append(e)

    def out_edges(self, block_id: str, *, types: Optional[set[EdgeType]] = None) -> list[Edge]:
        edges = self._out.get(block_id, [])
        if types is None:
            return list(edges)
        return [e for e in edges if e.type in types]

    def in_edges(self, block_id: str, *, types: Optional[set[EdgeType]] = None) -> list[Edge]:
        edges = self._in.get(block_id, [])
        if types is None:
            return list(edges)
        return [e for e in edges if e.type in types]

    def neighbors(
        self,
        block_id: str,
        *,
        types: Optional[set[EdgeType]] = None,
        include_incoming: bool = True,
        include_outgoing: bool = True,
        max_degree: int = 50,
    ) -> list[str]:
        seen: list[str] = []
        if include_outgoing:
            for e in self.out_edges(block_id, types=types):
                if e.dst not in seen:
                    seen.append(e.dst)
                if len(seen) >= max_degree:
                    return seen
        if include_incoming:
            for e in self.in_edges(block_id, types=types):
                if e.src not in seen:
                    seen.append(e.src)
                if len(seen) >= max_degree:
                    return seen
        return seen

