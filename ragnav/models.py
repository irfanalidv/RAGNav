from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional


class ConfidenceLevel(str, Enum):
    """
    Coarse retrieval confidence after hybrid fusion (RRF or weighted) and optional reranking.

    Thresholds assume **min–max normalized** scores in ``[0, 1]`` computed from the top
    of the fused candidate list (or from the reranker pool when a reranker is used), not
    raw BM25 or raw RRF sums.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


BlockType = Literal[
    "heading",
    "paragraph",
    "list_item",
    "table",
    "code",
    "email_header",
    "message",
    "quote",
    "unknown",
]


@dataclass(frozen=True)
class Document:
    doc_id: str
    source: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Block:
    block_id: str
    doc_id: str
    type: BlockType
    text: str
    parent_id: Optional[str] = None
    heading_path: tuple[str, ...] = ()
    anchors: dict[str, Any] = field(default_factory=dict)  # page/line/url offsets etc.
    metadata: dict[str, Any] = field(default_factory=dict)  # author, timestamp, channel, etc.


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    blocks: list[Block]
    confidence: ConfidenceLevel = ConfidenceLevel.LOW
    top_score: float = 0.0
    trace: dict[str, Any] = field(default_factory=dict)

