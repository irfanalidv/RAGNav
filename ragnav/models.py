from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


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
    trace: dict[str, Any] = field(default_factory=dict)

