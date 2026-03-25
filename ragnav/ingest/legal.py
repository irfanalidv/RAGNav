"""
Legal/numbered document ingestion: detects section structure from numbered headings,
lettered subsections, and definition blocks — the structure that markdown ``#`` heading
detection misses in contracts, regulations, and technical specifications.

Plain contracts are often flat text with ``12. Termination``-style lines rather than
markdown headers; this parser turns those signals into ``Block`` hierarchy and graph
edges so structure-aware retrieval and expansion behave like they do on markdown.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Optional, Union

from ..graph import BlockGraph, Edge
from ..models import Block, Document

_T1_ARTICLE = re.compile(r"^(ARTICLE|SECTION)\s+[IVXLCDM\d]+", re.I)
_T1_NUMBERED = re.compile(r"^\d+\.\s+[A-Z][A-Za-z ]+")
_T1_CAPS = re.compile(r"^[A-Z][A-Z ]{4,}\s*$")

_T2_NUMBERED = re.compile(r"^\d+\.\d+(?:\.\d+)*[\.\s]")
_T2_SECTION = re.compile(r"^Section\s+\d+(?:\.\d+)*", re.I)

_T3_LETTER = re.compile(r"^\s*\([a-z]\)\s")
_T3_ROMAN = re.compile(r"^\s*\([ivxlcdm]+\)\s", re.I)

_T4_DEF = re.compile(r'^\s*"[A-Z][A-Za-z]+"\s+means\b', re.I)

# Paragraphs shorter than this are prepended to the following block instead of standing alone.
MIN_BLOCK_CHARS = 80

_ClassifyResult = Union[Literal["blank"], Literal["body"], tuple[int, str]]


def _classify_line(line: str) -> _ClassifyResult:
    if not line.strip():
        return "blank"
    s = line.strip()
    if _T1_ARTICLE.match(s) or _T1_NUMBERED.match(s) or _T1_CAPS.match(s):
        return (1, s)
    if _T2_NUMBERED.match(s) or _T2_SECTION.match(s):
        return (2, s)
    if _T3_LETTER.match(line) or _T3_ROMAN.match(line):
        return (3, s)
    if _T4_DEF.match(line):
        return (4, s)
    return "body"


def ingest_legal(
    text: str,
    *,
    doc_id: str,
    metadata: Optional[dict[str, Any]] = None,
) -> tuple[Document, list[Block], BlockGraph]:
    """
    Parse legal or numbered prose into structured blocks with parent-child edges.

    Returns a ``BlockGraph`` (contains / parent / next) so hybrid retrieval can expand
    along the same edge types as markdown and PDF graph ingestion.
    """
    doc_meta: dict[str, Any] = {"type": "legal"}
    if metadata:
        doc_meta.update(metadata)
    doc = Document(doc_id=doc_id, source=None, metadata=doc_meta)

    lines = text.splitlines()
    blocks: list[Block] = []
    bid = 0
    carry: str = ""

    def next_id() -> str:
        nonlocal bid
        out = "%s#l%d" % (doc_id, bid)
        bid += 1
        return out

    # (tier, title text, block_id)
    stack: list[tuple[int, str, str]] = []
    para_buf: list[str] = []

    def heading_path_tuple() -> tuple[str, ...]:
        return tuple(t[1] for t in stack)

    def body_parent_id() -> Optional[str]:
        return stack[-1][2] if stack else None

    def flush_paragraph() -> None:
        nonlocal carry
        if not para_buf:
            return
        body = "\n".join(para_buf).strip()
        para_buf.clear()
        if not body:
            return
        merged = (carry + "\n" + body).strip() if carry else body
        carry = ""
        if len(merged) < MIN_BLOCK_CHARS:
            carry = merged
            return
        pid = body_parent_id()
        blocks.append(
            Block(
                block_id=next_id(),
                doc_id=doc_id,
                type="paragraph",
                text=merged,
                parent_id=pid,
                heading_path=heading_path_tuple(),
                anchors={},
            )
        )

    for line in lines:
        kind = _classify_line(line)
        if kind == "blank":
            flush_paragraph()
            continue
        if kind == "body":
            para_buf.append(line)
            continue
        assert isinstance(kind, tuple)
        tier, title = kind[0], kind[1]
        if tier == 1:
            flush_paragraph()
            stack.clear()
            if _T1_CAPS.match(title) and len(title.strip()) < MIN_BLOCK_CHARS:
                carry = (carry + "\n" + title).strip() if carry else title
                continue
            hb = next_id()
            blocks.append(
                Block(
                    block_id=hb,
                    doc_id=doc_id,
                    type="heading",
                    text=title,
                    parent_id=None,
                    heading_path=(title,),
                    anchors={},
                )
            )
            stack.append((1, title, hb))
        elif tier == 2:
            flush_paragraph()
            while stack and stack[-1][0] >= 2:
                stack.pop()
            parent_id = stack[-1][2] if stack else None
            path = heading_path_tuple() + (title,) if stack else (title,)
            hb = next_id()
            blocks.append(
                Block(
                    block_id=hb,
                    doc_id=doc_id,
                    type="heading",
                    text=title,
                    parent_id=parent_id,
                    heading_path=path,
                    anchors={},
                )
            )
            stack.append((2, title, hb))
        elif tier == 3:
            flush_paragraph()
            para_buf.append(line.strip())
        elif tier == 4:
            flush_paragraph()
            item = line.strip()
            if len(item) < MIN_BLOCK_CHARS:
                carry = (carry + "\n" + item).strip() if carry else item
                continue
            pid = body_parent_id()
            blocks.append(
                Block(
                    block_id=next_id(),
                    doc_id=doc_id,
                    type="list_item",
                    text=item,
                    parent_id=pid,
                    heading_path=heading_path_tuple(),
                    anchors={},
                )
            )

    flush_paragraph()
    if carry:
        blocks.append(
            Block(
                block_id=next_id(),
                doc_id=doc_id,
                type="paragraph",
                text=carry,
                parent_id=body_parent_id(),
                heading_path=heading_path_tuple(),
                anchors={},
            )
        )
        carry = ""

    g = BlockGraph()
    g.add_document(doc)
    for b in blocks:
        g.add_block(b)
        g.add_edge(Edge(src=doc.doc_id, dst=b.block_id, type="contains"))
        if b.parent_id:
            g.add_edge(Edge(src=b.parent_id, dst=b.block_id, type="parent"))
    for prev, cur in zip(blocks, blocks[1:]):
        g.add_edge(Edge(src=prev.block_id, dst=cur.block_id, type="next"))
    g.build_indexes()
    return doc, blocks, g
