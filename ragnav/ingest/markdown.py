from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from ..graph import BlockGraph, Edge
from ..models import Block, Document


@dataclass(frozen=True)
class MarkdownIngestOptions:
    doc_id_prefix: str = "md:"


_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def ingest_markdown_string(
    content: str,
    *,
    name: str = "document.md",
    metadata: Optional[dict[str, Any]] = None,
    opts: Optional[MarkdownIngestOptions] = None,
) -> tuple[Document, list[Block]]:
    """
    Turn markdown into blocks: headings, body text as paragraphs, fenced regions as code.

    Hierarchy uses ``heading_path`` and ``parent_id``. Fenced blocks are not scanned as headings.
    """
    opts = opts or MarkdownIngestOptions()
    doc_meta: dict[str, Any] = {"type": "markdown"}
    if metadata:
        doc_meta.update(metadata)
    doc = Document(doc_id=f"{opts.doc_id_prefix}{name}", source=name, metadata=doc_meta)

    lines = content.splitlines()
    n = len(lines)
    blocks: list[Block] = []
    parent_stack: list[tuple[int, str, str]] = []
    i = 0
    bid = 0

    def next_id(prefix: str) -> str:
        nonlocal bid
        out = f"{doc.doc_id}#{prefix}{bid}"
        bid += 1
        return out

    while i < n:
        stripped = lines[i].strip()

        if stripped.startswith("```"):
            fence_start = i
            i += 1
            code_lines: list[str] = []
            while i < n:
                if lines[i].strip().startswith("```"):
                    i += 1
                    break
                code_lines.append(lines[i])
                i += 1
            code_text = "\n".join(code_lines)
            parent_id = parent_stack[-1][2] if parent_stack else None
            heading_path = tuple(p[1] for p in parent_stack)
            blocks.append(
                Block(
                    block_id=next_id("c"),
                    doc_id=doc.doc_id,
                    type="code",
                    text=code_text,
                    parent_id=parent_id,
                    heading_path=heading_path,
                    anchors={"line_start": fence_start + 1, "line_end": i},
                )
            )
            continue

        m = _HEADER_RE.match(lines[i])
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            while parent_stack and parent_stack[-1][0] >= level:
                parent_stack.pop()
            heading_path = tuple(p[1] for p in parent_stack) + (title,)
            parent_id = parent_stack[-1][2] if parent_stack else None
            h_bid = next_id("h")
            blocks.append(
                Block(
                    block_id=h_bid,
                    doc_id=doc.doc_id,
                    type="heading",
                    text=title,
                    parent_id=parent_id,
                    heading_path=heading_path,
                    anchors={"line_start": i + 1, "line_end": i + 1},
                )
            )
            parent_stack.append((level, title, h_bid))
            j = i + 1
            body_lines: list[str] = []
            while j < n:
                if lines[j].strip().startswith("```"):
                    break
                if _HEADER_RE.match(lines[j]):
                    break
                body_lines.append(lines[j])
                j += 1
            while body_lines and not body_lines[-1].strip():
                body_lines.pop()
            body_text = "\n".join(body_lines).strip()
            if body_text:
                blocks.append(
                    Block(
                        block_id=next_id("p"),
                        doc_id=doc.doc_id,
                        type="paragraph",
                        text=body_text,
                        parent_id=h_bid,
                        heading_path=heading_path,
                        anchors={"line_start": i + 2, "line_end": j},
                    )
                )
            i = j
            continue

        j = i
        para_lines: list[str] = []
        while j < n:
            if lines[j].strip().startswith("```"):
                break
            if _HEADER_RE.match(lines[j]):
                break
            para_lines.append(lines[j])
            j += 1
        while para_lines and not para_lines[-1].strip():
            para_lines.pop()
        para_text = "\n".join(para_lines).strip()
        if para_text:
            blocks.append(
                Block(
                    block_id=next_id("p"),
                    doc_id=doc.doc_id,
                    type="paragraph",
                    text=para_text,
                    parent_id=None,
                    heading_path=(),
                    anchors={"line_start": i + 1, "line_end": j},
                )
            )
        i = j

    if not blocks:
        blocks.append(
            Block(
                block_id=next_id("p"),
                doc_id=doc.doc_id,
                type="paragraph",
                text=content.strip(),
                anchors={"line_start": 1, "line_end": max(1, n)},
            )
        )

    return doc, blocks


def ingest_markdown_string_graph(
    content: str,
    *,
    name: str = "document.md",
    metadata: Optional[dict[str, Any]] = None,
    opts: Optional[MarkdownIngestOptions] = None,
) -> BlockGraph:
    """
    Graph-native ingestion (preferred for new pipelines):
    - adds `contains` edges from doc_id -> block_id
    - adds `parent` edges from parent heading -> child block
    - adds `next` edges by line order
    """
    doc, blocks = ingest_markdown_string(content, name=name, metadata=metadata, opts=opts)

    g = BlockGraph()
    g.add_document(doc)
    for b in blocks:
        g.add_block(b)
        g.add_edge(Edge(src=doc.doc_id, dst=b.block_id, type="contains"))
        if b.parent_id:
            g.add_edge(Edge(src=b.parent_id, dst=b.block_id, type="parent"))

    ordered = sorted(blocks, key=lambda x: int(x.anchors.get("line_start", 0) or 0))
    for prev, cur in zip(ordered, ordered[1:]):
        g.add_edge(Edge(src=prev.block_id, dst=cur.block_id, type="next"))
    g.build_indexes()
    return g
