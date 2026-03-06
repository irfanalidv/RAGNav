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
_FENCE_RE = re.compile(r"^\s*```")


def ingest_markdown_string(
    content: str,
    *,
    name: str = "document.md",
    metadata: Optional[dict[str, Any]] = None,
    opts: Optional[MarkdownIngestOptions] = None,
) -> tuple[Document, list[Block]]:
    """
    Minimal, robust ingestion that turns markdown headings/sections into blocks.

    - Preserves hierarchy via heading_path and parent_id.
    - Avoids parsing headings inside fenced code blocks.
    """
    opts = opts or MarkdownIngestOptions()
    doc_meta: dict[str, Any] = {"type": "markdown"}
    if metadata:
        doc_meta.update(metadata)
    doc = Document(doc_id=f"{opts.doc_id_prefix}{name}", source=name, metadata=doc_meta)

    lines = content.splitlines()
    in_code = False

    headings: list[tuple[int, str, int]] = []  # (level, title, line_idx)
    for i, line in enumerate(lines):
        if _FENCE_RE.match(line.strip()):
            in_code = not in_code
            continue
        if in_code:
            continue
        m = _HEADER_RE.match(line)
        if m:
            headings.append((len(m.group(1)), m.group(2).strip(), i))

    if not headings:
        block = Block(
            block_id=f"{doc.doc_id}#b0",
            doc_id=doc.doc_id,
            type="paragraph",
            text=content.strip(),
            anchors={"line_start": 1, "line_end": len(lines)},
        )
        return doc, [block]

    # section spans
    blocks: list[Block] = []
    parent_stack: list[tuple[int, str, str]] = []  # (level, title, block_id)

    for idx, (level, title, start_i) in enumerate(headings):
        end_i = headings[idx + 1][2] - 1 if idx + 1 < len(headings) else len(lines) - 1
        section_text = "\n".join(lines[start_i : end_i + 1]).strip()

        while parent_stack and parent_stack[-1][0] >= level:
            parent_stack.pop()

        heading_path = tuple([p[1] for p in parent_stack] + [title])
        parent_id = parent_stack[-1][2] if parent_stack else None

        block_id = f"{doc.doc_id}#h{idx}"
        blocks.append(
            Block(
                block_id=block_id,
                doc_id=doc.doc_id,
                type="heading",
                text=section_text,
                parent_id=parent_id,
                heading_path=heading_path,
                anchors={"line_start": start_i + 1, "line_end": end_i + 1},
            )
        )

        parent_stack.append((level, title, block_id))

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

