from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..exceptions import RAGNavIngestError
from ..graph import BlockGraph, Edge
from ..models import Block, Document


@dataclass(frozen=True)
class HtmlIngestOptions:
    doc_id_prefix: str = "html:"
    max_chars_per_block: int = 4000


def _normalize_text(s: str) -> str:
    return " ".join((s or "").split())


def ingest_html_string_graph(
    html: str,
    *,
    name: str = "page.html",
    url: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    opts: Optional[HtmlIngestOptions] = None,
) -> BlockGraph:
    """
    HTML → BlockGraph
    - headings and paragraphs become blocks
    - sequential `next` edges
    - `link_to` edges for anchor hrefs (best-effort)
    """
    try:
        from bs4 import BeautifulSoup
    except Exception as e:
        raise RAGNavIngestError(
            "Missing optional dependency `beautifulsoup4`. Install with: pip install -e \".[messy]\""
        ) from e

    opts = opts or HtmlIngestOptions()
    doc_meta: dict[str, Any] = {"type": "html"}
    if url:
        doc_meta["url"] = url
    if metadata:
        doc_meta.update(metadata)

    doc = Document(doc_id=f"{opts.doc_id_prefix}{name}", source=url or name, metadata=doc_meta)
    g = BlockGraph()
    g.add_document(doc)

    soup = BeautifulSoup(html, "html.parser")

    # Basic main content heuristic: prefer <main>, else <body>
    root = soup.find("main") or soup.find("article") or soup.body or soup

    blocks: list[Block] = []
    links: list[tuple[str, str]] = []  # (block_id, href)

    idx = 0
    for el in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"], recursive=True):
        text = _normalize_text(el.get_text(" ", strip=True))
        if not text:
            continue
        text = text[: opts.max_chars_per_block]
        tag = el.name.lower()
        btype = "heading" if tag.startswith("h") else ("list_item" if tag == "li" else "paragraph")

        block_id = f"{doc.doc_id}#n{idx}"
        idx += 1
        b = Block(
            block_id=block_id,
            doc_id=doc.doc_id,
            type=btype,  # type: ignore[arg-type]
            text=text,
            anchors={"url": url, "tag": tag},
        )
        blocks.append(b)
        g.add_block(b)

        for a in el.find_all("a", href=True):
            href = str(a.get("href"))
            if href:
                links.append((block_id, href))

    # next edges
    for a, b in zip(blocks, blocks[1:]):
        g.add_edge(Edge(src=a.block_id, dst=b.block_id, type="next"))

    # link edges (href is stored; dst unknown without resolution)
    for src_id, href in links:
        g.add_edge(Edge(src=src_id, dst=f"{doc.doc_id}#href:{href}", type="link_to", metadata={"href": href}))

    g.build_indexes()
    return g

