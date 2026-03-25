from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..graph import BlockGraph, Edge
from ..models import Block, Document
from ..exceptions import RAGNavIngestError
from ..papers import pdf_heuristics as paper


@dataclass(frozen=True)
class PdfIngestOptions:
    doc_id_prefix: str = "pdf:"
    max_pages: Optional[int] = None
    # If true, try to split pages into heading/paragraph/caption blocks and
    # emit cross-reference edges (Figure/Table/Section/Appendix).
    paper_mode: bool = False


def ingest_pdf_bytes(
    data: bytes,
    *,
    name: str = "document.pdf",
    metadata: Optional[dict[str, Any]] = None,
    opts: Optional[PdfIngestOptions] = None,
) -> tuple[Document, list[Block]]:
    """
    Minimal PDF ingestion:
    - Extract text per page (PyMuPDF)
    - Create one block per page with anchors={"page": N}

    Each page becomes a block with ``anchors["page"]`` so retrieval can filter by page.
    """
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        raise RAGNavIngestError("Missing optional dependency `pymupdf`. Install with: pip install -e \".[pdf]\"") from e

    opts = opts or PdfIngestOptions()
    if opts.paper_mode:
        doc, blocks, _edges = ingest_pdf_bytes_paper(
            data,
            name=name,
            metadata=metadata,
            opts=opts,
        )
        return doc, blocks
    doc_meta: dict[str, Any] = {"type": "pdf"}
    if metadata:
        doc_meta.update(metadata)

    doc = Document(doc_id=f"{opts.doc_id_prefix}{name}", source=name, metadata=doc_meta)

    pdf = fitz.open(stream=data, filetype="pdf")
    total_pages = pdf.page_count
    max_pages = opts.max_pages if opts.max_pages is not None else total_pages
    max_pages = min(max_pages, total_pages)

    blocks: list[Block] = []
    for i in range(max_pages):
        page = pdf.load_page(i)
        text = page.get_text("text") or ""
        text = text.strip()
        if not text:
            continue
        blocks.append(
            Block(
                block_id=f"{doc.doc_id}#p{i+1}",
                doc_id=doc.doc_id,
                type="paragraph",
                text=text,
                anchors={"page": i + 1},
            )
        )

    return doc, blocks


def ingest_pdf_bytes_paper(
    data: bytes,
    *,
    name: str = "document.pdf",
    metadata: Optional[dict[str, Any]] = None,
    opts: Optional[PdfIngestOptions] = None,
) -> tuple[Document, list[Block], list[Edge]]:
    """
    Paper-oriented PDF ingestion:
    - Extract text per page (PyMuPDF)
    - Split into heading/paragraph/caption-ish blocks with page anchors
    - Emit `link_to` edges for cross-references: Figure/Table/Section/Appendix

    This is intentionally heuristic (no layout model) but works well for arXiv-style PDFs.
    """
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        raise RAGNavIngestError("Missing optional dependency `pymupdf`. Install with: pip install -e \".[pdf]\"") from e

    opts = opts or PdfIngestOptions()
    doc_meta: dict[str, Any] = {"type": "pdf", "mode": "paper"}
    if metadata:
        doc_meta.update(metadata)
    doc = Document(doc_id=f"{opts.doc_id_prefix}{name}", source=name, metadata=doc_meta)

    pdf = fitz.open(stream=data, filetype="pdf")
    total_pages = pdf.page_count
    max_pages = opts.max_pages if opts.max_pages is not None else total_pages
    max_pages = min(max_pages, total_pages)

    blocks: list[Block] = []
    edges: list[Edge] = []

    # Heading stack: list of (level, title, block_id)
    stack: list[tuple[int, str, str]] = []

    # Reference targets
    fig_by_num: dict[str, str] = {}
    tab_by_num: dict[str, str] = {}
    sec_by_num: dict[str, str] = {}
    app_by_letter: dict[str, str] = {}

    def current_heading_path() -> tuple[str, ...]:
        return tuple([t for _, t, _ in stack])

    def current_parent_id() -> Optional[str]:
        return stack[-1][2] if stack else None

    block_counter = 0

    def add_block(
        *,
        typ: str,
        text: str,
        page: int,
        line_start: int,
        line_end: int,
        parent_id: Optional[str],
        heading_path: tuple[str, ...],
        meta: Optional[dict[str, Any]] = None,
    ) -> Block:
        nonlocal block_counter
        bid = f"{doc.doc_id}#b{block_counter}"
        block_counter += 1
        b = Block(
            block_id=bid,
            doc_id=doc.doc_id,
            type=typ,  # type: ignore[arg-type]
            text=text.strip(),
            parent_id=parent_id,
            heading_path=heading_path,
            anchors={"page": page, "line_start": line_start, "line_end": line_end},
            metadata=meta or {},
        )
        blocks.append(b)
        return b

    prev_block_id: Optional[str] = None

    for page_idx in range(max_pages):
        page_num = page_idx + 1
        page = pdf.load_page(page_idx)
        text = (page.get_text("text") or "").strip()
        if not text:
            continue
        lines = [ln.rstrip() for ln in text.splitlines()]

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            # Captions (Figure/Table) - capture until blank line
            mfig = paper.FIG_CAPTION_RE.match(line)
            mtab = paper.TAB_CAPTION_RE.match(line)
            if mfig or mtab:
                kind = "figure" if mfig else "table"
                m = mfig or mtab
                assert m is not None
                num = m.group(1)
                rest = (m.group(2) or "").strip()
                cap_lines = [line]
                j = i + 1
                while j < len(lines) and lines[j].strip():
                    cap_lines.append(lines[j].strip())
                    j += 1
                cap_text = "\n".join(cap_lines) if rest else "\n".join(cap_lines)
                b = add_block(
                    typ="table" if kind == "table" else "paragraph",
                    text=cap_text,
                    page=page_num,
                    line_start=i + 1,
                    line_end=j,
                    parent_id=current_parent_id(),
                    heading_path=current_heading_path(),
                    meta={"caption_kind": kind, "caption_number": num},
                )
                if kind == "figure":
                    fig_by_num[num] = b.block_id
                else:
                    tab_by_num[num] = b.block_id
                if prev_block_id:
                    edges.append(Edge(src=prev_block_id, dst=b.block_id, type="next", metadata={"page": page_num}))
                prev_block_id = b.block_id
                i = j
                continue

            # Heading line
            level, title, sec_num = paper.is_heading_line(line)
            if level > 0 and title:
                # pop to parent level
                while stack and stack[-1][0] >= level:
                    stack.pop()

                b = add_block(
                    typ="heading",
                    text=title,
                    page=page_num,
                    line_start=i + 1,
                    line_end=i + 1,
                    parent_id=current_parent_id(),
                    heading_path=tuple([t for _, t, _ in stack] + [title]),
                    meta={"section_number": sec_num} if sec_num else {},
                )
                stack.append((level, title, b.block_id))
                if sec_num:
                    sec_by_num[sec_num] = b.block_id

                # Appendix headings like "Appendix A"
                if title.lower().startswith("appendix"):
                    parts = title.split()
                    if len(parts) >= 2 and len(parts[1]) == 1 and parts[1].isalpha():
                        app_by_letter[parts[1].upper()] = b.block_id

                if prev_block_id:
                    edges.append(Edge(src=prev_block_id, dst=b.block_id, type="next", metadata={"page": page_num}))
                prev_block_id = b.block_id
                i += 1
                continue

            # Paragraph: gather until blank line or heading/caption
            para_lines = [line]
            j = i + 1
            while j < len(lines):
                nxt = lines[j].strip()
                if not nxt:
                    break
                # stop if next looks like heading/caption
                if paper.FIG_CAPTION_RE.match(nxt) or paper.TAB_CAPTION_RE.match(nxt):
                    break
                lev2, title2, _ = paper.is_heading_line(nxt)
                if lev2 > 0 and title2:
                    break
                para_lines.append(nxt)
                j += 1

            para_text = "\n".join(para_lines).strip()
            b = add_block(
                typ="paragraph",
                text=para_text,
                page=page_num,
                line_start=i + 1,
                line_end=j,
                parent_id=current_parent_id(),
                heading_path=current_heading_path(),
            )
            if prev_block_id:
                edges.append(Edge(src=prev_block_id, dst=b.block_id, type="next", metadata={"page": page_num}))
            prev_block_id = b.block_id
            i = j + 1 if j < len(lines) and not lines[j].strip() else j

    # Second pass: cross-reference edges
    by_id = {b.block_id: b for b in blocks}
    for b in blocks:
        txt = b.text or ""
        # Figures
        for num in paper.find_fig_mentions(txt):
            dst = fig_by_num.get(num)
            if dst and dst in by_id and dst != b.block_id:
                edges.append(Edge(src=b.block_id, dst=dst, type="link_to", metadata={"ref_kind": "figure", "ref": num}))
        # Tables
        for num in paper.find_tab_mentions(txt):
            dst = tab_by_num.get(num)
            if dst and dst in by_id and dst != b.block_id:
                edges.append(Edge(src=b.block_id, dst=dst, type="link_to", metadata={"ref_kind": "table", "ref": num}))
        # Sections
        for sec in paper.find_sec_mentions(txt):
            dst = sec_by_num.get(sec)
            if dst and dst in by_id and dst != b.block_id:
                edges.append(Edge(src=b.block_id, dst=dst, type="link_to", metadata={"ref_kind": "section", "ref": sec}))
        # Appendices
        for letter in paper.find_app_mentions(txt):
            dst = app_by_letter.get(letter)
            if dst and dst in by_id and dst != b.block_id:
                edges.append(
                    Edge(src=b.block_id, dst=dst, type="link_to", metadata={"ref_kind": "appendix", "ref": letter})
                )

    return doc, blocks, edges


def ingest_pdf_file(
    path: str,
    *,
    name: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    opts: Optional[PdfIngestOptions] = None,
) -> tuple[Document, list[Block]]:
    with open(path, "rb") as f:
        data = f.read()
    return ingest_pdf_bytes(data, name=name or path.split("/")[-1], metadata=metadata, opts=opts)


def ingest_pdf_file_paper(
    path: str,
    *,
    name: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    opts: Optional[PdfIngestOptions] = None,
) -> tuple[Document, list[Block], list[Edge]]:
    with open(path, "rb") as f:
        data = f.read()
    return ingest_pdf_bytes_paper(data, name=name or path.split("/")[-1], metadata=metadata, opts=opts)


def ingest_pdf_bytes_graph(
    data: bytes,
    *,
    name: str = "document.pdf",
    metadata: Optional[dict[str, Any]] = None,
    opts: Optional[PdfIngestOptions] = None,
) -> BlockGraph:
    """
    Graph-native PDF ingestion (preferred for new pipelines).

    - If opts.paper_mode=True, uses `ingest_pdf_bytes_paper()` and includes cross-ref `link_to` edges.
    - Otherwise, falls back to 1-block-per-page but still emits `contains` and `next`.
    - Always emits `parent` edges when `parent_id` is present.
    """
    opts = opts or PdfIngestOptions()
    if opts.paper_mode:
        doc, blocks, edges = ingest_pdf_bytes_paper(data, name=name, metadata=metadata, opts=opts)
    else:
        doc, blocks = ingest_pdf_bytes(data, name=name, metadata=metadata, opts=opts)
        edges = []
        ordered = sorted(blocks, key=lambda x: int(x.anchors.get("page", 0) or 0))
        for prev, cur in zip(ordered, ordered[1:]):
            edges.append(Edge(src=prev.block_id, dst=cur.block_id, type="next"))

    g = BlockGraph()
    g.add_document(doc)
    for b in blocks:
        g.add_block(b)
        g.add_edge(Edge(src=doc.doc_id, dst=b.block_id, type="contains"))
        if b.parent_id:
            g.add_edge(Edge(src=b.parent_id, dst=b.block_id, type="parent"))
    for e in edges:
        g.add_edge(e)
    g.build_indexes()
    return g


def ingest_pdf_file_graph(
    path: str,
    *,
    name: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    opts: Optional[PdfIngestOptions] = None,
) -> BlockGraph:
    with open(path, "rb") as f:
        data = f.read()
    return ingest_pdf_bytes_graph(data, name=name or path.split("/")[-1], metadata=metadata, opts=opts)

