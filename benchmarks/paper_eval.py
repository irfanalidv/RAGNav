from __future__ import annotations

import json
from dataclasses import asdict

from ragnav.eval import EvalCase, score_retrieval
from ragnav.graph import Edge
from ragnav.llm.fake import FakeLLMClient
from ragnav.models import Block, Document
from ragnav.retrieval import RAGNavIndex, RAGNavRetriever


def _build_synthetic_paper_index() -> tuple[RAGNavRetriever, dict[str, Block]]:
    """
    Build a tiny synthetic "paper" with cross references.

    Goal: measure whether `retrieve_paper(... follow_refs=True ...)` can pull
    referenced evidence blocks even when page routing would otherwise omit them.
    """
    doc = Document(doc_id="pdf:synthetic_paper.pdf", source="synthetic_paper.pdf", metadata={"type": "pdf", "mode": "paper"})

    blocks: list[Block] = []

    def b(i: int, *, page: int, text: str, typ: str = "paragraph", meta: dict | None = None) -> Block:
        return Block(
            block_id=f"{doc.doc_id}#b{i}",
            doc_id=doc.doc_id,
            type=typ,  # type: ignore[arg-type]
            text=text,
            anchors={"page": page, "line_start": 1, "line_end": 1},
            metadata=meta or {},
        )

    # Page 1: mentions figure/table/appendix/section.
    blocks.append(b(0, page=1, typ="heading", text="1 Introduction", meta={"section_number": "1"}))
    blocks.append(b(1, page=1, text="We summarize results in Figure 1 and Table 2. See Appendix A for details."))
    blocks.append(b(2, page=1, typ="heading", text="2 Methods", meta={"section_number": "2"}))
    blocks.append(b(3, page=1, text="Our approach is described in Section 2.1."))

    # Page 2: captions / targets
    blocks.append(b(4, page=2, typ="heading", text="2.1 Model", meta={"section_number": "2.1"}))
    blocks.append(b(5, page=2, text="Figure 1: Accuracy improves when following cross-references.", meta={"caption_kind": "figure", "caption_number": "1"}))
    blocks.append(b(6, page=2, typ="table", text="Table 2: Ablation results across datasets.", meta={"caption_kind": "table", "caption_number": "2"}))

    # Page 3: appendix
    blocks.append(b(7, page=3, typ="heading", text="Appendix A Additional Details"))
    blocks.append(b(8, page=3, text="Appendix A: Hyperparameters and extra experiments."))

    by_id = {x.block_id: x for x in blocks}

    # Cross-reference edges: from mentions -> targets
    edges = [
        Edge(src=by_id[f"{doc.doc_id}#b1"].block_id, dst=by_id[f"{doc.doc_id}#b5"].block_id, type="link_to", metadata={"ref_kind": "figure", "ref": "1"}),
        Edge(src=by_id[f"{doc.doc_id}#b1"].block_id, dst=by_id[f"{doc.doc_id}#b6"].block_id, type="link_to", metadata={"ref_kind": "table", "ref": "2"}),
        # Point directly at an evidence-bearing appendix paragraph (not only the heading),
        # so we can measure "did we fetch the actual content?".
        Edge(src=by_id[f"{doc.doc_id}#b1"].block_id, dst=by_id[f"{doc.doc_id}#b8"].block_id, type="link_to", metadata={"ref_kind": "appendix", "ref": "A"}),
        Edge(src=by_id[f"{doc.doc_id}#b3"].block_id, dst=by_id[f"{doc.doc_id}#b4"].block_id, type="link_to", metadata={"ref_kind": "section", "ref": "2.1"}),
    ]

    fake = FakeLLMClient()
    idx = RAGNavIndex.build(documents=[doc], blocks=blocks, llm=fake, build_vectors=False, edges=edges)
    r = RAGNavRetriever(index=idx, llm=fake)
    return r, by_id


def main() -> None:
    retriever, by_id = _build_synthetic_paper_index()

    doc_id = "pdf:synthetic_paper.pdf"

    cases = [
        EvalCase(
            case_id="fig_ref",
            query="What does Figure 1 show?",
            expected_block_ids={by_id[f"{doc_id}#b5"].block_id},
            expected_pages={2},
            tags={"crossref", "figure"},
        ),
        EvalCase(
            case_id="table_ref",
            query="What is reported in Table 2?",
            expected_block_ids={by_id[f"{doc_id}#b6"].block_id},
            expected_pages={2},
            tags={"crossref", "table"},
        ),
        EvalCase(
            case_id="appendix_ref",
            query="What is in Appendix A?",
            expected_block_ids={by_id[f"{doc_id}#b8"].block_id},
            expected_pages={3},
            tags={"crossref", "appendix"},
        ),
        EvalCase(
            case_id="section_ref",
            query="What is described in Section 2.1?",
            expected_block_ids={by_id[f"{doc_id}#b4"].block_id},
            expected_pages={2},
            tags={"crossref", "section"},
        ),
    ]

    # Compare follow_refs ON vs OFF under aggressive page routing (top_pages=1).
    retrieved_on = [
        retriever.retrieve_paper(
            c.query,
            allowed_doc_ids={doc_id},
            top_pages=1,
            follow_refs=True,
            include_next=False,
            use_vectors=False,
            k_final=6,
        ).blocks
        for c in cases
    ]
    retrieved_off = [
        retriever.retrieve_paper(
            c.query,
            allowed_doc_ids={doc_id},
            top_pages=1,
            follow_refs=False,
            include_next=False,
            use_vectors=False,
            k_final=6,
        ).blocks
        for c in cases
    ]

    m_on = score_retrieval(cases, retrieved_on)
    m_off = score_retrieval(cases, retrieved_off)

    summary = {
        "suite": "paper_crossref_v1",
        "n_cases": len(cases),
        "follow_refs_true": asdict(m_on),
        "follow_refs_false": asdict(m_off),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

