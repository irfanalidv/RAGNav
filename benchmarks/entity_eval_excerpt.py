from __future__ import annotations

import json
from dataclasses import asdict

from ragnav.eval import EvalCase, score_retrieval
from ragnav.graphrag import EntityGraphRetriever, EntityGraphRetrieverConfig, build_entity_graph
from ragnav.llm.fake import FakeLLMClient
from ragnav.models import Block, Document
from ragnav.retrieval import RAGNavIndex, RAGNavRetriever


def _paper_excerpt_blocks() -> tuple[Document, list[Block]]:
    """
    Offline "realistic excerpt" suite:
    - includes multi-token benchmark names (GLUE, Stanford Question Answering Dataset)
    - includes task phrases (natural language inference, question answering)
    - includes metrics (F1, exact match, accuracy)

    This is intentionally not a full paper; it's a concentrated stress test for extraction.
    """
    doc = Document(doc_id="pdf:excerpt_paper.pdf", source="excerpt_paper.pdf", metadata={"type": "pdf", "mode": "paper"})

    blocks: list[Block] = []

    def b(i: int, *, page: int, heading_path: tuple[str, ...], text: str, typ: str = "paragraph") -> Block:
        return Block(
            block_id=f"{doc.doc_id}#b{i}",
            doc_id=doc.doc_id,
            type=typ,  # type: ignore[arg-type]
            text=text,
            heading_path=heading_path,
            anchors={"page": page, "line_start": 1, "line_end": 1},
        )

    blocks.append(b(0, page=1, typ="heading", heading_path=("Introduction",), text="1 Introduction"))
    blocks.append(
        b(
            1,
            page=1,
            heading_path=("Introduction",),
            text=(
                "We evaluate BERT on the Stanford Question Answering Dataset (SQuAD) for question answering "
                "and report F1 and exact match (EM)."
            ),
        )
    )
    blocks.append(b(2, page=2, typ="heading", heading_path=("Experiments",), text="2 Experiments"))
    blocks.append(
        b(
            3,
            page=2,
            heading_path=("Experiments",),
            text=(
                "We also evaluate RoBERTa on the GLUE benchmark, including MNLI for natural language inference "
                "and SST-2 for classification, and report accuracy."
            ),
        )
    )
    blocks.append(
        b(
            4,
            page=3,
            heading_path=("Datasets",),
            text="SQuAD is a reading comprehension dataset for question answering.",
        )
    )
    return doc, blocks


def main() -> None:
    doc, blocks = _paper_excerpt_blocks()
    by_id = {b.block_id: b for b in blocks}

    eg = build_entity_graph(blocks)
    egr = EntityGraphRetriever(graph=eg, blocks_by_id=by_id)
    cfg = EntityGraphRetrieverConfig(hops=2, max_evidence_blocks=8)

    cases = [
        EvalCase(
            case_id="bert-on-what",
            query="Which dataset was BERT evaluated on?",
            expected_text_substrings={"SQuAD"},
            expected_pages={1},
            tags={"excerpt", "evaluated_on"},
        ),
        EvalCase(
            case_id="what-is-squad-for",
            query="What is SQuAD used for?",
            expected_text_substrings={"reading comprehension"},
            expected_pages={3},
            tags={"excerpt", "dataset_task"},
        ),
        EvalCase(
            case_id="glue-contains-what",
            query="Which tasks are mentioned for GLUE?",
            expected_text_substrings={"natural language inference"},
            expected_pages={2},
            tags={"excerpt", "tasks"},
        ),
    ]

    retrieved_graph = [egr.retrieve(c.query, cfg=cfg)["blocks"] for c in cases]
    m_graph = score_retrieval(cases, retrieved_graph)

    fake = FakeLLMClient()
    idx = RAGNavIndex.build(documents=[doc], blocks=blocks, llm=fake, build_vectors=False)
    rag = RAGNavRetriever(index=idx, llm=fake)
    retrieved_bm25 = [rag.retrieve(c.query, use_vectors=False, k_final=6).blocks for c in cases]
    m_bm25 = score_retrieval(cases, retrieved_bm25)

    out = {
        "suite": "entity_excerpt_v1",
        "n_cases": len(cases),
        "entity_graph": asdict(m_graph),
        "hybrid_bm25": asdict(m_bm25),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

