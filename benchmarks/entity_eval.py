from __future__ import annotations

import json
from dataclasses import asdict

from ragnav.eval import EvalCase, score_retrieval
from ragnav.graphrag import EntityGraphRetriever, EntityGraphRetrieverConfig, build_entity_graph
from ragnav.llm.fake import FakeLLMClient
from ragnav.models import Block, Document
from ragnav.retrieval import RAGNavIndex, RAGNavRetriever


def _synthetic_multihop_blocks() -> tuple[Document, list[Block]]:
    doc = Document(doc_id="pdf:entity_paper.pdf", source="entity_paper.pdf", metadata={"type": "pdf", "mode": "paper"})

    blocks: list[Block] = []

    def b(i: int, *, page: int, text: str, typ: str = "paragraph", heading_path=()) -> Block:
        return Block(
            block_id=f"{doc.doc_id}#b{i}",
            doc_id=doc.doc_id,
            type=typ,  # type: ignore[arg-type]
            text=text,
            heading_path=heading_path,
            anchors={"page": page, "line_start": 1, "line_end": 1},
        )

    blocks.append(b(0, page=1, typ="heading", text="2 Experiments", heading_path=("Experiments",)))
    # Relation: BERT evaluated on SQuAD; metric F1.
    blocks.append(
        b(
            1,
            page=1,
            text="We evaluated BERT on the SQuAD dataset for question answering. We report F1 and accuracy.",
            heading_path=("Experiments",),
        )
    )
    # Second-hop: SQuAD described as reading comprehension QA.
    blocks.append(
        b(
            2,
            page=2,
            text="SQuAD is a reading comprehension dataset for question answering.",
            heading_path=("Datasets",),
        )
    )
    # Another entity: MNLI classification.
    blocks.append(
        b(
            3,
            page=2,
            text="We evaluated RoBERTa on the MNLI dataset for classification and report accuracy.",
            heading_path=("Experiments",),
        )
    )
    return doc, blocks


def main() -> None:
    doc, blocks = _synthetic_multihop_blocks()
    by_id = {b.block_id: b for b in blocks}

    # Build entity graph from blocks
    eg = build_entity_graph(blocks)
    egr = EntityGraphRetriever(graph=eg, blocks_by_id=by_id)

    cases = [
        EvalCase(
            case_id="bert-squad",
            query="Which dataset was BERT evaluated on?",
            expected_text_substrings={"SQuAD"},
            expected_pages={1},
            tags={"graph", "evaluated_on"},
        ),
        EvalCase(
            case_id="squad-task",
            query="What is SQuAD used for?",
            expected_text_substrings={"reading comprehension"},
            expected_pages={2},
            tags={"graph", "multihop"},
        ),
    ]

    cfg = EntityGraphRetrieverConfig(hops=2, max_evidence_blocks=8)
    retrieved = [egr.retrieve(c.query, cfg=cfg)["blocks"] for c in cases]
    metrics = score_retrieval(cases, retrieved)

    # Compare against plain hybrid retrieval (BM25-only since fake embeddings aren't meaningful)
    fake = FakeLLMClient()
    idx = RAGNavIndex.build(documents=[doc], blocks=blocks, llm=fake, build_vectors=False)
    rag = RAGNavRetriever(index=idx, llm=fake)
    retrieved_hybrid = [rag.retrieve(c.query, use_vectors=False, k_final=6).blocks for c in cases]
    metrics_hybrid = score_retrieval(cases, retrieved_hybrid)

    out = {
        "suite": "entity_multihop_v1",
        "n_cases": len(cases),
        "entity_graph": asdict(metrics),
        "hybrid_bm25": asdict(metrics_hybrid),
        "note": "Synthetic baseline; expand with real-paper suites next.",
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

