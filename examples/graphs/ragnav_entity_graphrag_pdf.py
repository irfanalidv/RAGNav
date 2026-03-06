from __future__ import annotations

from pathlib import Path

from ragnav.env import load_env
from ragnav.graphrag import EntityGraphRetriever, EntityGraphRetrieverConfig, build_entity_graph
from ragnav.ingest.pdf import PdfIngestOptions, ingest_pdf_bytes_paper
from ragnav.llm.mistral import MistralClient
from ragnav.retrieval import RAGNavIndex, RAGNavRetriever
from ragnav.utils import print_wrapped


PDF_URL = "https://arxiv.org/pdf/2507.13334.pdf"


def main() -> None:
    """
    Networked demo (requires requests + pymupdf + Mistral key):
    - download an arXiv PDF
    - ingest in paper mode (blocks + cross-ref edges)
    - build an entity graph from blocks (GraphRAG layer)
    - answer a few entity-centric queries by returning supporting blocks
    """
    load_env()

    try:
        import requests
    except Exception as e:
        raise RuntimeError("Missing optional dependency `requests`. Install with: pip install -e \".[pdf]\"") from e

    llm = MistralClient()

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = data_dir / "entity_demo_paper.pdf"

    if not pdf_path.exists():
        resp = requests.get(PDF_URL, timeout=60)
        resp.raise_for_status()
        pdf_path.write_bytes(resp.content)

    pdf_bytes = pdf_path.read_bytes()
    doc, blocks, edges = ingest_pdf_bytes_paper(
        pdf_bytes,
        name=pdf_path.name,
        metadata={"source": "arxiv", "url": PDF_URL},
        opts=PdfIngestOptions(max_pages=25, paper_mode=True),
    )

    # Standard retrieval index (optional fallback)
    idx = RAGNavIndex.build(documents=[doc], blocks=blocks, llm=llm, edges=edges, build_vectors=True)
    rag = RAGNavRetriever(index=idx, llm=llm)

    # GraphRAG entity graph
    eg = build_entity_graph(blocks)
    by_id = {b.block_id: b for b in blocks}
    egr = EntityGraphRetriever(graph=eg, blocks_by_id=by_id)

    queries = [
        "Which datasets are mentioned in the experiments?",
        "What metrics are reported?",
    ]
    cfg = EntityGraphRetrieverConfig(hops=2, max_evidence_blocks=10)

    for q in queries:
        print("\n## Query\n")
        print(q)

        out = egr.retrieve(q, cfg=cfg)
        blocks_out = out["blocks"]
        if not blocks_out:
            print("\nNo graph evidence found; falling back to hybrid retrieval.\n")
            res = rag.retrieve_paper(q, allowed_doc_ids={doc.doc_id})
            blocks_out = res.blocks[:8]

        print("\n## Evidence\n")
        for b in blocks_out[:6]:
            print(f"- page={b.anchors.get('page')}  id={b.block_id}")
            print_wrapped(b.text[:600])
            print("")


if __name__ == "__main__":
    main()

