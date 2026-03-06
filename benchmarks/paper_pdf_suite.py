from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ragnav.eval import EvalCase, score_retrieval
from ragnav.graphrag import EntityGraphRetriever, EntityGraphRetrieverConfig, build_entity_graph
from ragnav.llm.fake import FakeLLMClient
from ragnav.retrieval import RAGNavIndex, RAGNavRetriever
from ragnav.ingest.pdf import PdfIngestOptions, ingest_pdf_bytes_paper


def _load_pdfs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.glob("*.pdf") if p.is_file()])


def _load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _cases_from_manifest(manifest: dict[str, Any], *, pdf_name: str) -> list[EvalCase] | None:
    papers = manifest.get("papers")
    if not isinstance(papers, list):
        return None
    for p in papers:
        if not isinstance(p, dict):
            continue
        if p.get("file") != pdf_name:
            continue
        cases_in = p.get("cases")
        if not isinstance(cases_in, list):
            return None
        out: list[EvalCase] = []
        for c in cases_in:
            if not isinstance(c, dict):
                continue
            out.append(
                EvalCase(
                    case_id=str(c.get("case_id") or c.get("query") or "case"),
                    query=str(c.get("query") or ""),
                    expected_pages=set(int(x) for x in (c.get("expected_pages") or []) if isinstance(x, int) or str(x).isdigit()),
                    expected_text_substrings=set(str(x) for x in (c.get("expected_text_substrings") or []) if isinstance(x, (str, int))),
                    tags=set(str(x) for x in (c.get("tags") or []) if isinstance(x, (str, int))),
                )
            )
        return out or None
    return None


def main() -> None:
    """
    Optional "real PDF" suite:
    - Looks for PDFs under `data/papers/*.pdf`
    - If present and `pymupdf` is installed, runs paper ingest + retrieval checks.
    - If not, prints a JSON report with `skipped=true`.
    """
    pdf_root = Path("data") / "papers"
    manifest_path = pdf_root / "manifest.json"
    manifest = _load_manifest(manifest_path)
    pdfs = _load_pdfs(pdf_root)
    if not pdfs:
        print(
            json.dumps(
                {
                    "suite": "paper_pdf_suite_v1",
                    "skipped": True,
                    "reason": "No PDFs found under data/papers/*.pdf",
                },
                indent=2,
            )
        )
        return

    # Check pymupdf availability indirectly (ingest will raise otherwise).
    fake = FakeLLMClient()

    per_pdf: list[dict[str, Any]] = []
    for p in pdfs:
        try:
            pdf_bytes = p.read_bytes()
            doc, blocks, edges = ingest_pdf_bytes_paper(
                pdf_bytes,
                name=p.name,
                metadata={"source": "local"},
                opts=PdfIngestOptions(max_pages=15, paper_mode=True),
            )
        except Exception as e:
            per_pdf.append({"pdf": p.name, "ok": False, "error": f"{type(e).__name__}: {e}"})
            continue

        # BM25-only index for deterministic offline behavior
        idx = RAGNavIndex.build(documents=[doc], blocks=blocks, llm=fake, build_vectors=False, edges=edges)
        rag = RAGNavRetriever(index=idx, llm=fake)

        # GraphRAG entity layer
        eg = build_entity_graph(blocks)
        egr = EntityGraphRetriever(graph=eg, blocks_by_id={b.block_id: b for b in blocks})
        cfg = EntityGraphRetrieverConfig(hops=2, max_evidence_blocks=10)

        cases = None
        if manifest is not None:
            cases = _cases_from_manifest(manifest, pdf_name=p.name)
        if cases is None:
            # Generic queries: these should usually find something in real papers, but we keep them loose.
            cases = [
                EvalCase(case_id="datasets", query="Which datasets are mentioned?", expected_pages=set()),
                EvalCase(case_id="metrics", query="Which metrics are reported?", expected_pages=set()),
            ]

        retrieved_graph = [egr.retrieve(c.query, cfg=cfg)["blocks"] for c in cases]
        m_graph = score_retrieval(cases, retrieved_graph)

        retrieved_nav = [rag.retrieve_paper(c.query, allowed_doc_ids={doc.doc_id}, use_vectors=False).blocks for c in cases]
        m_nav = score_retrieval(cases, retrieved_nav)

        per_pdf.append(
            {
                "pdf": p.name,
                "ok": True,
                "n_blocks": len(blocks),
                "n_edges": len(edges),
                "manifest_used": manifest is not None and _cases_from_manifest(manifest, pdf_name=p.name) is not None,
                "n_cases": len(cases),
                "entity_graph_metrics": asdict(m_graph),
                "paper_nav_metrics": asdict(m_nav),
                "n_entities": len(eg.entities),
                "n_relations": len(eg.relations),
            }
        )

    ok = all(x.get("ok") is True for x in per_pdf)
    print(
        json.dumps(
            {
                "suite": "paper_pdf_suite_v1",
                "skipped": False,
                "ok": ok,
                "pdf_root": str(pdf_root),
                "manifest_path": str(manifest_path),
                "manifest_loaded": manifest is not None,
                "n_pdfs": len(pdfs),
                "per_pdf": per_pdf,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

