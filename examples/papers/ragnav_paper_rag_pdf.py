from __future__ import annotations

import argparse
from pathlib import Path

from ragnav.env import load_env
from ragnav.ingest.pdf import PdfIngestOptions, ingest_pdf_bytes_paper
from ragnav.llm.mistral import MistralClient
from ragnav.net import download_pdf
from ragnav.retrieval import RAGNavIndex, RAGNavRetriever
from ragnav.utils import print_wrapped


PDF_URL = "https://arxiv.org/pdf/2507.13334.pdf"


def main(argv: list[str] | None = None) -> None:
    """
    Paper-optimized demo:
    - paper-mode PDF ingest (headings/paragraphs/captions + cross-ref edges)
    - page-first routing
    - cross-reference following via `link_to`
    """
    parser = argparse.ArgumentParser(description="RAGNav paper-mode RAG over a PDF URL.")
    parser.add_argument("--pdf-url", default=PDF_URL, help="PDF URL (e.g. arXiv).")
    parser.add_argument("--pdf-name", default="paper.pdf", help="Local filename used for doc_id + caching.")
    parser.add_argument(
        "--query",
        default="What experiments were conducted? If a figure summarizes results, cite it.",
        help="Question to answer from retrieved evidence.",
    )
    parser.add_argument("--max-pages", type=int, default=25, help="Max pages to ingest from the PDF.")
    args = parser.parse_args(argv)

    load_env()

    llm = MistralClient()

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = data_dir / args.pdf_name
    pdf_bytes = download_pdf(args.pdf_url, out_path=pdf_path, timeout_s=60)

    doc, blocks, edges = ingest_pdf_bytes_paper(
        pdf_bytes,
        name=pdf_path.name,
        metadata={"source": "url", "url": args.pdf_url},
        opts=PdfIngestOptions(max_pages=args.max_pages, paper_mode=True),
    )
    index = RAGNavIndex.build(documents=[doc], blocks=blocks, llm=llm, edges=edges)
    retriever = RAGNavRetriever(index=index, llm=llm)

    query = args.query
    res = retriever.retrieve_paper(query, top_pages=4, follow_refs=True, include_next=True)

    print("## Routed pages")
    for doc_id, page, score, n in res.trace.get("routed_pages", []):
        print(f"- doc_id={doc_id} page={page} score={score:.4f} N={n}")

    print("\n## Retrieved evidence blocks (first 10)")
    for b in res.blocks[:10]:
        title = " > ".join(b.heading_path) if b.heading_path else None
        print(f"- page={b.anchors.get('page')}  title={title}  id={b.block_id}")

    context = "\n\n".join(b.text for b in res.blocks[:8])
    prompt = f"""Answer the question using ONLY the provided context.

Question: {query}

Context:
{context}
"""
    ans = llm.chat(messages=[{"role": "user", "content": prompt}], temperature=0)
    print("\n## Answer\n")
    print_wrapped(ans)


if __name__ == "__main__":
    main()

