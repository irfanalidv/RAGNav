from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from ..env import load_env
from ..ingest.pdf import PdfIngestOptions, ingest_pdf_bytes
from ..llm.base import LLMClient
from ..retrieval import RAGNavIndex, RAGNavRetriever
from ..cache.sqlite_cache import RetrievalCache
from ..net import download_pdf


@dataclass(frozen=True)
class VectorlessRagConfig:
    max_pages: int = 25
    max_blocks: int = 6
    k_final: int = 6


def vectorless_rag_from_pdf_bytes(
    *,
    pdf_bytes: bytes,
    llm: LLMClient,
    pdf_name: str = "document.pdf",
    cfg: VectorlessRagConfig = VectorlessRagConfig(),
) -> tuple[list[dict], str]:
    """
    Vectorless RAG:
    - No vector DB
    - No precomputed embeddings (build_vectors=False)
    - Retrieval uses BM25 + structure expansion
    """
    doc, blocks = ingest_pdf_bytes(pdf_bytes, name=pdf_name, opts=PdfIngestOptions(max_pages=cfg.max_pages))
    index = RAGNavIndex.build(documents=[doc], blocks=blocks, llm=llm, build_vectors=False)
    retriever = RAGNavRetriever(index=index, llm=llm)

    def retrieve(query: str, *, retrieval_cache: Optional[RetrievalCache] = None) -> list[dict]:
        return retriever.retrieve_raw(
            query,
            max_blocks=cfg.max_blocks,
            k_final=cfg.k_final,
            expand_structure=True,
            use_vectors=False,
            k_vec=0,
            w_vec=0.0,
            retrieval_cache=retrieval_cache,
        )

    return retrieve, doc.doc_id


def vectorless_answer(
    *,
    query: str,
    retrieve_fn,
    llm: LLMClient,
) -> str:
    hits = retrieve_fn(query)
    context = "\n\n".join(h.get("content", "") for h in hits)
    prompt = f"""Answer the question using ONLY the provided context.

Question: {query}

Context:
{context}
"""
    return llm.chat(messages=[{"role": "user", "content": prompt}], temperature=0)


def vectorless_rag_pdf_url(
    *,
    pdf_url: str,
    llm: LLMClient,
    cfg: VectorlessRagConfig = VectorlessRagConfig(),
    cache_path: Optional[Union[str, Path]] = Path("data") / "document.pdf",
) -> tuple[list[dict], str]:
    """
    End-to-end vectorless RAG setup for a PDF URL.
    """
    load_env()
    pdf_bytes = download_pdf(pdf_url, out_path=cache_path)
    retrieve_fn, doc_id = vectorless_rag_from_pdf_bytes(
        pdf_bytes=pdf_bytes, llm=llm, pdf_name=Path(str(cache_path)).name, cfg=cfg
    )
    return retrieve_fn, doc_id

