from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from ..env import load_env
from ..ingest.pdf import PdfIngestOptions, ingest_pdf_bytes
from ..llm.base import LLMClient
from ..retrieval import RAGNavIndex, RAGNavRetriever
from .vectorless import download_pdf
from ..cache.sqlite_cache import EmbeddingCache, RetrievalCache


@dataclass(frozen=True)
class HybridRagConfig:
    max_pages: int = 25
    max_blocks: int = 6
    k_final: int = 6
    k_bm25: int = 30
    k_vec: int = 30
    w_bm25: float = 0.45
    w_vec: float = 0.55
    embed_batch_size: int = 64


def hybrid_rag_pdf_url(
    *,
    pdf_url: str,
    llm: LLMClient,
    cfg: HybridRagConfig = HybridRagConfig(),
    cache_path: Optional[Union[str, Path]] = Path("data") / "document.pdf",
    embedding_cache: Optional[EmbeddingCache] = None,
    retrieval_cache: Optional[RetrievalCache] = None,
) -> tuple[RAGNavRetriever, str]:
    """
    Hybrid RAG over a PDF URL:
    - Candidate generation: BM25 + embeddings
    - Deterministic structure expansion
    """
    load_env()
    pdf_bytes = download_pdf(pdf_url, out_path=cache_path)
    pdf_name = Path(str(cache_path)).name if cache_path is not None else "document.pdf"

    doc, blocks = ingest_pdf_bytes(
        pdf_bytes,
        name=pdf_name,
        metadata={"source": "url", "url": pdf_url},
        opts=PdfIngestOptions(max_pages=cfg.max_pages),
    )
    index = RAGNavIndex.build(
        documents=[doc],
        blocks=blocks,
        llm=llm,
        embed_batch_size=cfg.embed_batch_size,
        build_vectors=True,
        embedding_cache=embedding_cache,
    )
    retriever = RAGNavRetriever(index=index, llm=llm)
    return retriever, doc.doc_id


def hybrid_answer(
    *,
    query: str,
    retriever: RAGNavRetriever,
    llm: LLMClient,
    cfg: HybridRagConfig,
    retrieval_cache: Optional[RetrievalCache] = None,
) -> str:
    hits = retriever.retrieve_raw(
        query,
        max_blocks=cfg.max_blocks,
        k_final=cfg.k_final,
        k_bm25=cfg.k_bm25,
        k_vec=cfg.k_vec,
        w_bm25=cfg.w_bm25,
        w_vec=cfg.w_vec,
        use_vectors=True,
        expand_structure=True,
        retrieval_cache=retrieval_cache,
    )
    context = "\n\n".join(h.get("content", "") for h in hits)
    prompt = f"""Answer the question using ONLY the provided context.

Question: {query}

Context:
{context}
"""
    return llm.chat(messages=[{"role": "user", "content": prompt}], temperature=0)

