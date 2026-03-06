from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from ..llm.base import LLMClient
from ..retrieval import RAGNavIndex, RAGNavRetriever
from ..ingest.pdf import PdfIngestOptions, ingest_pdf_bytes_paper
from ..answering.inline_citations import answer_with_inline_citations
from ..cache.sqlite_cache import EmbeddingCache
from ..net import download_pdf


@dataclass(frozen=True)
class PaperRAGConfig:
    max_pages: int = 25
    build_vectors: bool = True
    embed_batch_size: int = 64
    # Retrieval defaults (paper-optimized)
    top_pages: int = 4
    follow_refs: bool = True
    include_next: bool = True
    graph_hops: int = 1
    k_final: int = 10
    max_answer_blocks: int = 8


@dataclass
class PaperRAG:
    """
    A clean, paper-focused API surface.

    This is the recommended entrypoint if you're working primarily with research papers:
    - page-first routing
    - cross-reference following (Figure/Table/Appendix/Section) via `link_to` edges
    """

    llm: LLMClient
    retriever: RAGNavRetriever
    doc_id: str

    @classmethod
    def from_pdf_bytes(
        cls,
        pdf_bytes: bytes,
        *,
        llm: LLMClient,
        pdf_name: str = "paper.pdf",
        metadata: Optional[dict[str, Any]] = None,
        cfg: PaperRAGConfig = PaperRAGConfig(),
        embedding_cache: Optional[EmbeddingCache] = None,
    ) -> "PaperRAG":
        doc, blocks, edges = ingest_pdf_bytes_paper(
            pdf_bytes,
            name=pdf_name,
            metadata=metadata,
            opts=PdfIngestOptions(max_pages=cfg.max_pages, paper_mode=True),
        )
        idx = RAGNavIndex.build(
            documents=[doc],
            blocks=blocks,
            llm=llm,
            embed_batch_size=cfg.embed_batch_size,
            build_vectors=cfg.build_vectors,
            edges=edges,
            embedding_cache=embedding_cache,
        )
        r = RAGNavRetriever(index=idx, llm=llm)
        return cls(llm=llm, retriever=r, doc_id=doc.doc_id)

    @classmethod
    def from_pdf_url(
        cls,
        pdf_url: str,
        *,
        llm: LLMClient,
        pdf_name: str = "paper.pdf",
        metadata: Optional[dict[str, Any]] = None,
        cfg: PaperRAGConfig = PaperRAGConfig(),
        cache_path: Optional[Union[str, Path]] = Path("data") / "paper.pdf",
        embedding_cache: Optional[EmbeddingCache] = None,
        timeout_s: int = 60,
    ) -> "PaperRAG":
        pdf_bytes = download_pdf(pdf_url, out_path=cache_path, timeout_s=timeout_s)
        meta = dict(metadata or {})
        meta.update({"source": "url", "url": pdf_url})
        return cls.from_pdf_bytes(
            pdf_bytes,
            llm=llm,
            pdf_name=pdf_name,
            metadata=meta,
            cfg=cfg,
            embedding_cache=embedding_cache,
        )

    @classmethod
    def from_pdf_file(
        cls,
        path: Union[str, Path],
        *,
        llm: LLMClient,
        pdf_name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        cfg: PaperRAGConfig = PaperRAGConfig(),
        embedding_cache: Optional[EmbeddingCache] = None,
    ) -> "PaperRAG":
        p = Path(path)
        pdf_bytes = p.read_bytes()
        meta = dict(metadata or {})
        meta.update({"source": "file", "path": str(p)})
        return cls.from_pdf_bytes(
            pdf_bytes,
            llm=llm,
            pdf_name=pdf_name or p.name,
            metadata=meta,
            cfg=cfg,
            embedding_cache=embedding_cache,
        )

    def retrieve(self, query: str, *, cfg: PaperRAGConfig = PaperRAGConfig(), **kwargs: Any):
        """
        Paper-optimized retrieval result (blocks + trace).

        kwargs are forwarded to `RAGNavRetriever.retrieve_paper()`.
        """
        return self.retriever.retrieve_paper(
            query,
            allowed_doc_ids={self.doc_id},
            top_pages=cfg.top_pages,
            follow_refs=cfg.follow_refs,
            include_next=cfg.include_next,
            graph_hops=cfg.graph_hops,
            k_final=cfg.k_final,
            **kwargs,
        )

    def answer(self, query: str, *, cfg: PaperRAGConfig = PaperRAGConfig(), **kwargs: Any) -> str:
        """
        Minimal answer helper: retrieve paper evidence then answer strictly from it.
        """
        res = self.retrieve(query, cfg=cfg, **kwargs)
        context = "\n\n".join(b.text for b in res.blocks[: cfg.max_answer_blocks])
        prompt = f"""Answer the question using ONLY the provided context.

Question: {query}

Context:
{context}
"""
        return self.llm.chat(messages=[{"role": "user", "content": prompt}], temperature=0)

    def answer_cited(self, query: str, *, cfg: PaperRAGConfig = PaperRAGConfig(), **kwargs: Any) -> str:
        """
        Answer with inline citations `[[block_id]]` for every sentence.
        """
        res = self.retrieve(query, cfg=cfg, **kwargs)
        cited = answer_with_inline_citations(
            llm=self.llm,
            query=query,
            blocks=res.blocks[: cfg.max_answer_blocks],
            temperature=0,
        )
        return cited.answer

