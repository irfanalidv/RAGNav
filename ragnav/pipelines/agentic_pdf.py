from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from ..llm.base import LLMClient
from ..cache.sqlite_cache import EmbeddingCache, RetrievalCache
from .agentic import AgenticConfig, agentic_retrieve_then_answer
from .hybrid import HybridRagConfig, hybrid_rag_pdf_url


@dataclass(frozen=True)
class AgenticPdfConfig:
    retrieval: HybridRagConfig = HybridRagConfig()
    agent: AgenticConfig = AgenticConfig()
    max_tool_blocks: int = 8


def agentic_pdf_url(
    *,
    pdf_url: str,
    query: str,
    llm: LLMClient,
    cfg: AgenticPdfConfig = AgenticPdfConfig(),
    cache_path: Optional[Union[str, Path]] = Path("data") / "document.pdf",
    embedding_cache: Optional[EmbeddingCache] = None,
    retrieval_cache: Optional[RetrievalCache] = None,
) -> tuple[str, str]:
    """
    Agentic retrieval over a PDF URL (hybrid retrieval tool).

    Returns (answer, doc_id).
    """
    retriever, doc_id = hybrid_rag_pdf_url(
        pdf_url=pdf_url,
        llm=llm,
        cfg=cfg.retrieval,
        cache_path=cache_path,
        embedding_cache=embedding_cache,
        retrieval_cache=retrieval_cache,
    )

    def tool(q: str, max_blocks: int) -> list[dict]:
        return retriever.retrieve_raw(
            q,
            max_blocks=min(max_blocks, cfg.max_tool_blocks),
            retrieval_cache=retrieval_cache,
        )

    answer = agentic_retrieve_then_answer(query=query, llm=llm, retrieve_raw=tool, cfg=cfg.agent)
    return answer, doc_id

