from __future__ import annotations

from . import pdf_heuristics

__all__ = [
    "PaperRAG",
    "PaperRAGConfig",
    "pdf_heuristics",
]


def __getattr__(name: str):
    if name == "PaperRAG":
        from .rag import PaperRAG

        return PaperRAG
    if name == "PaperRAGConfig":
        from .rag import PaperRAGConfig

        return PaperRAGConfig
    raise AttributeError(name)
