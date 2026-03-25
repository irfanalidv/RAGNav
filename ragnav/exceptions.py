from __future__ import annotations


class RAGNavError(Exception):
    """Base for all ragnav exceptions."""


class RAGNavCacheError(RAGNavError):
    """Cache open, read, or write failed."""


class RAGNavIngestError(RAGNavError):
    """Document ingestion failed — malformed input or missing optional dependency."""


class RAGNavEmbeddingError(RAGNavError):
    """Embedding call failed or returned unexpected shape."""


class RAGNavLLMError(RAGNavError):
    """LLM call failed or returned unparseable output."""


class RAGNavPolicyViolation(RAGNavError):
    """Retrieved content was blocked by the active ContentPolicy."""


class BudgetExceededError(RAGNavError):
    """Cumulative LLM cost reached the configured ``budget_usd`` limit."""
