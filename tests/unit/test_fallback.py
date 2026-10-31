from __future__ import annotations

from unittest.mock import MagicMock

from ragnav.models import Block, ConfidenceLevel, RetrievalResult
from ragnav.retrieval import FallbackConfig, QueryFallback


def test_fallback_skips_retry_when_high_confidence():
    retriever = MagicMock()
    retriever.retrieve.return_value = RetrievalResult(
        query="q",
        blocks=[Block(block_id="b", doc_id="d", type="paragraph", text="x")],
        confidence=ConfidenceLevel.HIGH,
        top_score=0.95,
    )
    llm = MagicMock()
    fb = QueryFallback(retriever, llm)
    out = fb.retrieve("q", top_k=5)
    assert out.attempts == 1
    assert out.improved is False
    llm.chat.assert_not_called()


def test_fallback_retries_and_improves_on_low_confidence():
    b = Block(block_id="b", doc_id="d", type="paragraph", text="evidence")
    low = RetrievalResult(
        query="original",
        blocks=[],
        confidence=ConfidenceLevel.LOW,
        top_score=0.2,
    )
    high = RetrievalResult(
        query="variation one",
        blocks=[b],
        confidence=ConfidenceLevel.HIGH,
        top_score=0.92,
    )
    retriever = MagicMock()
    retriever.retrieve.side_effect = [low, high]
    llm = MagicMock()
    llm.chat.return_value = "variation one\n"
    fb = QueryFallback(retriever, llm)
    out = fb.retrieve("original", top_k=5)
    assert out.attempts == 2
    assert out.improved is True
    assert out.winning_query == "variation one"
    assert out.final_result.top_score == 0.92


def test_fallback_max_attempts_one_skips_variation():
    retriever = MagicMock()
    retriever.retrieve.return_value = RetrievalResult(
        query="q",
        blocks=[],
        confidence=ConfidenceLevel.LOW,
        top_score=0.1,
    )
    llm = MagicMock()
    cfg = FallbackConfig(max_attempts=1)
    fb = QueryFallback(retriever, llm, config=cfg)
    out = fb.retrieve("q")
    assert out.attempts == 1
    llm.chat.assert_not_called()


def test_fallback_llm_failure_returns_best_so_far():
    retriever = MagicMock()
    retriever.retrieve.return_value = RetrievalResult(
        query="q",
        blocks=[],
        confidence=ConfidenceLevel.LOW,
        top_score=0.15,
    )
    llm = MagicMock()
    llm.chat.side_effect = RuntimeError("network down")
    fb = QueryFallback(retriever, llm)
    out = fb.retrieve("q")
    assert out.attempts == 1
    assert out.final_result.top_score == 0.15


def test_fallback_prefers_higher_confidence_even_when_top_scores_tie():
    """Normalized top_score is 1.0 for most result lists; confidence must decide."""
    b = Block(block_id="b", doc_id="d", type="paragraph", text="evidence")
    medium = RetrievalResult(
        query="q", blocks=[b], confidence=ConfidenceLevel.MEDIUM, top_score=1.0
    )
    high = RetrievalResult(query="v", blocks=[b], confidence=ConfidenceLevel.HIGH, top_score=1.0)
    retriever = MagicMock()
    retriever.retrieve.side_effect = [medium, high]
    llm = MagicMock()
    llm.chat.return_value = "v\n"
    out = QueryFallback(retriever, llm).retrieve("q")
    assert out.improved is True
    assert out.winning_query == "v"
    assert out.final_result.confidence is ConfidenceLevel.HIGH


def test_fallback_end_to_end_with_real_retriever():
    """Against a real index the variation must be able to win (it could not before)."""
    from ragnav.models import Document
    from ragnav.retrieval import RAGNavIndex, RAGNavRetriever

    blocks = [
        Block(block_id="r", doc_id="d", type="paragraph", text="Refund window is 30 days."),
        Block(block_id="t", doc_id="d", type="paragraph", text="Tokens expire after 30 days."),
        Block(block_id="o", doc_id="d", type="paragraph", text="Office hours are 9 to 5."),
    ]
    index = RAGNavIndex.build(documents=[Document(doc_id="d")], blocks=blocks, build_vectors=False)
    retriever = RAGNavRetriever(index=index)
    llm = MagicMock()
    llm.chat.return_value = "refund window\n"
    out = QueryFallback(retriever, llm).retrieve(
        "30 days", use_vectors=False, expand_structure=False
    )
    first = retriever.retrieve("30 days", use_vectors=False, expand_structure=False)
    assert first.confidence is ConfidenceLevel.MEDIUM  # two equally good BM25 hits
    assert out.winning_query == "refund window"
    assert out.final_result.blocks[0].block_id == "r"
