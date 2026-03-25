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
