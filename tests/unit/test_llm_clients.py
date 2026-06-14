from __future__ import annotations

from ragnav.cost import CostTracker, estimate_tokens_from_text
from ragnav.exceptions import BudgetExceededError, RAGNavCacheError, RAGNavError
from ragnav.llm.fake import FakeLLMClient, FakeLLMConfig
from ragnav.llm.instrumented import InstrumentedLLMClient


def test_estimate_tokens_from_text_scales_with_length():
    assert estimate_tokens_from_text("a" * 40) >= estimate_tokens_from_text("a")


def test_fake_llm_chat_records_cost_when_tracker_set():
    tracker = CostTracker(budget_usd=1.0)
    llm = FakeLLMClient(cost_tracker=tracker)
    out = llm.chat(messages=[{"role": "user", "content": "hello"}])
    assert out.startswith("[fake-llm:")
    assert tracker.report().total_cost_usd >= 0.0
    assert tracker.report().calls == 1


def test_instrumented_llm_client_records_stats():
    inner = FakeLLMClient()
    wrapped = InstrumentedLLMClient(inner=inner)
    wrapped.chat(messages=[{"role": "user", "content": "hi"}])
    vecs = wrapped.embed(inputs=["a", "b"])
    assert wrapped.stats.chat_calls == 1
    assert wrapped.stats.embed_calls == 1
    assert wrapped.stats.embedded_texts == 2
    assert len(vecs) == 2


def test_exception_hierarchy():
    assert issubclass(RAGNavCacheError, RAGNavError)
    assert issubclass(BudgetExceededError, RAGNavError)


def test_fake_llm_embed_dimension_configurable():
    llm = FakeLLMClient(cfg=FakeLLMConfig(dim=8))
    vec = llm.embed(inputs=["test"])[0]
    assert len(vec) == 8
