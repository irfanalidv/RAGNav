from __future__ import annotations

import pytest

from ragnav.cost import CostReport, CostTracker, estimate_tokens_from_text
from ragnav.exceptions import BudgetExceededError


def test_estimate_tokens_from_text_minimum_one():
    assert estimate_tokens_from_text("") == 1
    assert estimate_tokens_from_text("abcd") == 1


def test_cost_tracker_record_accumulates():
    t = CostTracker()
    t.record("gpt-4o-mini", 4, 4)
    t.record("gpt-4o-mini", 4, 4)
    r = t.report()
    assert r.calls == 2
    assert r.total_input_tokens == 8
    assert r.total_output_tokens == 8
    expected = (4 * 0.15 + 4 * 0.60) / 1_000_000 * 2
    assert abs(r.total_cost_usd - expected) < 1e-12


def test_cost_tracker_unknown_model_uses_default_pricing():
    t = CostTracker()
    t.record("unknown-model-xyz", 1_000_000, 1_000_000)
    r = t.report()
    assert r.calls == 1
    expected = (1.0 + 3.0)  # per 1M in + out with _DEFAULT_PRICING
    assert abs(r.total_cost_usd - expected) < 1e-9


def test_check_budget_no_budget_never_raises():
    t = CostTracker(budget_usd=None)
    t.record("gpt-4o-mini", 10_000_000, 10_000_000)
    t.check_budget()


def test_check_budget_raises_when_exceeded():
    t = CostTracker(budget_usd=1e-12)
    t.record("gpt-4o-mini", 10_000_000, 10_000_000)
    with pytest.raises(BudgetExceededError) as exc:
        t.check_budget()
    assert "budget" in str(exc.value).lower()


def test_cost_report_summary_includes_fields():
    r = CostReport(
        total_input_tokens=10,
        total_output_tokens=20,
        total_cost_usd=0.00123,
        budget_usd=1.0,
        calls=3,
    )
    s = r.summary()
    assert "calls=3" in s
    assert "10in" in s
    assert "20out" in s
    assert "0.00123" in s
