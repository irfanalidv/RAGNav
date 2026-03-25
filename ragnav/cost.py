"""
Token usage tracking and budget enforcement for RAGNav LLM calls.

Attach a ``CostTracker`` to a concrete ``LLMClient`` (e.g. ``MistralClient``) or pass
one through ``RAGNavRetriever`` so routing and tree-search prompts accrue visible cost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .exceptions import BudgetExceededError

logger = logging.getLogger(__name__)


# Approximate USD per 1M tokens (input, output). Estimates only — check provider pricing.
_PRICING: dict[str, tuple[float, float]] = {
    "mistral-small": (0.20, 0.60),
    "mistral-medium": (2.70, 8.10),
    "mistral-large": (3.00, 9.00),
    "mistral-large-latest": (3.00, 9.00),
    "open-mistral-7b": (0.25, 0.25),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}
_DEFAULT_PRICING = (1.00, 3.00)


def estimate_tokens_from_text(text: str) -> int:
    """Rough token count without tiktoken (≈4 characters per token)."""
    return max(1, len(text) // 4)


@dataclass
class CostReport:
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    budget_usd: Optional[float]
    calls: int

    def summary(self) -> str:
        budget_str = "/%.4f" % self.budget_usd if self.budget_usd is not None else ""
        return (
            "calls=%d  tokens=%din+%dout  cost=$%.5f%s"
            % (
                self.calls,
                self.total_input_tokens,
                self.total_output_tokens,
                self.total_cost_usd,
                budget_str,
            )
        )


@dataclass
class CostTracker:
    """
    Records estimated token usage and USD cost after each successful LLM chat call.

    Call :meth:`check_budget` **immediately before** each planned LLM request; call
    :meth:`record` **after** the response returns so the next ``check_budget`` sees
    updated spend.
    """

    budget_usd: Optional[float] = None
    _calls: int = field(default=0, repr=False)
    _input_tokens: int = field(default=0, repr=False)
    _output_tokens: int = field(default=0, repr=False)
    _cost_usd: float = field(default=0.0, repr=False)

    def record(self, model: str, input_tokens: int, output_tokens: int) -> None:
        in_price, out_price = _PRICING.get(model, _DEFAULT_PRICING)
        cost = (input_tokens * in_price + output_tokens * out_price) / 1_000_000
        self._calls += 1
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens
        self._cost_usd += cost
        logger.debug(
            "llm call #%d model=%r tokens=%d+%d cost=$%.5f total=$%.5f",
            self._calls,
            model,
            input_tokens,
            output_tokens,
            cost,
            self._cost_usd,
        )

    def check_budget(self) -> None:
        """
        Abort before the next LLM call if spend already meets or exceeds ``budget_usd``.

        No-op when ``budget_usd`` is unset. Invoke this at the start of each chat attempt.
        """
        if self.budget_usd is not None and self._cost_usd >= self.budget_usd:
            raise BudgetExceededError(
                "Cost budget $%.4f exceeded (current: $%.5f).\n"
                "Raise ``budget_usd`` on ``CostTracker`` or stop passing a tracker."
                % (self.budget_usd, self._cost_usd)
            )

    def report(self) -> CostReport:
        return CostReport(
            total_input_tokens=self._input_tokens,
            total_output_tokens=self._output_tokens,
            total_cost_usd=self._cost_usd,
            budget_usd=self.budget_usd,
            calls=self._calls,
        )
