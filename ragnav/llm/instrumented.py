from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .base import LLMClient


@dataclass
class LLMStats:
    chat_calls: int = 0
    embed_calls: int = 0
    chat_ms: float = 0.0
    embed_ms: float = 0.0
    embedded_texts: int = 0


@dataclass
class InstrumentedLLMClient(LLMClient):
    inner: LLMClient
    stats: LLMStats = field(default_factory=LLMStats)

    def chat(
        self, *, messages: list[dict[str, str]], model: Optional[str] = None, temperature: float = 0
    ) -> str:
        t0 = time.perf_counter()
        try:
            return self.inner.chat(messages=messages, model=model, temperature=temperature)
        finally:
            self.stats.chat_calls += 1
            self.stats.chat_ms += (time.perf_counter() - t0) * 1000.0

    def embed(self, *, inputs: Iterable[str], model: Optional[str] = None) -> list[list[float]]:
        items = list(inputs)
        t0 = time.perf_counter()
        try:
            return self.inner.embed(inputs=items, model=model)
        finally:
            self.stats.embed_calls += 1
            self.stats.embedded_texts += len(items)
            self.stats.embed_ms += (time.perf_counter() - t0) * 1000.0

