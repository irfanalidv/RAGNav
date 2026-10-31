from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol


class LLMClient(Protocol):
    """
    Provider-agnostic interface used by RAGNav.

    Concrete clients (e.g. ``MistralClient``, ``FakeLLMClient``) may expose an optional
    ``cost_tracker`` attribute; ``RAGNavRetriever`` assigns it when you pass ``cost_tracker=...``.
    """

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0,
    ) -> str: ...

    def embed(
        self,
        *,
        inputs: Iterable[str],
        model: str | None = None,
    ) -> list[list[float]]: ...
