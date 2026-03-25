from __future__ import annotations

from typing import Iterable, Optional, Protocol


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
        model: Optional[str] = None,
        temperature: float = 0,
    ) -> str: ...

    def embed(
        self,
        *,
        inputs: Iterable[str],
        model: Optional[str] = None,
    ) -> list[list[float]]: ...

