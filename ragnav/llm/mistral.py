from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

from ..cost import CostTracker, estimate_tokens_from_text
from ..exceptions import RAGNavLLMError
from .base import LLMClient


def _message_text(message: Any) -> str:
    """Text of a chat message whose ``content`` is a string, a list of chunks, or None.

    Reasoning models in mistralai 2.x can return a list of content chunks (text plus
    "thinking" chunks) instead of a plain string; keep only the text chunks.
    """
    content = getattr(message, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for chunk in content:
        if getattr(chunk, "type", "text") == "text":
            text = getattr(chunk, "text", None)
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


@dataclass(frozen=True)
class MistralConfig:
    api_key_env: str = "MISTRAL_API_KEY"
    chat_model: str = "mistral-large-latest"
    embed_model: str = "mistral-embed"


class MistralClient(LLMClient):
    """
    Thin wrapper so the rest of RAGNav stays provider-agnostic.

    Note: `mistralai` is an optional dependency. Install with:
      pip install -e ".[mistral]"
    """

    def __init__(self, cfg: MistralConfig | None = None, cost_tracker: CostTracker | None = None):
        try:
            from dotenv import load_dotenv
        except ImportError:
            pass  # python-dotenv is optional; plain environment variables still work
        else:
            load_dotenv()

        try:
            # mistralai >= 2 moved the client under `mistralai.client`; 1.x exports it at the top.
            try:
                from mistralai.client import Mistral
            except ImportError:
                from mistralai import Mistral  # type: ignore[no-redef,attr-defined,unused-ignore]
        except Exception as e:
            raise RAGNavLLMError(
                'Missing optional dependency `mistralai`. Install with: pip install -e ".[mistral]"'
            ) from e

        self.cfg = cfg or MistralConfig()
        api_key = os.getenv(self.cfg.api_key_env)
        if not api_key:
            raise RAGNavLLMError(
                f"Missing {self.cfg.api_key_env}. Set it in your environment or a `.env` file; do "
                "not hardcode keys."
            )
        self._client = Mistral(api_key=api_key)
        self.cost_tracker = cost_tracker

    def chat(
        self, *, messages: list[dict[str, str]], model: str | None = None, temperature: float = 0
    ) -> str:
        model_name = model or self.cfg.chat_model
        if self.cost_tracker is not None:
            self.cost_tracker.check_budget()
        prompt_text = "\n".join(str(m.get("content") or "") for m in messages)
        resp = self._client.chat.complete(
            model=model_name,
            messages=cast(Any, messages),  # plain role/content dicts are accepted at runtime
            temperature=temperature,
        )
        text = _message_text(resp.choices[0].message).strip()
        if self.cost_tracker is not None:
            self.cost_tracker.record(
                model_name,
                estimate_tokens_from_text(prompt_text),
                estimate_tokens_from_text(text),
            )
        return text

    def embed(self, *, inputs: Iterable[str], model: str | None = None) -> list[list[float]]:
        resp = self._client.embeddings.create(
            model=model or self.cfg.embed_model,
            inputs=list(inputs),
        )
        vectors = [d.embedding for d in resp.data]
        if any(v is None for v in vectors):
            raise RAGNavLLMError("Mistral embeddings response contained an item without a vector.")
        return [v for v in vectors if v is not None]

    def usage_from_response(self, response_obj: Any) -> dict[str, Any]:
        # Placeholder for future: standardized token accounting.
        return {}
