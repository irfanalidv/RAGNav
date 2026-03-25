from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from ..cost import CostTracker, estimate_tokens_from_text
from ..exceptions import RAGNavLLMError
from .base import LLMClient

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

    def __init__(self, cfg: Optional[MistralConfig] = None, cost_tracker: Optional[CostTracker] = None):
        try:
            from dotenv import load_dotenv
        except Exception:
            load_dotenv = None

        if load_dotenv:
            load_dotenv()

        try:
            from mistralai import Mistral
        except Exception as e:
            raise RAGNavLLMError(
                "Missing optional dependency `mistralai`. Install with: pip install -e \".[mistral]\""
            ) from e

        self.cfg = cfg or MistralConfig()
        api_key = os.getenv(self.cfg.api_key_env)
        if not api_key:
            raise RAGNavLLMError(
                f"Missing {self.cfg.api_key_env}. Set it in your environment or a `.env` file; do not hardcode keys."
            )
        self._client = Mistral(api_key=api_key)
        self.cost_tracker = cost_tracker

    def chat(
        self, *, messages: list[dict[str, str]], model: Optional[str] = None, temperature: float = 0
    ) -> str:
        model_name = model or self.cfg.chat_model
        if self.cost_tracker is not None:
            self.cost_tracker.check_budget()
        prompt_text = "\n".join(str(m.get("content") or "") for m in messages)
        resp = self._client.chat.complete(
            model=model_name,
            messages=messages,
            temperature=temperature,
        )
        text = (resp.choices[0].message.content or "").strip()
        if self.cost_tracker is not None:
            self.cost_tracker.record(
                model_name,
                estimate_tokens_from_text(prompt_text),
                estimate_tokens_from_text(text),
            )
        return text

    def embed(self, *, inputs: Iterable[str], model: Optional[str] = None) -> list[list[float]]:
        resp = self._client.embeddings.create(
            model=model or self.cfg.embed_model,
            inputs=list(inputs),
        )
        return [d.embedding for d in resp.data]

    def usage_from_response(self, response_obj: Any) -> dict[str, Any]:
        # Placeholder for future: standardized token accounting.
        return {}

