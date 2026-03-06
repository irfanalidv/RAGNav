from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional

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

    def __init__(self, cfg: Optional[MistralConfig] = None):
        try:
            from dotenv import load_dotenv
        except Exception:
            load_dotenv = None

        if load_dotenv:
            load_dotenv()

        try:
            from mistralai import Mistral
        except Exception as e:
            raise RuntimeError(
                "Missing optional dependency `mistralai`. Install with: pip install -e \".[mistral]\""
            ) from e

        self.cfg = cfg or MistralConfig()
        api_key = os.getenv(self.cfg.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing {self.cfg.api_key_env}. Set it in your environment (do not hardcode keys)."
            )
        self._client = Mistral(api_key=api_key)

    def chat(
        self, *, messages: list[dict[str, str]], model: Optional[str] = None, temperature: float = 0
    ) -> str:
        resp = self._client.chat.complete(
            model=model or self.cfg.chat_model,
            messages=messages,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()

    def embed(self, *, inputs: Iterable[str], model: Optional[str] = None) -> list[list[float]]:
        resp = self._client.embeddings.create(
            model=model or self.cfg.embed_model,
            inputs=list(inputs),
        )
        return [d.embedding for d in resp.data]

    def usage_from_response(self, response_obj: Any) -> dict[str, Any]:
        # Placeholder for future: standardized token accounting.
        return {}

