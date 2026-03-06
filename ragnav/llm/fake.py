from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class FakeLLMConfig:
    dim: int = 64


class FakeLLMClient:
    """
    Offline client for tests/examples (no network, no keys).
    Produces deterministic embeddings from text hashes and a simple chat response.
    """

    def __init__(self, cfg: Optional[FakeLLMConfig] = None):
        self.cfg = cfg or FakeLLMConfig()

    def chat(
        self, *, messages: list[dict[str, str]], model: Optional[str] = None, temperature: float = 0
    ) -> str:
        user = ""
        for m in messages:
            if m.get("role") == "user":
                user = m.get("content", "")
        # Extremely simple "model": echo a stub with a stable fingerprint
        fp = hashlib.sha256(user.encode("utf-8")).hexdigest()[:12]
        return f"[fake-llm:{fp}] {user[:200]}".strip()

    def embed(self, *, inputs: Iterable[str], model: Optional[str] = None) -> list[list[float]]:
        out: list[list[float]] = []
        for text in inputs:
            h = hashlib.sha256((text or "").encode("utf-8")).digest()
            # Expand hash bytes into floats in [-1, 1]
            vec = []
            for i in range(self.cfg.dim):
                b = h[i % len(h)]
                vec.append((b / 127.5) - 1.0)
            out.append(vec)
        return out

