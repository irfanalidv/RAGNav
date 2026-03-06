from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from ..models import Block


@dataclass(frozen=True)
class PolicyResult:
    kept: list[Block]
    dropped_block_ids: list[str]
    sanitized_block_ids: list[str]


class ContentPolicy(Protocol):
    """
    Retrieval-time content policy.

    Goals:
    - Drop blocks that look like prompt injection or tool-abuse attempts.
    - Optionally sanitize retrieved text before feeding it to an LLM.
    """

    def apply(self, *, query: str, blocks: list[Block]) -> PolicyResult: ...


_INJECTION_LINE_RE = re.compile(
    r"\b(ignore|disregard)\b.*\b(instruction|instructions|system|developer)\b"
    r"|\byou are chatgpt\b"
    r"|\b(system prompt|developer message)\b"
    r"|\bexfiltrat(e|ion)\b"
    r"|\bdo not follow\b.*\babove\b",
    flags=re.IGNORECASE,
)

_DANGEROUS_TOOLING_RE = re.compile(
    r"\b(curl|wget|powershell|bash|sh|python)\b.*\bhttps?://",
    flags=re.IGNORECASE,
)

_SECRET_RE = re.compile(
    r"\b(api[_-]?key|password|secret|token)\b\s*[:=]\s*([A-Za-z0-9_\-]{8,})",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class SimpleInjectionPolicy:
    """
    A conservative, dependency-free baseline policy.

    - Drops blocks that contain high-confidence injection patterns.
    - Redacts obvious secret-like strings (best-effort).
    """

    redact_secrets: bool = True
    max_lines: int = 80

    def apply(self, *, query: str, blocks: list[Block]) -> PolicyResult:
        kept: list[Block] = []
        dropped: list[str] = []
        sanitized: list[str] = []

        for b in blocks:
            text = b.text or ""
            if _INJECTION_LINE_RE.search(text) or _DANGEROUS_TOOLING_RE.search(text):
                dropped.append(b.block_id)
                continue

            new_text = text
            # Truncate extremely long blocks to limit "hidden instructions" surface.
            lines = new_text.splitlines()
            if len(lines) > self.max_lines:
                new_text = "\n".join(lines[: self.max_lines]).strip()

            if self.redact_secrets:
                new_text2 = _SECRET_RE.sub(r"\1: [REDACTED]", new_text)
                new_text = new_text2

            if new_text != text:
                sanitized.append(b.block_id)
                b = Block(
                    block_id=b.block_id,
                    doc_id=b.doc_id,
                    type=b.type,
                    text=new_text,
                    parent_id=b.parent_id,
                    heading_path=b.heading_path,
                    anchors=b.anchors,
                    metadata=b.metadata,
                )

            kept.append(b)

        return PolicyResult(kept=kept, dropped_block_ids=dropped, sanitized_block_ids=sanitized)

