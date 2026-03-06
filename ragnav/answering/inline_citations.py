from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from ..json_utils import extract_json
from ..llm.base import LLMClient
from ..models import Block


_CITE_RE = re.compile(r"\[\[([^\]]+)\]\]")


@dataclass(frozen=True)
class CitedAnswer:
    answer: str
    cited_block_ids: tuple[str, ...]
    trace: dict[str, Any]


def build_cited_context(blocks: list[Block], *, max_chars: int = 18000) -> str:
    """
    Produce a citeable context that gives the model stable block_ids to cite.
    """
    parts: list[str] = []
    used = 0
    for b in blocks:
        title = " > ".join(b.heading_path) if b.heading_path else None
        page = b.anchors.get("page")
        header = f"BLOCK_ID={b.block_id}  PAGE={page}  TITLE={title}"
        body = (b.text or "").strip()
        chunk = f"{header}\n{body}\n"
        if used + len(chunk) > max_chars:
            break
        parts.append(chunk)
        used += len(chunk)
    return "\n---\n".join(parts).strip()


def _sentences(text: str) -> list[str]:
    # Simple heuristic sentence split for validation.
    raw = (text or "").strip()
    if not raw:
        return []
    parts = re.split(r"(?<=[.!?])\s+", raw)
    return [p.strip() for p in parts if p.strip()]


def validate_inline_citations(answer: str, *, allowed_block_ids: set[str]) -> dict[str, Any]:
    """
    Validate that:
    - every sentence contains at least one [[block_id]] citation
    - all cited block_ids are from the allowed set
    """
    sents = _sentences(answer)
    cited = _CITE_RE.findall(answer or "")
    cited_clean = [c.strip() for c in cited if c.strip()]
    cited_set = set(cited_clean)

    unknown = sorted([c for c in cited_set if c not in allowed_block_ids])
    sentences_missing = []
    for i, s in enumerate(sents):
        if not _CITE_RE.search(s):
            sentences_missing.append(i)

    ok = (not unknown) and (len(sentences_missing) == 0) and (len(cited_set) > 0)
    return {
        "ok": ok,
        "n_sentences": len(sents),
        "n_citations": len(cited_clean),
        "unique_cited_block_ids": sorted(list(cited_set)),
        "unknown_block_ids": unknown,
        "sentences_missing_citations": sentences_missing,
    }


def answer_with_inline_citations(
    *,
    llm: LLMClient,
    query: str,
    blocks: list[Block],
    temperature: float = 0,
    max_context_chars: int = 18000,
) -> CitedAnswer:
    """
    Produce a grounded answer with inline citations like: `... [[block_id]]`

    Contract:
    - Every sentence MUST include at least one citation.
    - Citations MUST only reference provided block_ids.
    """
    allowed = {b.block_id for b in blocks}
    context = build_cited_context(blocks, max_chars=max_context_chars)
    prompt = f"""
You are answering a question using ONLY the provided context blocks.

Rules (STRICT):
- Every sentence in your answer MUST end with one or more citations in the form [[BLOCK_ID]].
- You MUST only cite BLOCK_ID values that appear in the provided context.
- If the context is insufficient, say so, but still cite the closest relevant blocks.

Question: {query}

Context blocks:
{context}
""".strip()

    raw = llm.chat(messages=[{"role": "user", "content": prompt}], temperature=temperature)

    # Optional: allow JSON form too (handy for some providers).
    ans = raw
    try:
        parsed = extract_json(raw)
        if isinstance(parsed, dict) and isinstance(parsed.get("answer"), str):
            ans = parsed["answer"]
    except Exception:
        pass

    v = validate_inline_citations(ans, allowed_block_ids=allowed)
    if not v["ok"]:
        raise ValueError(f"Answer citation validation failed: {v}")

    cited_ids = tuple(v["unique_cited_block_ids"])
    return CitedAnswer(answer=ans.strip(), cited_block_ids=cited_ids, trace={"validation": v})

