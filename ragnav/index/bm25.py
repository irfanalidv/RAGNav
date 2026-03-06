from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from rank_bm25 import BM25Okapi

from ..models import Block


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


@dataclass
class BM25Index:
    blocks: list[Block]
    block_doc_ids: list[str]
    _bm25: BM25Okapi
    _tokens: list[list[str]]

    @classmethod
    def build(cls, blocks: list[Block]) -> "BM25Index":
        tokens = [_tokenize(b.text) for b in blocks]
        doc_ids = [b.doc_id for b in blocks]
        return cls(blocks=blocks, block_doc_ids=doc_ids, _bm25=BM25Okapi(tokens), _tokens=tokens)

    def search(
        self, query: str, *, k: int = 20, allowed_doc_ids: Optional[set[str]] = None
    ) -> list[tuple[Block, float]]:
        q = _tokenize(query)
        scores = self._bm25.get_scores(q)
        if allowed_doc_ids is not None:
            idxs = [
                i
                for i in range(len(self.blocks))
                if self.block_doc_ids[i] in allowed_doc_ids
            ]
        else:
            idxs = list(range(len(self.blocks)))
        idxs = sorted(idxs, key=lambda i: float(scores[i]), reverse=True)[:k]
        return [(self.blocks[i], float(scores[i])) for i in idxs]

