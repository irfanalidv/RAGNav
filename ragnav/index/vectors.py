from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..models import Block


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(mat, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return mat / denom


@dataclass
class VectorIndex:
    blocks: list[Block]
    block_doc_ids: list[str]
    vectors: np.ndarray  # shape: (n, d), normalized

    @classmethod
    def build(cls, blocks: list[Block], embeddings: list[list[float]]) -> "VectorIndex":
        if len(blocks) != len(embeddings):
            raise ValueError("blocks and embeddings length mismatch")
        mat = np.asarray(embeddings, dtype=np.float32)
        if mat.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        doc_ids = [b.doc_id for b in blocks]
        return cls(blocks=blocks, block_doc_ids=doc_ids, vectors=_l2_normalize(mat))

    def search(
        self,
        query_vec: list[float],
        *,
        k: int = 20,
        allowed_doc_ids: Optional[set[str]] = None,
    ) -> list[tuple[Block, float]]:
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        q = _l2_normalize(q)
        sims = (self.vectors @ q.T).reshape(-1)  # cosine since normalized
        if allowed_doc_ids is not None:
            mask = np.fromiter((d in allowed_doc_ids for d in self.block_doc_ids), dtype=bool)
            filtered = sims.copy()
            filtered[~mask] = -1e9
            idxs = np.argsort(-filtered)[:k]
            return [(self.blocks[int(i)], float(filtered[int(i)])) for i in idxs]
        idxs = np.argsort(-sims)[:k]
        return [(self.blocks[int(i)], float(sims[int(i)])) for i in idxs]

