from __future__ import annotations

import pytest

from ragnav.exceptions import RAGNavEmbeddingError
from ragnav.index.vectors import VectorIndex
from ragnav.models import Block


def test_vector_index_from_embeddings_search_ranks_by_cosine():
    blocks = [
        Block(block_id="a", doc_id="d", type="paragraph", text="alpha"),
        Block(block_id="b", doc_id="d", type="paragraph", text="beta"),
    ]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    idx = VectorIndex.from_embeddings(blocks, embeddings)
    hits = idx.search_for_query("ignored", query_embedding=[1.0, 0.0, 0.0], k=2)
    assert hits[0][0].block_id == "a"
    assert hits[0][1] >= hits[1][1]


def test_vector_index_from_embeddings_length_mismatch_raises():
    blocks = [Block(block_id="a", doc_id="d", type="paragraph", text="x")]
    with pytest.raises(RAGNavEmbeddingError):
        VectorIndex.from_embeddings(blocks, [])


def test_vector_index_query_raises_on_external_backend():
    blocks = [Block(block_id="a", doc_id="d", type="paragraph", text="x")]
    idx = VectorIndex.from_embeddings(blocks, [[1.0, 0.0]])
    with pytest.raises(RAGNavEmbeddingError):
        idx.query("x")
