from __future__ import annotations

from ragnav.index.bm25 import BM25Index, _tokenize
from ragnav.models import Block


def test_tokenize_lowercases_alphanumeric():
    assert _tokenize("Hello World_123") == ["hello", "world_123"]


def test_bm25_search_ranks_matching_document_highest():
    blocks = [
        Block(block_id="a", doc_id="d", type="paragraph", text="zebra moon"),
        Block(block_id="b", doc_id="d", type="paragraph", text="alpha keyword match"),
        Block(block_id="c", doc_id="d", type="paragraph", text="unrelated content here"),
    ]
    idx = BM25Index.build(blocks)
    hits = idx.search("alpha keyword", k=3)
    assert hits[0][0].block_id == "b"
    assert hits[0][1] > hits[1][1]


def test_bm25_search_empty_query_returns_results_without_error():
    blocks = [Block(block_id="a", doc_id="d", type="paragraph", text="text")]
    idx = BM25Index.build(blocks)
    hits = idx.search("", k=1)
    assert len(hits) == 1


def test_bm25_search_respects_allowed_doc_ids():
    blocks = [
        Block(block_id="a", doc_id="doc_a", type="paragraph", text="shared token"),
        Block(block_id="b", doc_id="doc_b", type="paragraph", text="shared token"),
    ]
    idx = BM25Index.build(blocks)
    hits = idx.search("shared", k=2, allowed_doc_ids={"doc_b"})
    assert len(hits) == 1
    assert hits[0][0].doc_id == "doc_b"
