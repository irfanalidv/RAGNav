from __future__ import annotations

import builtins
from unittest.mock import patch

import pytest

from ragnav.exceptions import RAGNavEmbeddingError
from ragnav.models import Block
from ragnav.reranking import CrossEncoderReranker


def test_cross_encoder_rerank_empty_returns_empty():
    r = CrossEncoderReranker()
    assert r.rerank("query", [], top_k=5) == []


def test_cross_encoder_rerank_raises_without_sentence_transformers():
    real_import = builtins.__import__

    def fake_import(name: str, globals_arg=None, locals_arg=None, fromlist=(), level=0):
        if name == "sentence_transformers" or (
            fromlist and any(x.startswith("sentence_transformers") for x in fromlist)
        ):
            raise ImportError("simulated missing package")
        return real_import(name, globals_arg, locals_arg, fromlist, level)

    blk = Block(
        block_id="b0",
        doc_id="d",
        type="paragraph",
        text="Paris is the capital of France.",
    )
    with patch.object(builtins, "__import__", fake_import):
        with pytest.raises(RAGNavEmbeddingError) as excinfo:
            CrossEncoderReranker().rerank("capital of France", [blk], top_k=1)
    msg = str(excinfo.value).lower()
    assert "ragnav[embeddings]" in msg or "pip install" in msg


def test_cross_encoder_rerank_orders_by_relevance():
    pytest.importorskip("sentence_transformers")
    blocks = [
        Block(block_id="b0", doc_id="d", type="paragraph", text="Berlin is a large city in Germany."),
        Block(block_id="b1", doc_id="d", type="paragraph", text="Rome is in Italy."),
        Block(
            block_id="b2",
            doc_id="d",
            type="paragraph",
            text="Paris is the capital of France and a major European city.",
        ),
        Block(block_id="b3", doc_id="d", type="paragraph", text="Madrid is the capital of Spain."),
        Block(block_id="b4", doc_id="d", type="paragraph", text="Ottawa is the capital of Canada."),
    ]
    r = CrossEncoderReranker()
    out = r.rerank("What is the capital of France?", blocks, top_k=3)
    assert out[0].block_id == "b2"
