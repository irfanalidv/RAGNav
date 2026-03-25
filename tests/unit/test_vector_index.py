from __future__ import annotations

import builtins

import pytest

from ragnav.exceptions import RAGNavEmbeddingError
from ragnav.index.vectors import VectorIndex
from ragnav.models import Block


def _blocks_10() -> list[Block]:
    texts = [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
        "Rome is the capital of Italy.",
        "Madrid is the capital of Spain.",
        "London is in the United Kingdom.",
        "Ottawa is the capital of Canada.",
        "Tokyo is the capital of Japan.",
        "Canberra is the capital of Australia.",
        "Brasília is the capital of Brazil.",
        "Buenos Aires is a large city in Argentina.",
    ]
    return [
        Block(block_id=f"b{i}", doc_id="d", type="paragraph", text=t) for i, t in enumerate(texts)
    ]


def test_from_embeddings_ranks_by_cosine():
    blocks = _blocks_10()
    dim = 8
    embeddings = []
    for i, b in enumerate(blocks):
        v = [0.0] * dim
        v[i % dim] = 1.0
        embeddings.append(v)
    vi = VectorIndex.from_embeddings(blocks, embeddings)
    assert vi.has_fitted
    assert not vi.uses_sentence_transformers
    q = [1.0] + [0.0] * (dim - 1)
    hits = vi.search(q, k=3)
    assert hits[0][0].block_id == "b0"


def test_sentence_transformer_build_query_france_top3():
    pytest.importorskip("sentence_transformers")
    from ragnav.index import vectors as vv

    vv._ST_MODELS.clear()
    blocks = _blocks_10()
    texts = [b.text for b in blocks]
    ids = [b.block_id for b in blocks]
    vi = VectorIndex(model_name="all-MiniLM-L6-v2")
    assert not vi.has_fitted
    vi.build(texts, ids, blocks=blocks)
    assert vi.has_fitted
    assert vi.uses_sentence_transformers
    ranked = vi.query("What is the capital of France?", top_k=3)
    top_ids = {bid for bid, _ in ranked}
    assert "b0" in top_ids


def test_sentence_transformer_missing_shows_install_hint(monkeypatch):
    from ragnav.index import vectors as vv

    vv._ST_MODELS.clear()

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sentence_transformers":
            raise ImportError("simulated missing package")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    blocks = _blocks_10()
    vi = VectorIndex(model_name="all-MiniLM-L6-v2")
    with pytest.raises(RAGNavEmbeddingError) as excinfo:
        vi.build([b.text for b in blocks], [b.block_id for b in blocks], blocks=blocks)
    assert "pip install" in str(excinfo.value).lower() or "ragnav[embeddings]" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_search_for_query_dispatches_st_vs_external():
    blocks = [
        Block(block_id="a", doc_id="d", type="paragraph", text="alpha"),
        Block(block_id="b", doc_id="d", type="paragraph", text="beta"),
    ]
    vi = VectorIndex.from_embeddings(blocks, [[1.0, 0.0], [0.0, 1.0]])
    out = vi.search_for_query("ignored", query_embedding=[1.0, 0.0], k=2)
    assert out[0][0].block_id == "a"
