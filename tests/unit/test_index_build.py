from __future__ import annotations

from unittest.mock import MagicMock

from unittest.mock import patch

import pytest

from ragnav.exceptions import RAGNavEmbeddingError, RAGNavLLMError
from ragnav.llm.fake import FakeLLMClient
from ragnav.models import Block, Document
from ragnav.observability import Trace
from ragnav.retrieval import RAGNavIndex


def test_index_build_without_vectors_skips_embedding():
    doc = Document(doc_id="d")
    blocks = [Block(block_id="b1", doc_id="d", type="paragraph", text="offline only")]
    idx = RAGNavIndex.build(
        documents=[doc], blocks=blocks, llm=FakeLLMClient(), build_vectors=False
    )
    assert idx.has_vectors is False
    assert idx.bm25.search("offline", k=1)[0][0].block_id == "b1"


def test_index_build_uses_llm_embeddings_when_sentence_transformers_disabled():
    llm = MagicMock()
    llm.embed = MagicMock(side_effect=lambda inputs, model=None: [[0.5] * 8 for _ in inputs])
    doc = Document(doc_id="d")
    blocks = [
        Block(block_id="b1", doc_id="d", type="paragraph", text="alpha"),
        Block(block_id="b2", doc_id="d", type="paragraph", text="beta"),
    ]
    idx = RAGNavIndex.build(
        documents=[doc],
        blocks=blocks,
        llm=llm,
        use_sentence_transformers=False,
        embed_batch_size=1,
    )
    assert idx.has_vectors is True
    assert llm.embed.call_count >= 2


def test_index_build_records_trace_for_embedding_cache(tmp_path):
    from ragnav.cache.sqlite_cache import EmbeddingCache, SqliteCacheConfig, SqliteKV

    llm = FakeLLMClient()
    doc = Document(doc_id="d")
    blocks = [Block(block_id="b1", doc_id="d", type="paragraph", text="trace me")]
    cache = EmbeddingCache(SqliteKV(SqliteCacheConfig(db_path=str(tmp_path / "emb.db"))))
    trace = Trace()
    idx = RAGNavIndex.build(
        documents=[doc],
        blocks=blocks,
        llm=llm,
        use_sentence_transformers=False,
        embedding_cache=cache,
        trace=trace,
    )
    assert idx.has_vectors is True
    assert trace.counters.get("embed_texts", 0) >= 1


def test_index_build_raises_without_llm_when_vectors_required():
    with patch("ragnav.llm.mistral.MistralClient", side_effect=ImportError("mistral unavailable")):
        with pytest.raises(RAGNavLLMError):
            RAGNavIndex.build(
                documents=[Document(doc_id="d")],
                blocks=[Block(block_id="b", doc_id="d", type="paragraph", text="x")],
                llm=None,
                build_vectors=True,
                use_sentence_transformers=False,
            )


def test_index_build_raises_when_llm_returns_incomplete_embeddings():
    llm = MagicMock()
    llm.embed = MagicMock(return_value=[])
    doc = Document(doc_id="d")
    blocks = [Block(block_id="b1", doc_id="d", type="paragraph", text="missing embed")]
    with pytest.raises(RAGNavEmbeddingError):
        RAGNavIndex.build(
            documents=[doc],
            blocks=blocks,
            llm=llm,
            use_sentence_transformers=False,
        )
