from __future__ import annotations

from unittest.mock import MagicMock

from ragnav.llm.fake import FakeLLMClient
from ragnav.models import Block, Document
from ragnav.models import ConfidenceLevel
from ragnav.retrieval import RAGNavIndex, RAGNavRetriever
from ragnav.retrieval._helpers import _rrf_fuse, _score_confidence


def test_ragnav_index_build_and_retrieve():
    doc = Document(doc_id="d1", source="s", metadata={})
    blocks = [
        Block(block_id="b1", doc_id="d1", type="paragraph", text="Paris is the capital of France."),
        Block(block_id="b2", doc_id="d1", type="paragraph", text="Berlin is in Germany."),
    ]
    llm = FakeLLMClient()
    idx = RAGNavIndex.build(
        documents=[doc],
        blocks=blocks,
        llm=llm,
        embed_batch_size=32,
        use_sentence_transformers=False,
    )
    ret = RAGNavRetriever(index=idx, llm=llm)
    res = ret.retrieve("capital France", k_final=4, k_bm25=10, k_vec=10)
    assert res.query == "capital France"
    assert isinstance(res.blocks, list)
    assert any("Paris" in b.text for b in res.blocks)


def test_ragnav_retriever_accepts_explicit_llm_mock():
    doc = Document(doc_id="d1")
    blocks = [Block(block_id="b1", doc_id="d1", type="paragraph", text="hello world")]
    mock_llm = MagicMock()
    mock_llm.embed = MagicMock(side_effect=lambda inputs, model=None: [[0.1] * 64 for _ in inputs])
    idx = RAGNavIndex.build(
        documents=[doc],
        blocks=blocks,
        llm=mock_llm,
        embed_batch_size=32,
        use_sentence_transformers=False,
    )
    ret = RAGNavRetriever(index=idx, llm=mock_llm)
    res = ret.retrieve("hello", k_final=2)
    mock_llm.embed.assert_called()
    assert res.blocks


def test_ragnav_index_ten_docs_retrieve_non_empty():
    llm = FakeLLMClient()
    docs = [Document(doc_id=f"d{i}") for i in range(10)]
    blocks = [
        Block(
            block_id=f"b{i}",
            doc_id=f"d{i}",
            type="paragraph",
            text=f"Document {i} discusses topic number {i} and related ideas.",
        )
        for i in range(10)
    ]
    idx = RAGNavIndex.build(
        documents=docs,
        blocks=blocks,
        llm=llm,
        embed_batch_size=32,
        use_sentence_transformers=False,
    )
    assert idx.has_vectors
    ret = RAGNavRetriever(index=idx, llm=llm)
    res = ret.retrieve("topic number 5", k_final=6, k_bm25=20, k_vec=20)
    assert res.blocks


def test_retrieve_bm25_only_non_empty():
    llm = FakeLLMClient()
    docs = [Document(doc_id="d%d" % i) for i in range(5)]
    blocks = [
        Block(
            block_id="b%d" % i,
            doc_id="d%d" % i,
            type="paragraph",
            text="token%d alpha bravo charlie" % i,
        )
        for i in range(5)
    ]
    idx = RAGNavIndex.build(
        documents=docs,
        blocks=blocks,
        llm=llm,
        embed_batch_size=16,
        use_sentence_transformers=False,
    )
    ret = RAGNavRetriever(index=idx, llm=llm)
    res = ret.retrieve(
        "token3 alpha",
        top_k=5,
        bm25_weight=1.0,
        vector_weight=0.0,
        expand_structure=False,
        expand_graph=False,
    )
    assert res.blocks


def test_score_confidence_high_medium_low():
    assert _score_confidence(0.9, 0.5) == ConfidenceLevel.HIGH
    assert _score_confidence(0.7, 0.65) == ConfidenceLevel.MEDIUM
    assert _score_confidence(0.4, 0.2) == ConfidenceLevel.LOW


def test_retrieve_empty_when_allowed_doc_ids_exclude_all():
    llm = FakeLLMClient()
    doc = Document(doc_id="d1")
    blocks = [Block(block_id="b1", doc_id="d1", type="paragraph", text="alpha beta gamma delta epsilon.")]
    idx = RAGNavIndex.build(
        documents=[doc],
        blocks=blocks,
        llm=llm,
        embed_batch_size=8,
        use_sentence_transformers=False,
    )
    ret = RAGNavRetriever(index=idx, llm=llm)
    res = ret.retrieve("alpha", allowed_doc_ids={"other-doc"}, k_final=4, expand_structure=False)
    assert res.blocks == []
    assert res.confidence == ConfidenceLevel.LOW
    assert res.top_score == 0.0


def test_retrieve_squad_like_passage_high_confidence():
    llm = FakeLLMClient()
    ctx = (
        "Super Bowl 50 was an American football game to determine the champion of the National "
        "Football League for the 2015 season. The American Football Conference champion Denver "
        "Broncos defeated the National Football Conference champion Carolina Panthers."
    )
    did = "squadtest"
    doc = Document(doc_id=did, source="s", metadata={})
    noise = Block(
        block_id="%s:p0" % did,
        doc_id=did,
        type="paragraph",
        text="Unrelated: medieval pottery glazing techniques in southern France circa 1400.",
    )
    gold = Block(block_id="%s:p1" % did, doc_id=did, type="paragraph", text=ctx)
    idx = RAGNavIndex.build(
        documents=[doc],
        blocks=[noise, gold],
        llm=llm,
        use_sentence_transformers=True,
        vector_model="all-MiniLM-L6-v2",
        embed_batch_size=8,
    )
    ret = RAGNavRetriever(index=idx, llm=llm)
    # Lexical-only fusion spreads scores enough for normalized top-two confidence.
    res = ret.retrieve(
        "Denver Broncos Super Bowl 50 AFC champion",
        top_k=10,
        bm25_weight=1.0,
        vector_weight=0.0,
        expand_structure=False,
        expand_graph=False,
        fusion="weighted",
    )
    assert res.confidence in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM)
    assert res.top_score >= 0.60
    assert any("Denver" in b.text for b in res.blocks)


def test_rrf_fuse_prefers_items_ranked_in_multiple_lists():
    list_a = [("x", 0.0), ("y", 0.0)]
    list_b = [("y", 0.0), ("z", 0.0)]
    fused = _rrf_fuse([list_a, list_b], k=60)
    assert fused[0][0] == "y"


def test_ragnav_index_no_vectors_when_build_vectors_false():
    llm = FakeLLMClient()
    doc = Document(doc_id="d0")
    blocks = [Block(block_id="b0", doc_id="d0", type="paragraph", text="only bm25")]
    idx = RAGNavIndex.build(
        documents=[doc],
        blocks=blocks,
        llm=llm,
        build_vectors=False,
    )
    assert not idx.has_vectors
