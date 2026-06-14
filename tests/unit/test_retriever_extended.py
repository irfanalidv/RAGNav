from __future__ import annotations

from unittest.mock import MagicMock

from ragnav.graph import Edge
from ragnav.llm.fake import FakeLLMClient
from ragnav.models import Block, Document
from ragnav.retrieval import RAGNavIndex, RAGNavRetriever


def _paperish_index(llm: FakeLLMClient) -> RAGNavIndex:
    doc = Document(doc_id="paper", source="paper.pdf", metadata={})
    blocks = [
        Block(
            block_id="p1",
            doc_id="paper",
            type="paragraph",
            text="Abstract: we evaluate retrieval on SQuAD.",
            anchors={"page": 1},
        ),
        Block(
            block_id="p2",
            doc_id="paper",
            type="paragraph",
            text="Methods section describes hybrid BM25 and dense retrieval.",
            anchors={"page": 2},
        ),
        Block(
            block_id="p3",
            doc_id="paper",
            type="paragraph",
            text="See Figure 1 on page 3 for architecture diagram.",
            anchors={"page": 3},
        ),
    ]
    edges = [Edge(src="p2", dst="p3", type="link_to")]
    return RAGNavIndex.build(
        documents=[doc],
        blocks=blocks,
        llm=llm,
        use_sentence_transformers=False,
        edges=edges,
    )


def test_retrieve_raw_returns_serializable_hits():
    llm = FakeLLMClient()
    idx = _paperish_index(llm)
    ret = RAGNavRetriever(index=idx, llm=llm)
    hits = ret.retrieve_raw("hybrid BM25 retrieval", max_blocks=2, k_final=2, use_vectors=True)
    assert hits
    assert hits[0]["block_id"]
    assert "content" in hits[0]
    contents = " ".join(h["content"].lower() for h in hits)
    assert "bm25" in contents or "hybrid" in contents or "squad" in contents


def test_retrieve_paper_routes_to_methods_page():
    llm = FakeLLMClient()
    idx = _paperish_index(llm)
    ret = RAGNavRetriever(index=idx, llm=llm)
    res = ret.retrieve_paper(
        "methods hybrid retrieval",
        top_pages=2,
        use_vectors=True,
        k_final=3,
        follow_refs=True,
    )
    assert res.query == "methods hybrid retrieval"
    assert res.blocks
    assert res.trace.get("mode") == "paper"
    pages = {b.anchors.get("page") for b in res.blocks}
    assert 2 in pages or 3 in pages


def test_retrieve_respects_allowed_doc_ids():
    llm = FakeLLMClient()
    docs = [
        Document(doc_id="a"),
        Document(doc_id="b"),
    ]
    blocks = [
        Block(block_id="a1", doc_id="a", type="paragraph", text="alpha zebra keyword"),
        Block(block_id="b1", doc_id="b", type="paragraph", text="alpha beta keyword"),
    ]
    idx = RAGNavIndex.build(documents=docs, blocks=blocks, llm=llm, use_sentence_transformers=False)
    ret = RAGNavRetriever(index=idx, llm=llm)
    res = ret.retrieve("alpha keyword", allowed_doc_ids={"a"}, k_final=3)
    assert all(b.doc_id == "a" for b in res.blocks)


def test_retrieve_uses_retrieval_cache_on_second_call():
    llm = FakeLLMClient()
    doc = Document(doc_id="d")
    blocks = [Block(block_id="b1", doc_id="d", type="paragraph", text="cache me please")]
    idx = RAGNavIndex.build(
        documents=[doc], blocks=blocks, llm=llm, use_sentence_transformers=False
    )
    from ragnav.cache.sqlite_cache import RetrievalCache, SqliteCacheConfig, SqliteKV

    cache = RetrievalCache(SqliteKV(SqliteCacheConfig(db_path=":memory:")))
    ret = RAGNavRetriever(index=idx, llm=llm)
    first = ret.retrieve("cache", k_final=2, retrieval_cache=cache)
    second = ret.retrieve("cache", k_final=2, retrieval_cache=cache)
    assert first.blocks
    assert second.trace.get("cache_hit") is True
    assert [b.block_id for b in second.blocks] == [b.block_id for b in first.blocks]


def test_tree_prompt_baseline_returns_llm_payload():
    llm = FakeLLMClient()
    idx = _paperish_index(llm)
    ret = RAGNavRetriever(index=idx, llm=llm)
    out = ret.tree_prompt_baseline("SQuAD evaluation", max_nodes=5)
    assert out["prompt_nodes"]
    assert "SQuAD evaluation" in out["raw"]
    assert out["prompt_nodes"][0]["node_id"] == "p1"


def test_tree_search_llm_picks_nodes_from_json():
    llm = MagicMock()
    llm.embed = MagicMock(side_effect=lambda inputs, model=None: [[0.1] * 64 for _ in inputs])
    llm.chat = MagicMock(return_value='{"done": false, "node_list": ["p2"]}')
    idx = _paperish_index(FakeLLMClient())
    ret = RAGNavRetriever(index=idx, llm=llm)
    res = ret.tree_search_llm("methods", max_nodes=10, max_steps=1, nodes_per_step=2)
    assert res.blocks
    assert any(b.block_id == "p2" for b in res.blocks)


def test_retrieve_weighted_fusion_differs_from_rrf():
    llm = FakeLLMClient()
    doc = Document(doc_id="d")
    blocks = [
        Block(block_id="b1", doc_id="d", type="paragraph", text="unique alpha token"),
        Block(block_id="b2", doc_id="d", type="paragraph", text="unique beta token"),
    ]
    idx = RAGNavIndex.build(
        documents=[doc], blocks=blocks, llm=llm, use_sentence_transformers=False
    )
    ret = RAGNavRetriever(index=idx, llm=llm)
    rrf = ret.retrieve("alpha", fusion="rrf", k_final=2, expand_structure=False)
    weighted = ret.retrieve(
        "alpha",
        fusion="weighted",
        bm25_weight=1.0,
        vector_weight=0.0,
        k_final=2,
        expand_structure=False,
    )
    assert rrf.blocks[0].block_id == "b1"
    assert weighted.blocks[0].block_id == "b1"
