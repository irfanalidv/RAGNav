from __future__ import annotations

from unittest.mock import MagicMock

from ragnav.llm.fake import FakeLLMClient
from ragnav.models import Block, Document
from ragnav.retrieval import RAGNavIndex, RAGNavRetriever
from ragnav.retrieval.tree_search import hybrid_tree_search_llm, tree_search_llm


def _indexed_retriever(llm=None):
    llm = llm or FakeLLMClient()
    doc = Document(doc_id="d", source="doc.md")
    blocks = [
        Block(block_id="h1", doc_id="d", type="heading", text="Overview", heading_path=("Overview",)),
        Block(
            block_id="p1",
            doc_id="d",
            type="paragraph",
            text="Paris is the capital of France.",
            parent_id="h1",
            heading_path=("Overview",),
        ),
        Block(
            block_id="p2",
            doc_id="d",
            type="paragraph",
            text="Berlin is the capital of Germany.",
            parent_id="h1",
            heading_path=("Overview",),
        ),
    ]
    idx = RAGNavIndex.build(documents=[doc], blocks=blocks, llm=llm, use_sentence_transformers=False)
    return RAGNavRetriever(index=idx, llm=llm)


def test_tree_search_llm_falls_back_to_hybrid_when_llm_returns_empty():
    llm = MagicMock()
    llm.embed = MagicMock(side_effect=lambda inputs, model=None: [[0.1] * 64 for _ in inputs])
    llm.chat = MagicMock(return_value='{"done": false, "node_list": []}')
    ret = _indexed_retriever(llm)
    res = tree_search_llm(ret, "Paris capital", max_steps=1, nodes_per_step=2)
    assert res.blocks
    assert res.trace.get("fallback") is True


def test_hybrid_tree_search_llm_picks_candidate_nodes():
    llm = MagicMock()
    llm.embed = MagicMock(side_effect=lambda inputs, model=None: [[0.1] * 64 for _ in inputs])
    llm.chat = MagicMock(return_value='{"node_list": ["p1"]}')
    ret = _indexed_retriever(llm)
    res = hybrid_tree_search_llm(ret, "Paris France capital", pick_k=2, k_candidates=5)
    assert any(b.block_id == "p1" for b in res.blocks)
    assert res.trace.get("mode") == "hybrid_tree_search_llm"


def test_hybrid_tree_search_llm_falls_back_when_json_invalid():
    llm = MagicMock()
    llm.embed = MagicMock(side_effect=lambda inputs, model=None: [[0.1] * 64 for _ in inputs])
    llm.chat = MagicMock(return_value="not json at all")
    ret = _indexed_retriever(llm)
    res = hybrid_tree_search_llm(ret, "Paris", pick_k=2)
    assert res.blocks
    assert res.trace.get("fallback") is True
