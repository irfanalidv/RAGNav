from __future__ import annotations

"""
Offline regression tests (no network, no API keys).

Run:
  python3 -m benchmarks.regression
"""

from ragnav.cache.sqlite_cache import EmbeddingCache, RetrievalCache, SqliteCacheConfig, SqliteKV
from ragnav.graph import Edge
from ragnav.ingest.markdown import ingest_markdown_string
from ragnav.llm.fake import FakeLLMClient
from ragnav.observability import Trace
from ragnav.retrieval import RAGNavIndex, RAGNavRetriever
from ragnav.models import Block, Document


def test_acl_filtering() -> None:
    doc, blocks = ingest_markdown_string(
        "# Root\n\n## Secret\nTop secret content.\n",
        name="sec.md",
        metadata={"acl": ["user:alice"]},
    )
    fake = FakeLLMClient()
    idx = RAGNavIndex.build(documents=[doc], blocks=blocks, llm=fake, build_vectors=True)
    r = RAGNavRetriever(index=idx, llm=fake)

    # alice can see
    res_ok = r.retrieve("secret", principal="user:alice")
    assert len(res_ok.blocks) > 0

    # bob cannot see (should return empty or non-secret)
    res_no = r.retrieve("secret", principal="user:bob")
    assert all("Top secret" not in b.text for b in res_no.blocks)


def test_graph_expansion() -> None:
    doc, blocks = ingest_markdown_string("# A\n\n## One\nHello\n\n## Two\nWorld\n", name="a.md")
    # Create a next edge between first two blocks
    if len(blocks) < 2:
        raise AssertionError("expected >= 2 blocks")
    edges = [Edge(src=blocks[0].block_id, dst=blocks[1].block_id, type="next")]

    fake = FakeLLMClient()
    idx = RAGNavIndex.build(documents=[doc], blocks=blocks, llm=fake, build_vectors=False, edges=edges)
    r = RAGNavRetriever(index=idx, llm=fake)
    res = r.retrieve("Hello", use_vectors=False, expand_graph=True)
    ids = {b.block_id for b in res.blocks}
    assert blocks[1].block_id in ids, "graph expansion should pull neighbor"


def test_retrieval_cache() -> None:
    doc, blocks = ingest_markdown_string("# A\n\n## One\nNeedle\n", name="a.md")
    fake = FakeLLMClient()
    idx = RAGNavIndex.build(documents=[doc], blocks=blocks, llm=fake, build_vectors=False)
    r = RAGNavRetriever(index=idx, llm=fake)

    kv = SqliteKV(SqliteCacheConfig(db_path="cache/regression.sqlite3"))
    rcache = RetrievalCache(kv)

    q = "Needle"
    res1 = r.retrieve(q, use_vectors=False, retrieval_cache=rcache)
    res2 = r.retrieve(q, use_vectors=False, retrieval_cache=rcache)
    assert res1.blocks and res2.blocks
    assert res2.trace.get("cache_hit") is True


def test_embedding_cache() -> None:
    doc, blocks = ingest_markdown_string("# A\n\n## One\nNeedle\n", name="a.md")
    fake = FakeLLMClient()
    kv = SqliteKV(SqliteCacheConfig(db_path="cache/regression.sqlite3"))
    ecache = EmbeddingCache(kv)
    tr = Trace()

    _ = RAGNavIndex.build(documents=[doc], blocks=blocks, llm=fake, build_vectors=True, embedding_cache=ecache, trace=tr)
    # second build should hit cache for all embedded texts
    tr2 = Trace()
    _ = RAGNavIndex.build(documents=[doc], blocks=blocks, llm=fake, build_vectors=True, embedding_cache=ecache, trace=tr2)
    assert tr2.counters.get("embed_cache_hits", 0) > 0


def test_paper_page_routing_and_crossref_follow() -> None:
    """
    Ensure page-first routing can pull evidence from linked pages via `link_to`.
    """
    doc = Document(doc_id="pdf:paper.pdf", source="paper.pdf", metadata={"type": "pdf", "mode": "paper"})
    # Page 1 mentions Figure 1; Page 2 contains the Figure 1 caption.
    b0 = Block(
        block_id=f"{doc.doc_id}#b0",
        doc_id=doc.doc_id,
        type="paragraph",
        text="We summarize results in Figure 1. The main contribution is routing pages.",
        anchors={"page": 1, "line_start": 1, "line_end": 1},
    )
    b1 = Block(
        block_id=f"{doc.doc_id}#b1",
        doc_id=doc.doc_id,
        type="paragraph",
        text="Figure 1: Accuracy improves when following cross-references.",
        anchors={"page": 2, "line_start": 1, "line_end": 1},
        metadata={"caption_kind": "figure", "caption_number": "1"},
    )
    edges = [Edge(src=b0.block_id, dst=b1.block_id, type="link_to", metadata={"ref_kind": "figure", "ref": "1"})]

    fake = FakeLLMClient()
    idx = RAGNavIndex.build(documents=[doc], blocks=[b0, b1], llm=fake, build_vectors=False, edges=edges)
    r = RAGNavRetriever(index=idx, llm=fake)

    res = r.retrieve_paper("What does Figure 1 show?", top_pages=1, follow_refs=True, include_next=False, use_vectors=False)
    ids = {b.block_id for b in res.blocks}
    assert b1.block_id in ids, "expected linked caption block to be pulled via link_to expansion"


def main() -> None:
    test_acl_filtering()
    test_graph_expansion()
    test_retrieval_cache()
    test_embedding_cache()
    test_paper_page_routing_and_crossref_follow()
    print("regression_ok")


if __name__ == "__main__":
    main()

