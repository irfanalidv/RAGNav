from __future__ import annotations

from ragnav.graph import BlockGraph, Edge
from ragnav.models import Block, Document


def test_block_graph_neighbors_follows_outgoing_and_incoming():
    g = BlockGraph()
    doc = Document(doc_id="doc1", source="x")
    a = Block(block_id="a", doc_id="doc1", type="paragraph", text="A")
    b = Block(block_id="b", doc_id="doc1", type="paragraph", text="B")
    g.add_document(doc)
    g.add_block(a)
    g.add_block(b)
    g.add_edge(Edge(src="a", dst="b", type="next"))
    g.build_indexes()
    nbrs = g.neighbors("a", types={"next"}, include_incoming=False)
    assert nbrs == ["b"]
    nbrs_in = g.neighbors("b", types={"next"}, include_outgoing=False)
    assert nbrs_in == ["a"]


def test_block_graph_out_edges_filters_by_type():
    g = BlockGraph()
    g.add_block(Block(block_id="a", doc_id="d", type="paragraph", text="a"))
    g.add_block(Block(block_id="b", doc_id="d", type="paragraph", text="b"))
    g.add_edge(Edge(src="a", dst="b", type="reply_to"))
    g.add_edge(Edge(src="a", dst="b", type="next"))
    g.build_indexes()
    reply_only = g.out_edges("a", types={"reply_to"})
    assert len(reply_only) == 1
    assert reply_only[0].type == "reply_to"
