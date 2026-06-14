from __future__ import annotations

from ragnav.graph import Edge
from ragnav.models import Block, ConfidenceLevel, Document, RetrievalResult
from ragnav.retrieval._helpers import (
    _allowed_by_constraints,
    _confidence_normalize_top_two,
    _dedupe_keep_best,
    _edges_in,
    _edges_out,
    _expand_graph,
    _expand_structure,
    _fuse_retrieval_rankings,
    _minmax_norm,
    _rrf_fuse,
    _safe_title,
    _score_confidence,
    _with_trace,
)


def _blk(bid: str, text: str, *, parent: str | None = None, path: tuple[str, ...] = ()) -> Block:
    return Block(
        block_id=bid,
        doc_id="d",
        type="paragraph",
        text=text,
        parent_id=parent,
        heading_path=path,
    )


def test_rrf_fuse_single_list_preserves_order():
    fused = _rrf_fuse([[("a", 1.0), ("b", 0.5), ("c", 0.1)]])
    assert [x[0] for x in fused] == ["a", "b", "c"]


def test_rrf_fuse_item_high_in_both_lists_outranks_single_list_leader():
    list_a = [("shared", 10.0), ("only_a", 5.0)]
    list_b = [("shared", 8.0), ("only_b", 4.0)]
    fused = _rrf_fuse([list_a, list_b])
    assert fused[0][0] == "shared"
    shared_score = next(s for bid, s in fused if bid == "shared")
    only_a_score = next(s for bid, s in fused if bid == "only_a")
    assert shared_score > only_a_score


def test_rrf_fuse_empty_lists_returns_empty():
    assert _rrf_fuse([]) == []


def test_minmax_norm_all_equal_returns_zeros():
    assert _minmax_norm([3.0, 3.0, 3.0]) == [0.0, 0.0, 0.0]


def test_fuse_retrieval_rankings_weighted_prefers_higher_bm25():
    b1 = _blk("a", "alpha token")
    b2 = _blk("b", "beta token")
    bm = [(b1, 10.0), (b2, 1.0)]
    ve = [(b2, 10.0), (b1, 1.0)]
    merged = _fuse_retrieval_rankings(bm, ve, fusion="weighted", w_bm25=1.0, w_vec=0.0)
    assert merged[0][0].block_id == "a"


def test_fuse_retrieval_rankings_rrf_empty_when_no_lists():
    assert _fuse_retrieval_rankings([], [], fusion="rrf", w_bm25=0.0, w_vec=0.0) == []


def test_dedupe_keep_best_keeps_max_score():
    b = _blk("x", "text")
    out = _dedupe_keep_best([(b, 0.2), (b, 0.9), (b, 0.5)])
    assert len(out) == 1
    assert out[0][1] == 0.9


def test_expand_structure_includes_parent_and_child():
    heading = Block(
        block_id="h1",
        doc_id="d",
        type="heading",
        text="Section",
        heading_path=("Section",),
    )
    child = _blk("p1", "body", parent="h1", path=("Section",))
    by_id = {heading.block_id: heading, child.block_id: child}
    by_parent = {heading.block_id: [child]}
    expanded = _expand_structure(by_id, by_parent, [child])
    ids = {b.block_id for b in expanded}
    assert "h1" in ids and "p1" in ids


def test_expand_graph_follows_link_to_edge():
    a = _blk("a", "seed")
    b = _blk("b", "linked")
    by_id = {a.block_id: a, b.block_id: b}
    edge = Edge(src="a", dst="b", type="link_to")
    out_e = _edges_out([edge])
    in_e = _edges_in([edge])
    expanded = _expand_graph(by_id, out_e, in_e, [a], edge_types={"link_to"}, hops=1)
    assert {x.block_id for x in expanded} == {"a", "b"}


def test_allowed_by_constraints_metadata_and_acl():
    doc = Document(doc_id="d", metadata={"team": "alpha", "acl": ["alice"]})
    block_ok = Block(block_id="b1", doc_id="d", type="paragraph", text="x")
    block_bad_team = Block(block_id="b2", doc_id="d", type="paragraph", text="y")
    wrong_team_doc = Document(doc_id="d2", metadata={"team": "beta", "acl": ["alice"]})
    assert _allowed_by_constraints(
        block=block_ok, doc=doc, required_doc_metadata={"team": "alpha"}, principal="alice"
    )
    assert not _allowed_by_constraints(
        block=block_bad_team,
        doc=wrong_team_doc,
        required_doc_metadata={"team": "alpha"},
        principal="alice",
    )
    locked_doc = Document(doc_id="d3", metadata={"acl": ["bob"]})
    assert not _allowed_by_constraints(
        block=block_ok, doc=locked_doc, required_doc_metadata=None, principal="alice"
    )


def test_confidence_normalize_and_score_boundaries():
    blocks = [_blk(f"b{i}", "t") for i in range(4)]
    scored = sorted([(blocks[i], float(i)) for i in range(4)], key=lambda x: x[1], reverse=True)
    top, second = _confidence_normalize_top_two(scored)
    assert top >= second
    assert _score_confidence(0.9, 0.5) == ConfidenceLevel.HIGH
    assert _score_confidence(0.55, 0.54) == ConfidenceLevel.LOW


def test_confidence_normalize_tied_scores_use_degenerate_mapping():
    b = _blk("only", "one")
    top, second = _confidence_normalize_top_two([(b, 1.0)])
    assert top == 1.0 and second == 0.0
    tied = [_blk("a", "a"), _blk("b", "b")]
    top2, second2 = _confidence_normalize_top_two([(tied[0], 5.0), (tied[1], 5.0)])
    assert _score_confidence(top2, second2) == ConfidenceLevel.MEDIUM


def test_safe_title_prefers_heading_path():
    b = Block(block_id="x", doc_id="d", type="paragraph", text="ignored", heading_path=("A", "B"))
    assert _safe_title(b) == "A > B"


def test_with_trace_merges_payload():
    res = RetrievalResult(query="q", blocks=[], confidence=ConfidenceLevel.LOW)
    merged = _with_trace(res, {"step": "test"})
    assert merged.trace["step"] == "test"
    assert merged.query == "q"
