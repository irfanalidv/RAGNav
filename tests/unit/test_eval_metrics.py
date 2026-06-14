from __future__ import annotations

import pytest

from ragnav.eval.cases import EvalCase, EvalSuite
from ragnav.eval.metrics import score_retrieval
from ragnav.models import Block


def test_score_retrieval_computes_hit_rates():
    cases = [
        EvalCase(case_id="c1", query="q1", expected_block_ids={"b1"}),
        EvalCase(case_id="c2", query="q2", expected_pages={2}),
    ]
    retrieved = [
        [Block(block_id="b1", doc_id="d", type="paragraph", text="hit")],
        [Block(block_id="x", doc_id="d", type="paragraph", text="p", anchors={"page": 2})],
    ]
    m = score_retrieval(cases, retrieved)
    assert m.n_cases == 2
    assert m.block_hit_rate == 1.0
    assert m.page_hit_rate == 1.0
    assert m.avg_blocks_returned == 1.0


def test_score_retrieval_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        score_retrieval([], [[Block(block_id="b", doc_id="d", type="paragraph", text="")]])


def test_eval_suite_holds_cases():
    suite = EvalSuite(suite_id="s1", cases=[EvalCase(case_id="c1", query="q")])
    assert suite.suite_id == "s1"
    assert len(suite.cases) == 1
