from __future__ import annotations

import pytest

from ragnav.answering.inline_citations import (
    answer_with_inline_citations,
    build_cited_context,
    validate_inline_citations,
)
from ragnav.llm.fake import FakeLLMClient
from ragnav.models import Block


def test_build_cited_context_includes_block_ids():
    blocks = [
        Block(
            block_id="b1",
            doc_id="d",
            type="paragraph",
            text="Paris is the capital.",
            anchors={"page": 1},
            heading_path=("Intro",),
        )
    ]
    ctx = build_cited_context(blocks)
    assert "BLOCK_ID=b1" in ctx
    assert "Paris is the capital." in ctx


def test_validate_inline_citations_flags_unknown_ids():
    allowed = {"b1"}
    bad = "Paris is great [[b2]]."
    v = validate_inline_citations(bad, allowed_block_ids=allowed)
    assert v["ok"] is False
    assert "b2" in v["unknown_block_ids"]


def test_validate_inline_citations_requires_per_sentence_citation():
    allowed = {"b1"}
    bad = "First sentence. Second without cite."
    v = validate_inline_citations(bad, allowed_block_ids=allowed)
    assert v["ok"] is False
    assert v["sentences_missing_citations"]


def test_answer_with_inline_citations_accepts_valid_llm_output():
    blocks = [
        Block(block_id="b1", doc_id="d", type="paragraph", text="Paris is the capital of France.")
    ]

    class CitingLLM(FakeLLMClient):
        def chat(self, *, messages, model=None, temperature=0):
            return "Paris is the capital of France [[b1]]."

    out = answer_with_inline_citations(
        llm=CitingLLM(),
        query="capital of France?",
        blocks=blocks,
    )
    assert "b1" in out.cited_block_ids
    assert "[[b1]]" in out.answer


def test_answer_with_inline_citations_raises_on_invalid_citations():
    blocks = [Block(block_id="b1", doc_id="d", type="paragraph", text="text")]

    class BadLLM(FakeLLMClient):
        def chat(self, *, messages, model=None, temperature=0):
            return "Unsupported claim with no citation."

    with pytest.raises(ValueError, match="citation validation failed"):
        answer_with_inline_citations(llm=BadLLM(), query="q", blocks=blocks)
