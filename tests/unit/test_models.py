from __future__ import annotations

import pytest

from ragnav.models import Block, Document, RetrievalResult


def test_block_is_frozen():
    b = Block(
        block_id="a",
        doc_id="d",
        type="paragraph",
        text="x",
    )
    with pytest.raises(AttributeError):
        b.text = "y"  # type: ignore[misc]


def test_document_is_frozen():
    d = Document(doc_id="d1")
    with pytest.raises(AttributeError):
        d.source = "s"  # type: ignore[misc]


def test_retrieval_result_blocks_is_list():
    b = Block(block_id="a", doc_id="d", type="paragraph", text="t")
    r = RetrievalResult(query="q", blocks=[b])
    assert isinstance(r.blocks, list)
    assert len(r.blocks) == 1


def test_block_parent_id_default_none():
    b = Block(block_id="a", doc_id="d", type="paragraph", text="x")
    assert b.parent_id is None
