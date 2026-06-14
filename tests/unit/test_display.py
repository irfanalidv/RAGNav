from __future__ import annotations

import io
import sys

from ragnav.display import create_node_mapping, print_tree, print_wrapped
from ragnav.models import Block


def test_print_wrapped_breaks_long_lines(capsys):
    print_wrapped("word " * 30, width=20)
    out = capsys.readouterr().out
    assert out.count("\n") >= 2


def test_create_node_mapping_includes_block_fields():
    b = Block(
        block_id="b1",
        doc_id="d",
        type="paragraph",
        text="hello",
        heading_path=("H",),
        anchors={"page": 2},
    )
    m = create_node_mapping([b])
    assert m["b1"]["title"] == "H"
    assert m["b1"]["anchors"]["page"] == 2


def test_print_tree_shows_heading_and_line_anchors(capsys):
    b = Block(
        block_id="b1",
        doc_id="d",
        type="paragraph",
        text="body",
        heading_path=("Section",),
        anchors={"line_start": 1, "line_end": 3},
    )
    print_tree([b])
    out = capsys.readouterr().out
    assert "Section" in out
    assert "lines 1-3" in out
