from __future__ import annotations

from ragnav.ingest.legal import ingest_legal


def test_legal_ingest_numbered_headings_and_lettered_subparagraph():
    text = """12. Termination
12.1 For Cause. Either party may end this agreement.
(a) Written notice is required within thirty days.
"""
    doc_id = "cuad:testcontract.md"
    doc, blocks, g = ingest_legal(text, doc_id=doc_id)
    assert doc.doc_id == doc_id
    by_text = {b.text: b for b in blocks}
    assert "12. Termination" in by_text
    assert "12.1 For Cause. Either party may end this agreement." in by_text
    h12 = by_text["12. Termination"]
    h121 = by_text["12.1 For Cause. Either party may end this agreement."]
    assert h12.type == "heading"
    assert h121.type == "heading"
    assert h121.parent_id == h12.block_id
    para = by_text["(a) Written notice is required within thirty days."]
    assert para.type == "paragraph"
    assert para.parent_id == h121.block_id
    parent_edges = {(e.src, e.dst) for e in g.edges if e.type == "parent"}
    assert (h12.block_id, h121.block_id) in parent_edges
    assert (h121.block_id, para.block_id) in parent_edges


def test_legal_ingest_all_caps_definitions_heading():
    text = """DEFINITIONS
Capitalized terms used in this agreement are defined below.
"""
    doc, blocks, g = ingest_legal(text, doc_id="legal:defs.md")
    paras = [b for b in blocks if b.type == "paragraph"]
    assert len(paras) == 1
    assert "DEFINITIONS" in paras[0].text
    assert "Capitalized terms" in paras[0].text
