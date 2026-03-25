from __future__ import annotations

from ragnav.ingest.markdown import ingest_markdown_string


def test_headings_and_paragraphs():
    md = """# Title

Intro paragraph.

## Section A

Body under A.
"""
    doc, blocks = ingest_markdown_string(md, name="t.md")
    types = [b.type for b in blocks]
    assert "heading" in types
    assert "paragraph" in types
    headings = [b for b in blocks if b.type == "heading"]
    assert any("Title" in h.text for h in headings)
    assert any("Section A" in h.text for h in headings)


def test_heading_path_nested():
    md = """# Alpha

x

## Beta

y
"""
    _doc, blocks = ingest_markdown_string(md, name="t.md")
    beta_heading = next(b for b in blocks if b.type == "heading" and "Beta" in b.text)
    assert beta_heading.heading_path == ("Alpha", "Beta")


def test_fenced_code_block():
    md = """# Doc

```python
a = 1
```

After.
"""
    _doc, blocks = ingest_markdown_string(md, name="t.md")
    code_blocks = [b for b in blocks if b.type == "code"]
    assert len(code_blocks) == 1
    assert "a = 1" in code_blocks[0].text
