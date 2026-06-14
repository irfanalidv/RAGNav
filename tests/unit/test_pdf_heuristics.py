from __future__ import annotations

from ragnav.papers.pdf_heuristics import (
    find_app_mentions,
    find_fig_mentions,
    find_sec_mentions,
    find_tab_mentions,
    is_heading_line,
)


def test_is_heading_line_numbered_section():
    level, title, num = is_heading_line("3.2 Retrieval Methods")
    assert level == 2
    assert title == "Retrieval Methods"
    assert num == "3.2"


def test_is_heading_line_canonical_introduction():
    level, title, num = is_heading_line("Introduction")
    assert level == 1
    assert title == "Introduction"
    assert num is None


def test_find_mentions_in_body_text():
    text = "See Figure 3 and Table 2 in Section 4.1; details in Appendix A."
    assert "3" in find_fig_mentions(text)
    assert "2" in find_tab_mentions(text)
    assert "4.1" in find_sec_mentions(text)
    assert "A" in find_app_mentions(text)
