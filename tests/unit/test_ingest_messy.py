from __future__ import annotations

import pytest

from ragnav.ingest.chat import ingest_slack_messages_graph
from ragnav.ingest.email import ingest_eml_bytes_graph


def test_ingest_slack_messages_graph_threads_and_next_edges():
    messages = [
        {"ts": "2", "text": "Thread reply", "thread_ts": "1", "user": "u2"},
        {"ts": "1", "text": "Root message", "user": "u1"},
        {"ts": "3", "text": "Follow-up", "user": "u1"},
    ]
    g = ingest_slack_messages_graph(messages, name="demo.json", channel="general")
    blocks = list(g.blocks.values())
    assert len(blocks) == 3
    block_ids = {b.block_id for b in blocks}
    reply_edges = [e for e in g.edges if e.type == "reply_to"]
    assert reply_edges
    assert reply_edges[0].src in block_ids
    next_edges = [e for e in g.edges if e.type == "next"]
    assert len(next_edges) == 2


def test_ingest_eml_bytes_graph_splits_header_and_body():
    raw = b"""From: alice@example.com
To: bob@example.com
Subject: Budget Q4
Message-Id: <msg-1@example.com>
Content-Type: text/plain; charset=utf-8

Revenue grew 12% in Q4.
"""
    g = ingest_eml_bytes_graph(raw, name="message.eml")
    blocks = list(g.blocks.values())
    texts = " ".join(b.text for b in blocks).lower()
    assert "revenue" in texts
    assert "budget q4" in texts
    assert any(b.type == "email_header" for b in blocks)


def test_ingest_html_string_graph_builds_blocks_and_links():
    bs4 = pytest.importorskip("bs4")
    del bs4  # used only for skip guard

    from ragnav.ingest.html import ingest_html_string_graph

    html = """
    <html><body>
      <h1>Title</h1>
      <p>Paris is the capital of France. See <a href="#refs">refs</a>.</p>
      <p>Berlin is the capital of Germany.</p>
    </body></html>
    """
    g = ingest_html_string_graph(html, name="page.html", url="https://example.com/page")
    blocks = list(g.blocks.values())
    assert len(blocks) >= 3
    combined = " ".join(b.text for b in blocks).lower()
    assert "paris" in combined and "berlin" in combined
    link_edges = [e for e in g.edges if e.type == "link_to"]
    assert link_edges
