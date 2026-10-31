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


def test_ingest_multipart_eml_prefers_plain_text_and_threads_on_references():
    from email.message import EmailMessage

    from ragnav.ingest.email import ingest_eml_bytes_graph

    msg = EmailMessage()
    msg["From"] = "a@example.com"
    msg["To"] = "b@example.com"
    msg["Subject"] = "Re: launch"
    msg["Message-Id"] = "<2@example.com>"
    msg["References"] = "<0@example.com> <1@example.com>"
    msg.set_content("Plain body wins.")
    msg.add_alternative("<p>HTML body</p>", subtype="html")

    g = ingest_eml_bytes_graph(msg.as_bytes(), name="t.eml", metadata={"box": "inbox"})
    body = next(b for b in g.blocks.values() if b.block_id.endswith("#body"))
    assert body.text == "Plain body wins."
    reply = [e for e in g.edges if e.type == "reply_to"]
    assert reply and reply[0].dst == "email:msgid:<1@example.com>"


def test_ingest_html_only_eml_falls_back_to_first_text_part():
    from email.message import EmailMessage

    from ragnav.ingest.email import ingest_eml_bytes_graph

    msg = EmailMessage()
    msg["Subject"] = "html only"
    msg.add_alternative("<p>Only HTML</p>", subtype="html")
    msg.add_attachment(b"\x00\x01", maintype="application", subtype="octet-stream")
    g = ingest_eml_bytes_graph(msg.as_bytes())
    body = next(b for b in g.blocks.values() if b.block_id.endswith("#body"))
    assert "Only HTML" in body.text


def test_same_thread_edges_link_real_messages():
    messages = [
        {"ts": "1", "text": "Root message", "thread_ts": "1"},
        {"ts": "2", "text": "First reply", "thread_ts": "1"},
        {"ts": "3", "text": "Second reply", "thread_ts": "1"},
        {"ts": "4", "text": "Unrelated", "user": "u3"},
    ]
    g = ingest_slack_messages_graph(messages, name="t.json")
    ids = set(g.blocks)
    same = {(e.src, e.dst) for e in g.edges if e.type == "same_thread"}
    assert same, "expected same_thread edges"
    assert all(src in ids and dst in ids for src, dst in same)
    members = {b for b in ids if not b.endswith("m4")}
    assert {(a, b) for a in members for b in members if a != b} == same
