from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

from ..graph import BlockGraph, Edge
from ..models import Block, Document


@dataclass(frozen=True)
class ChatIngestOptions:
    doc_id_prefix: str = "chat:"
    max_message_chars: int = 2000


def ingest_slack_messages_graph(
    messages: list[dict[str, Any]],
    *,
    name: str = "slack.json",
    channel: str | None = None,
    metadata: dict[str, Any] | None = None,
    opts: ChatIngestOptions | None = None,
) -> BlockGraph:
    """
    Slack-like message list → BlockGraph
    Expected fields per message (best-effort):
      - ts (string/float-like)
      - thread_ts (optional)
      - user (optional)
      - text

    Emits:
      - `next` edges for time order
      - `reply_to` edges within a thread (message -> parent thread root)
      - `same_thread` edges between all messages of a thread, root included
    """
    opts = opts or ChatIngestOptions()
    doc_meta: dict[str, Any] = {"type": "chat", "platform": "slack"}
    if channel:
        doc_meta["channel"] = channel
    if metadata:
        doc_meta.update(metadata)

    doc = Document(doc_id=f"{opts.doc_id_prefix}{name}", source=name, metadata=doc_meta)
    g = BlockGraph()
    g.add_document(doc)

    # sort by ts
    def _ts(m: dict[str, Any]) -> float:
        try:
            return float(m.get("ts", 0))
        except (TypeError, ValueError):
            return 0.0

    msgs = sorted(messages, key=_ts)

    id_by_ts: dict[str, str] = {}
    blocks: list[Block] = []
    for i, m in enumerate(msgs):
        ts = str(m.get("ts", f"idx:{i}"))
        text = str(m.get("text", "") or "").strip()[: opts.max_message_chars]
        if not text:
            continue
        user = m.get("user")
        thread_ts = m.get("thread_ts")

        block_id = f"{doc.doc_id}#m{ts}"
        id_by_ts[ts] = block_id
        b = Block(
            block_id=block_id,
            doc_id=doc.doc_id,
            type="message",
            text=text,
            anchors={"ts": ts, "channel": channel},
            metadata={"user": user, "thread_ts": thread_ts},
        )
        blocks.append(b)
        g.add_block(b)

    # next edges
    for a, b in itertools.pairwise(blocks):
        g.add_edge(Edge(src=a.block_id, dst=b.block_id, type="next"))

    # thread edges
    members_by_thread: dict[str, list[str]] = {}
    for b in blocks:
        thread_ts = b.metadata.get("thread_ts")
        if not thread_ts:
            continue
        thread_ts = str(thread_ts)
        members = members_by_thread.setdefault(thread_ts, [])
        root_id = id_by_ts.get(thread_ts)
        if root_id and root_id not in members:
            members.append(root_id)
        if b.block_id not in members:
            members.append(b.block_id)
        if root_id and root_id != b.block_id:
            g.add_edge(
                Edge(
                    src=b.block_id, dst=root_id, type="reply_to", metadata={"thread_ts": thread_ts}
                )
            )
    # same_thread links every message to the other messages of its thread (root included),
    # so one hop of graph expansion pulls in the whole conversation.
    for thread_ts, members in members_by_thread.items():
        for src in members:
            for dst in members:
                if src != dst:
                    g.add_edge(
                        Edge(
                            src=src, dst=dst, type="same_thread", metadata={"thread_ts": thread_ts}
                        )
                    )

    g.build_indexes()
    return g
