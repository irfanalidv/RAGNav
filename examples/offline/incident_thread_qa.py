"""
Incident-channel Q&A: answer from the whole Slack thread, not one message.

What it shows
    Slack-style messages are ingested with ``ingest_slack_messages_graph`` into a block
    graph with ``reply_to`` (reply -> thread root) and ``next`` edges. A plain keyword hit
    finds one message; ``expand_graph=True`` with two hops walks reply -> root -> other
    replies, so the context also has the original report and the fix posted by others,
    and nothing from unrelated chatter in the channel.

What it needs
    ``pip install ragnav``. Offline BM25 retrieval; no API key, no model download.

Run
    python examples/offline/incident_thread_qa.py
"""

from __future__ import annotations

from ragnav import RAGNavIndex, RAGNavRetriever
from ragnav.ingest.chat import ingest_slack_messages_graph

MESSAGES = [
    {"ts": "100.0", "user": "dana", "text": "Checkout API is returning 502s since 09:40."},
    {
        "ts": "101.0",
        "user": "eli",
        "text": "Seeing it too. Started right after deploy 4182.",
        "thread_ts": "100.0",
    },
    {
        "ts": "102.0",
        "user": "dana",
        "text": "Root cause: the new connection pool limit is 5, payments needs 50.",
        "thread_ts": "100.0",
    },
    {
        "ts": "103.0",
        "user": "eli",
        "text": "Rolled back deploy 4182; error rate back to normal.",
        "thread_ts": "100.0",
    },
    {"ts": "200.0", "user": "fay", "text": "Reminder: team lunch moved to Thursday."},
]


def show(label: str, blocks: list) -> None:  # type: ignore[type-arg]
    print(label)
    for b in blocks:
        print(f"  {b.metadata.get('user', '?'):>5}: {b.text}")
    print()


def main() -> int:
    graph = ingest_slack_messages_graph(MESSAGES, name="incident.json", channel="#incidents")
    index = RAGNavIndex.build(
        documents=list(graph.documents.values()),
        blocks=list(graph.blocks.values()),
        edges=graph.edges,
        build_vectors=False,
    )
    retriever = RAGNavRetriever(index=index)
    query = "What was the root cause of the checkout 502 errors?"
    print(f"Q: {query}\n")

    plain = retriever.retrieve(query, k_final=1, use_vectors=False, expand_structure=False)
    show("Keyword hit only:", plain.blocks)

    threaded = retriever.retrieve(
        query,
        k_final=1,
        use_vectors=False,
        expand_structure=False,
        expand_graph=True,
        graph_edge_types={"reply_to"},
        graph_hops=2,  # hit -> thread root -> the root's other replies
    )
    show("With thread expansion (expand_graph=True):", threaded.blocks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
