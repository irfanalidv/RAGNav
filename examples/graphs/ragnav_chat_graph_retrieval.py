from __future__ import annotations

"""
Demo: ingest a Slack-like chat into a BlockGraph, then do graph-aware retrieval.
"""

from ragnav.env import load_env
from ragnav.graph import Edge
from ragnav.ingest.chat import ingest_slack_messages_graph
from ragnav.llm.mistral import MistralClient
from ragnav.retrieval import RAGNavIndex, RAGNavRetriever


def main() -> None:
    load_env()
    llm = MistralClient()

    messages = [
        {"ts": "1.0", "user": "alice", "text": "We should rotate the Mistral key.", "thread_ts": "1.0"},
        {"ts": "2.0", "user": "bob", "text": "Agree. Also add .env to .gitignore.", "thread_ts": "1.0"},
        {"ts": "3.0", "user": "alice", "text": "Done. What's next for observability?", "thread_ts": "1.0"},
        {"ts": "4.0", "user": "carol", "text": "Add timing traces + retrieval cache.", "thread_ts": "1.0"},
    ]

    g = ingest_slack_messages_graph(messages, name="slack-demo.json", channel="#eng")

    index = RAGNavIndex.build(
        documents=list(g.documents.values()),
        blocks=list(g.blocks.values()),
        llm=llm,
        build_vectors=False,  # keep this cheap for demo
        edges=g.edges,
    )
    retriever = RAGNavRetriever(index=index, llm=llm)

    query = "What should we do next for observability?"
    res = retriever.retrieve(query, use_vectors=False, expand_graph=True, graph_hops=1)

    print("== Retrieved blocks (graph-aware) ==")
    for b in res.blocks[:6]:
        print(f"- {b.anchors} {b.metadata.get('user')}: {b.text[:120]}")


if __name__ == "__main__":
    main()

