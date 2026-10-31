"""
Help-centre search: structure-aware retrieval over several markdown articles.

What it shows
    Three support articles are ingested with ``ingest_markdown_string`` (headings become
    parent blocks), indexed, and queried. Each hit carries its heading path, so an answer
    can say *where* it came from; ``expand_structure`` pulls in the section header for
    context; ``required_doc_metadata`` restricts a query to one product area; and every
    result reports a HIGH / MEDIUM / LOW confidence.

What it needs
    ``pip install ragnav``. Runs offline with BM25 (no API key, no model download).
    Pass ``--embeddings`` to use hybrid BM25 + sentence-transformers instead
    (``pip install "ragnav[embeddings]"``; downloads all-MiniLM-L6-v2 once).

Run
    python examples/offline/help_center_search.py [--embeddings]
"""

from __future__ import annotations

import sys

from ragnav import RAGNavIndex, RAGNavRetriever
from ragnav.ingest.markdown import ingest_markdown_string

ARTICLES = {
    "billing.md": (
        {"area": "billing"},
        """# Billing and refunds

## Refund window
Refunds are available within 30 days of the original purchase date.
Requests after 30 days are only accepted for defective products.

## Cancelling a subscription
Cancellation takes effect at the end of the current billing period.
Unused time is not refunded.
""",
    ),
    "auth.md": (
        {"area": "developer"},
        """# API authentication

## Tokens
Every request needs a Bearer token. Tokens expire after 90 days.
Each account can have up to 5 active tokens.

## Rotating a token
Generate a new token under Settings, deploy it, then revoke the old one.
""",
    ),
    "limits.md": (
        {"area": "developer"},
        """# Rate limits

## Limits per plan
Free accounts get 100 requests per minute. Pro accounts get 1000 requests per minute.

## Exceeding the limit
The API returns HTTP 429 with a Retry-After header. Back off exponentially.
""",
    ),
}


def main() -> int:
    use_embeddings = "--embeddings" in sys.argv
    documents, blocks = [], []
    for name, (metadata, text) in ARTICLES.items():
        doc, doc_blocks = ingest_markdown_string(text, name=name, metadata=metadata)
        documents.append(doc)
        blocks.extend(doc_blocks)

    index = RAGNavIndex.build(documents=documents, blocks=blocks, build_vectors=use_embeddings)
    retriever = RAGNavRetriever(index=index)
    mode = "hybrid BM25 + embeddings" if use_embeddings else "BM25 only"
    print(f"Indexed {len(blocks)} blocks from {len(documents)} articles ({mode})\n")

    for query, filters in [
        ("What is the refund window for a purchase?", None),
        ("What happens when I exceed the rate limit?", None),
        ("How many tokens can I have?", {"area": "developer"}),
    ]:
        res = retriever.retrieve(
            query, k_final=2, use_vectors=use_embeddings, required_doc_metadata=filters
        )
        top = res.blocks[0]  # best hit first; its parents / children follow it
        evidence = next((b for b in res.blocks if b.type != "heading"), top)
        print(f"Q: {query}" + (f"   (filter: {filters})" if filters else ""))
        print(f"   confidence={res.confidence.value}  top hit=[{top.block_id}]")
        print(f"   section : {' > '.join(evidence.heading_path)}")
        print(f"   evidence: {evidence.text.splitlines()[0]}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
