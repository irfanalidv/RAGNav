"""
Contract clause lookup: numbered legal text -> clause-level blocks with provenance.

What it shows
    ``ingest_legal`` parses a numbered services agreement into section / clause blocks
    with ``parent`` / ``next`` edges. Queries return the matching clause together with
    its section heading (structure expansion), and ``expand_graph`` adds the adjacent
    clause, which is where exceptions and carve-outs usually live.

What it needs
    ``pip install ragnav``. Offline BM25 retrieval; no API key, no model download.

Run
    python examples/offline/contract_clause_lookup.py
"""

from __future__ import annotations

from ragnav import RAGNavIndex, RAGNavRetriever
from ragnav.ingest.legal import ingest_legal

AGREEMENT = """MASTER SERVICES AGREEMENT

1. DEFINITIONS
1.1 "Services" means the hosted software and support described in an Order Form.
1.2 "Confidential Information" means non-public information disclosed by either party.

2. FEES AND PAYMENT
2.1 Customer shall pay all invoices within thirty (30) days of the invoice date.
2.2 Late payments accrue interest at one percent (1%) per month.

3. TERM AND TERMINATION
3.1 This Agreement starts on the Effective Date and continues for twelve (12) months.
3.2 Either party may terminate for convenience with sixty (60) days written notice.
3.3 Either party may terminate immediately if the other party materially breaches and
fails to cure within thirty (30) days of notice.

4. LIMITATION OF LIABILITY
4.1 Each party's total liability is capped at the fees paid in the prior twelve months.
4.2 The cap in Section 4.1 does not apply to breaches of confidentiality.
"""


def show(retriever: RAGNavRetriever, query: str, **kwargs: object) -> None:
    res = retriever.retrieve(query, k_final=1, use_vectors=False, **kwargs)
    print(f"Q: {query}   (confidence={res.confidence.value})")
    for b in res.blocks:
        marker = "*" if b.block_id == res.trace["seed_block_ids"][0] else " "
        print(f"  {marker} [{b.block_id}] {b.text.splitlines()[0][:88]}")
    print()


def main() -> int:
    doc, blocks, graph = ingest_legal(AGREEMENT, doc_id="legal:msa.txt")
    index = RAGNavIndex.build(
        documents=[doc], blocks=blocks, edges=graph.edges, build_vectors=False
    )
    retriever = RAGNavRetriever(index=index)
    print(f"Parsed {len(blocks)} blocks and {len(graph.edges)} edges from the agreement.")
    print("(* = top-ranked clause; other lines are its section and neighbours)\n")

    show(retriever, "How much notice is needed to terminate for convenience?")
    show(retriever, "When are invoices due?")
    show(
        retriever,
        "Is the liability cap applicable to confidentiality breaches?",
        expand_graph=True,
        graph_edge_types={"next"},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
