from __future__ import annotations

from ragnav.ingest.markdown import ingest_markdown_string
from ragnav.env import load_env
from ragnav.llm.mistral import MistralClient
from ragnav.retrieval import RAGNavIndex, RAGNavRetriever
from ragnav.display import print_wrapped


DOC_FINANCE = """
# ACME Corp Q1 FY25 Earnings (Excerpt)

## Highlights
Revenue grew 18% YoY driven by subscriptions.

## EBITDA Adjustments
We adjusted EBITDA for stock-based compensation and restructuring costs.
EBITDA margin improved due to lower hosting costs.

## Guidance
We expect continued margin expansion next quarter.
"""

DOC_COOKING = """
# Pasta Carbonara Notes

## Ingredients
Eggs, guanciale, pecorino, black pepper.

## Method
Temper eggs, emulsify with pasta water, avoid scrambling.
"""

DOC_ENGINEERING = """
# Service Architecture (Internal)

## Deployment
We deploy weekly using blue/green.

## Cost controls
We cap spend via budget alerts and autoscaling policies.
"""


def main() -> None:
    # Mirrors PageIndex "Search by Semantics" doc routing, but fully inside RAGNav.
    load_env()

    mistral = MistralClient()

    docs_blocks = []
    docs = []
    for name, content in [
        ("acme_q1.md", DOC_FINANCE),
        ("carbonara.md", DOC_COOKING),
        ("arch.md", DOC_ENGINEERING),
    ]:
        doc, blocks = ingest_markdown_string(content, name=name)
        docs.append(doc)
        docs_blocks.extend(blocks)

    index = RAGNavIndex.build(documents=docs, blocks=docs_blocks, llm=mistral)
    retriever = RAGNavRetriever(index=index, llm=mistral)

    query = "What EBITDA adjustments were applied, and why?"

    print("## Step 1: Multi-document routing by semantics (PageIndex-style DocScore)\n")
    ranked = retriever.route_documents_by_semantics(query, top_docs=3)
    for doc_id, score, n in ranked:
        print(f"- doc_id={doc_id}  DocScore={score:.4f}  N={n}")

    top_doc_ids = {doc_id for doc_id, _, _ in ranked[:1]}
    print("\nSelected docs:", ", ".join(sorted(top_doc_ids)))

    print("\n## Step 2: Focused retrieval inside selected docs\n")
    res = retriever.retrieve(query, allowed_doc_ids=top_doc_ids, k_final=6)
    for b in res.blocks[:8]:
        title = " > ".join(b.heading_path) if b.heading_path else b.block_id
        print(f"- {title}  {b.anchors}")

    print("\n## Step 3: Answer generation\n")
    context = "\n\n".join(b.text for b in res.blocks[:6])
    prompt = f"""Answer the question using ONLY the provided context.

Question: {query}

Context:
{context}
"""
    ans = mistral.chat(messages=[{"role": "user", "content": prompt}], temperature=0)
    print_wrapped(ans)


if __name__ == "__main__":
    main()

