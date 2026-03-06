from __future__ import annotations

from ragnav.ingest.markdown import ingest_markdown_string
from ragnav.env import load_env
from ragnav.llm.mistral import MistralClient
from ragnav.retrieval import RAGNavIndex, RAGNavRetriever
from ragnav.utils import print_wrapped


DOC_FINANCE = """
# ACME Corp Q1 FY25 Earnings (Excerpt)

## Highlights
Revenue grew 18% YoY driven by subscriptions.

## EBITDA Adjustments
We adjusted EBITDA for stock-based compensation and restructuring costs.
EBITDA margin improved due to lower hosting costs.
"""

DOC_COOKING = """
# Pasta Carbonara Notes

## Ingredients
Eggs, guanciale, pecorino, black pepper.
"""

DOC_ENGINEERING = """
# Service Architecture (Internal)

## Deployment
We deploy weekly using blue/green.
"""


def main() -> None:
    # Mirrors PageIndex "Search by Description" doc selection.
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

    query = "What EBITDA adjustments were applied?"

    print("## Step 1: Generate doc descriptions (LLM)\n")
    descs = retriever.generate_doc_descriptions()
    for doc_id, desc in descs.items():
        print(f"- {doc_id}: {desc}")

    print("\n## Step 2: Route documents by description (LLM selection)\n")
    picked = retriever.route_documents_by_description(query, descriptions=descs, top_docs=2)
    print("Picked doc_ids:", picked)

    print("\n## Step 3: Focused retrieval + answer generation\n")
    res = retriever.retrieve(query, allowed_doc_ids=set(picked), k_final=6)
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

