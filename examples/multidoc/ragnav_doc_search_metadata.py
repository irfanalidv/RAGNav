from __future__ import annotations

from ragnav.ingest.markdown import ingest_markdown_string
from ragnav.env import load_env
from ragnav.llm.mistral import MistralClient
from ragnav.json_utils import extract_json
from ragnav.retrieval import RAGNavIndex, RAGNavRetriever
from ragnav.utils import print_wrapped


DOC_1 = """
# ACME Corp Q1 FY25 Earnings (Excerpt)

## EBITDA Adjustments
Adjusted EBITDA excludes stock-based compensation and restructuring costs.
"""

DOC_2 = """
# ACME Corp Q4 FY24 Earnings (Excerpt)

## EBITDA Adjustments
Adjusted EBITDA excludes one-time acquisition integration costs.
"""

DOC_3 = """
# ZETA Corp Q1 FY25 Earnings (Excerpt)

## EBITDA Adjustments
Adjusted EBITDA excludes foreign exchange impacts.
"""


def main() -> None:
    # Mirrors PageIndex "Search by Metadata" idea (filter docs first), but local and deterministic.
    load_env()

    mistral = MistralClient()

    docs_blocks = []
    docs = []
    for name, content, meta in [
        ("acme_q1_fy25.md", DOC_1, {"company": "ACME", "period": "Q1 FY25"}),
        ("acme_q4_fy24.md", DOC_2, {"company": "ACME", "period": "Q4 FY24"}),
        ("zeta_q1_fy25.md", DOC_3, {"company": "ZETA", "period": "Q1 FY25"}),
    ]:
        doc, blocks = ingest_markdown_string(content, name=name, metadata=meta)
        docs.append(doc)
        docs_blocks.extend(blocks)

    index = RAGNavIndex.build(documents=docs, blocks=docs_blocks, llm=mistral)
    retriever = RAGNavRetriever(index=index, llm=mistral)

    query = "For ACME in Q1 FY25, what EBITDA adjustments were applied?"

    print("## Step 1: Query -> metadata filters (LLM, like query-to-SQL but simpler)\n")
    filter_prompt = f"""
Extract metadata filters from the query.

Query: {query}

Return JSON:
{{
  "company": "<company or null>",
  "period": "<period or null>"
}}
Return ONLY JSON.
""".strip()
    raw = mistral.chat(messages=[{"role": "user", "content": filter_prompt}], temperature=0)
    print(raw)

    # Parse (with fallback) and filter.
    fallback = {"company": "ACME", "period": "Q1 FY25"}
    try:
        parsed = extract_json(raw)
        required = {
            "company": parsed.get("company") if isinstance(parsed, dict) else None,
            "period": parsed.get("period") if isinstance(parsed, dict) else None,
        }
        required = {k: v for k, v in required.items() if v is not None}
        if not required:
            required = fallback
    except Exception:
        required = fallback

    print("\n## Step 2: Filter docs by metadata (deterministic)\n")
    allowed = set(retriever.route_documents_by_metadata(required=required))
    print("Allowed doc_ids:", sorted(allowed))

    print("\n## Step 3: Focused retrieval + answer generation\n")
    res = retriever.retrieve(query, allowed_doc_ids=allowed, k_final=6)
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

