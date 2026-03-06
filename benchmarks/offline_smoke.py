from __future__ import annotations

from ragnav.ingest.markdown import ingest_markdown_string
from ragnav.llm.fake import FakeLLMClient
from ragnav.retrieval import RAGNavIndex, RAGNavRetriever


MARKDOWN = """
# Doc A

## Section 1
This document talks about EBITDA adjustments and restructuring costs.

## Section 2
More finance details.
"""


def main() -> None:
    doc, blocks = ingest_markdown_string(MARKDOWN, name="a.md")
    fake = FakeLLMClient()

    index = RAGNavIndex.build(documents=[doc], blocks=blocks, llm=fake)
    retriever = RAGNavRetriever(index=index, llm=fake)

    q = "What EBITDA adjustments are mentioned?"
    res = retriever.retrieve(q, k_final=3)

    assert res.blocks, "expected at least one retrieved block"
    # Should retrieve something containing EBITDA in the offline setup.
    assert any("EBITDA" in b.text for b in res.blocks), "expected EBITDA-related block in results"

    routed = retriever.route_documents_by_semantics(q, top_docs=1)
    assert routed and routed[0][0] == doc.doc_id, "expected doc routing to pick the only doc"

    print("offline_smoke_ok")


if __name__ == "__main__":
    main()

