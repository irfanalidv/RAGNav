from __future__ import annotations

from unittest.mock import MagicMock, patch

from ragnav.llm.fake import FakeLLMClient
from ragnav.models import Block, Document
from ragnav.pipelines.hybrid import HybridRagConfig, hybrid_answer
from ragnav.pipelines.vectorless import VectorlessRagConfig, vectorless_answer, vectorless_rag_from_pdf_bytes


def test_vectorless_answer_grounded_in_retrieved_context():
    hits = [{"content": "Paris is the capital of France."}]

    def retrieve(_query: str, *, retrieval_cache=None):
        return hits

    answer = vectorless_answer(
        query="What is the capital of France?",
        retrieve_fn=retrieve,
        llm=FakeLLMClient(),
    )
    assert "Paris" in answer or "capital" in answer.lower()


def test_hybrid_answer_calls_retriever_and_llm():
    retriever = MagicMock()
    retriever.retrieve_raw.return_value = [
        {"content": "Hybrid BM25 plus embeddings improve recall."}
    ]
    llm = FakeLLMClient()
    text = hybrid_answer(
        query="How does hybrid retrieval work?",
        retriever=retriever,
        llm=llm,
        cfg=HybridRagConfig(max_blocks=3, k_final=3),
    )
    retriever.retrieve_raw.assert_called_once()
    assert "Hybrid" in text or "hybrid" in text.lower() or "fake-llm" in text


def test_vectorless_rag_from_pdf_bytes_builds_bm25_only_index():
    llm = FakeLLMClient()
    doc = Document(doc_id="pdf:demo.pdf", source="demo.pdf", metadata={})
    blocks = [
        Block(
            block_id="p0",
            doc_id="pdf:demo.pdf",
            type="paragraph",
            text="Evaluation methods include accuracy and F1.",
            anchors={"page": 1},
        )
    ]

    with patch("ragnav.pipelines.vectorless.ingest_pdf_bytes", return_value=(doc, blocks)):
        retrieve_fn, doc_id = vectorless_rag_from_pdf_bytes(
            pdf_bytes=b"%PDF-fake",
            llm=llm,
            pdf_name="demo.pdf",
            cfg=VectorlessRagConfig(max_pages=5, max_blocks=4, k_final=4),
        )
    hits = retrieve_fn("accuracy F1 evaluation")
    assert doc_id == "pdf:demo.pdf"
    assert hits
    assert any("accuracy" in h.get("content", "").lower() for h in hits)
