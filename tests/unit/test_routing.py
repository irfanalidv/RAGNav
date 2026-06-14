from __future__ import annotations

import json

from ragnav.llm.fake import FakeLLMClient
from ragnav.models import Block, Document
from ragnav.retrieval import RAGNavIndex
from ragnav.retrieval.routing import (
    generate_doc_descriptions,
    route_documents_by_description,
    route_documents_by_metadata,
    route_documents_by_semantics,
    route_pages_by_semantics,
)


def _build_two_doc_index(llm: FakeLLMClient) -> RAGNavIndex:
    docs = [
        Document(doc_id="finance", source="finance.md", metadata={"domain": "finance"}),
        Document(doc_id="sports", source="sports.md", metadata={"domain": "sports"}),
    ]
    blocks = [
        Block(
            block_id="f1",
            doc_id="finance",
            type="paragraph",
            text="Quarterly revenue and earnings guidance for investors.",
            anchors={"page": 1},
        ),
        Block(
            block_id="s2",
            doc_id="sports",
            type="paragraph",
            text="Super Bowl 50 Denver Broncos championship game recap.",
            anchors={"page": 3},
        ),
    ]
    return RAGNavIndex.build(
        documents=docs,
        blocks=blocks,
        llm=llm,
        use_sentence_transformers=False,
    )


def test_route_documents_by_semantics_ranks_answer_doc_first():
    llm = FakeLLMClient()
    index = _build_two_doc_index(llm)
    ranked = route_documents_by_semantics(index, llm, "Denver Broncos Super Bowl", top_docs=2)
    assert ranked
    assert ranked[0][0] == "sports"
    assert ranked[0][2] >= 1


def test_route_pages_by_semantics_returns_page_numbers():
    llm = FakeLLMClient()
    index = _build_two_doc_index(llm)
    ranked = route_pages_by_semantics(index, llm, "revenue earnings", top_pages=2)
    assert ranked
    doc_id, page, score, n_blocks = ranked[0]
    assert doc_id == "finance"
    assert page == 1
    assert score > 0.0
    assert n_blocks >= 1


def test_route_documents_by_metadata_filters():
    llm = FakeLLMClient()
    index = _build_two_doc_index(llm)
    out = route_documents_by_metadata(index, required={"domain": "finance"})
    assert out == ["finance"]


def test_generate_doc_descriptions_uses_outline_titles():
    llm = FakeLLMClient()
    doc = Document(doc_id="d1", source="paper.pdf", metadata={})
    blocks = [
        Block(
            block_id="h1",
            doc_id="d1",
            type="heading",
            text="Introduction",
            heading_path=("Introduction",),
        ),
    ]
    index = RAGNavIndex.build(documents=[doc], blocks=blocks, llm=llm, build_vectors=False)
    desc = generate_doc_descriptions(index, llm, max_titles=5)
    assert "d1" in desc
    assert "Introduction" in desc["d1"] or "fake-llm" in desc["d1"]


def test_route_documents_by_description_parses_json_answer():
    class PickFinanceLLM(FakeLLMClient):
        def chat(self, *, messages, model=None, temperature=0):
            return json.dumps({"thinking": "finance", "answer": ["finance"]})

    index = _build_two_doc_index(PickFinanceLLM())
    picked = route_documents_by_description(
        index,
        PickFinanceLLM(),
        "revenue",
        descriptions={"finance": "Finance doc", "sports": "Sports doc"},
        top_docs=2,
    )
    assert picked == ["finance"]


def test_route_documents_by_description_falls_back_on_bad_json():
    llm = FakeLLMClient()
    index = _build_two_doc_index(llm)
    picked = route_documents_by_description(
        index,
        llm,
        "Broncos Super Bowl",
        descriptions={"finance": "Finance", "sports": "Sports"},
        top_docs=1,
    )
    assert picked == ["sports"]
