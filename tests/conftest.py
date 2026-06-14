from __future__ import annotations

import pytest

from ragnav.llm.fake import FakeLLMClient
from ragnav.models import Block, Document


def require_sentence_transformers() -> None:
    """Skip (not fail) when sentence_transformers or its torch stack is broken."""
    try:
        import sentence_transformers  # noqa: F401
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"sentence_transformers unavailable: {exc}")


@pytest.fixture
def offline_llm() -> FakeLLMClient:
    return FakeLLMClient()


@pytest.fixture
def paris_berlin_doc() -> tuple[Document, list[Block]]:
    doc = Document(doc_id="d1", source="demo.md", metadata={})
    blocks = [
        Block(
            block_id="b_paris",
            doc_id="d1",
            type="paragraph",
            text="Paris is the capital of France.",
        ),
        Block(
            block_id="b_berlin",
            doc_id="d1",
            type="paragraph",
            text="Berlin is the capital of Germany.",
        ),
    ]
    return doc, blocks


@pytest.fixture
def two_doc_index(offline_llm: FakeLLMClient):
    from ragnav.retrieval import RAGNavIndex

    docs = [
        Document(doc_id="finance", source="finance.md", metadata={"domain": "finance"}),
        Document(doc_id="sports", source="sports.md", metadata={"domain": "sports"}),
    ]
    blocks = [
        Block(
            block_id="f1",
            doc_id="finance",
            type="paragraph",
            text="Revenue grew 12% year over year in Q4 earnings report.",
        ),
        Block(
            block_id="s1",
            doc_id="sports",
            type="paragraph",
            text="The Denver Broncos won Super Bowl 50 against Carolina Panthers.",
        ),
    ]
    index = RAGNavIndex.build(
        documents=docs,
        blocks=blocks,
        llm=offline_llm,
        use_sentence_transformers=False,
    )
    return index, offline_llm
