from __future__ import annotations

from ragnav.graphrag.entities import Entity, Relation
from ragnav.graphrag.graph import EntityGraph
from ragnav.graphrag.lexicon import contains_any, normalize_key
from ragnav.graphrag.retriever import EntityGraphRetriever, EntityGraphRetrieverConfig
from ragnav.models import Block


def test_lexicon_contains_any_finds_metric_phrase():
    assert contains_any("We report F1 and accuracy on SQuAD.", {"f1", "accuracy"})
    assert not contains_any("unrelated text", {"bleu"})


def test_normalize_key_lowercases():
    assert normalize_key("  BERT  ") == "bert"


def test_entity_graph_relations_indexed():
    g = EntityGraph()
    g.add_entity(Entity(entity_id="e1", name="BERT", type="model"))
    g.add_entity(Entity(entity_id="e2", name="SQuAD", type="dataset"))
    g.add_relation(
        Relation(
            src="e1",
            dst="e2",
            type="evaluated_on",
            evidence_block_ids=("b_evidence",),
        )
    )
    g.build_indexes()
    out = g.out_relations("e1", types={"evaluated_on"})
    assert out[0].dst == "e2"


def test_entity_graph_retriever_expands_evidence_blocks():
    g = EntityGraph()
    g.add_entity(Entity(entity_id="m_bert", name="BERT model", type="model"))
    g.add_entity(Entity(entity_id="d_squad", name="SQuAD dataset", type="dataset"))
    g.add_relation(
        Relation(
            src="m_bert",
            dst="d_squad",
            type="evaluated_on",
            evidence_block_ids=("ev1",),
        )
    )
    g.build_indexes()
    blocks = {
        "ev1": Block(
            block_id="ev1",
            doc_id="paper",
            type="paragraph",
            text="BERT evaluated on SQuAD with strong F1.",
        )
    }
    ret = EntityGraphRetriever(graph=g, blocks_by_id=blocks)
    matched = ret.match_entities("BERT SQuAD", cfg=EntityGraphRetrieverConfig())
    assert "m_bert" in matched
    out = ret.retrieve("BERT SQuAD")
    assert "ev1" in out["evidence_block_ids"]
    assert out["blocks"][0].text.startswith("BERT evaluated")


def test_entity_graph_retriever_hybrid_fallback_delegates():
    from ragnav.llm.fake import FakeLLMClient
    from ragnav.models import Document
    from ragnav.retrieval import RAGNavIndex, RAGNavRetriever

    llm = FakeLLMClient()
    doc = Document(doc_id="d")
    blocks = [Block(block_id="b1", doc_id="d", type="paragraph", text="Paris capital France")]
    idx = RAGNavIndex.build(documents=[doc], blocks=blocks, llm=llm, use_sentence_transformers=False)
    rag = RAGNavRetriever(index=idx, llm=llm)
    g = EntityGraph()
    egr = EntityGraphRetriever(graph=g, blocks_by_id={})
    res = egr.hybrid_fallback("Paris capital", rag_retriever=rag, k_final=2)
    assert res.blocks
    assert any("Paris" in b.text for b in res.blocks)
