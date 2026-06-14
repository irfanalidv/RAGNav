from __future__ import annotations

from ragnav.graphrag.extract import (
    EntityExtractConfig,
    _candidates_from_text,
    _entity_id,
    _lexicon_mentions,
    _norm_name,
    _phrase_candidates,
    build_entity_graph,
)
from ragnav.models import Block


def test_norm_name_and_entity_id_stable():
    assert _norm_name("  BERT   model  ") == "BERT model"
    assert _entity_id("model", "BERT").startswith("model:")


def test_candidates_from_text_finds_acronyms_and_capitalized():
    cands = _candidates_from_text("We evaluate BERT on the SQuAD dataset.")
    assert "BERT" in cands
    assert "SQuAD" in cands


def test_phrase_candidates_extracts_dataset_and_metric_phrases():
    text = "We evaluated on the SQuAD dataset and report F1 for question answering."
    phrases = _phrase_candidates(text, max_phrases=5)
    assert any("SQuAD" in p for p in phrases)
    assert any("question answering" in p.lower() for p in phrases)


def test_lexicon_mentions_finds_squad_and_f1():
    mentions = _lexicon_mentions("Results on squad with f1 score for reading comprehension.")
    assert "squad" in mentions["dataset"]
    assert "f1" in mentions["metric"]


def test_build_entity_graph_links_model_to_dataset_and_metric():
    blocks = [
        Block(
            block_id="m1",
            doc_id="paper",
            type="paragraph",
            text=(
                "We evaluate BERT on the SQuAD dataset using accuracy and F1 metrics "
                "for question answering tasks."
            ),
            heading_path=("Methods",),
        )
    ]
    graph = build_entity_graph(blocks, cfg=EntityExtractConfig(max_entities_per_block=8))
    entity_types = {e.type for e in graph.entities.values()}
    assert "dataset" in entity_types
    assert "model" in entity_types or "method" in entity_types
    rel_types = {r.type for r in graph.relations}
    assert "evaluated_on" in rel_types or "uses_metric" in rel_types or "defined_in" in rel_types
