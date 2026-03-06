from __future__ import annotations

import re
from dataclasses import dataclass

from ..models import Block
from .entities import Entity, Relation
from .graph import EntityGraph
from . import lexicon


_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-+/]*")

# Very lightweight "paper entity" patterns. These are deliberately conservative:
# better to miss than hallucinate relations.
_DATASET_CUES = re.compile(r"\bdataset(?:s)?\b", re.IGNORECASE)
_EVAL_ON_RE = re.compile(r"\b(?:evaluate|evaluated|evaluation)\s+(?:on|using)\b", re.IGNORECASE)
_USES_METRIC_RE = re.compile(r"\b(?:metric|metrics)\b|\b(?:accuracy|f1|bleu|rouge|mrr|ndcg|anls)\b", re.IGNORECASE)
_TASK_RE = re.compile(r"\b(?:task|tasks)\b|\b(?:classification|retrieval|summarization|qa|question answering)\b", re.IGNORECASE)

_PAREN_ALIAS_RE = re.compile(r"\b([A-Z][A-Za-z0-9][A-Za-z0-9 \-/]{2,60})\s*\(([^)]+)\)")
_ON_DATASET_RE = re.compile(r"\bon\s+the\s+([A-Z][A-Za-z0-9][A-Za-z0-9 \-/]{2,60})\s+(?:dataset|benchmark)\b", re.IGNORECASE)
_REPORT_METRIC_RE = re.compile(r"\b(?:report|reports|reported)\s+([A-Za-z][A-Za-z0-9 \-/]{1,30})\b", re.IGNORECASE)
_FOR_TASK_RE = re.compile(r"\bfor\s+(question answering|reading comprehension|classification|retrieval|summarization|natural language inference)\b", re.IGNORECASE)


@dataclass(frozen=True)
class EntityExtractConfig:
    max_entities_per_block: int = 12
    max_phrases_per_block: int = 10
    enable_lexicon: bool = True


def _norm_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip())


def _entity_id(typ: str, name: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return f"{typ}:{key}" if key else f"{typ}:unknown"


def _candidates_from_text(text: str) -> list[str]:
    # Heuristic: capitalized tokens and acronyms are common in papers (BERT, SQuAD, ImageNet).
    toks = _TOKEN.findall(text or "")
    out: list[str] = []
    for t in toks:
        if len(t) < 2:
            continue
        if t.isupper() and len(t) <= 10:
            out.append(t)
            continue
        if t[0].isupper() and any(c.islower() for c in t[1:]):
            out.append(t)
            continue
    return out


def _phrase_candidates(text: str, *, max_phrases: int) -> list[str]:
    """
    Try to capture multi-token entities (dataset names, benchmarks, metrics).
    """
    out: list[str] = []
    for m in _ON_DATASET_RE.finditer(text or ""):
        out.append(m.group(1).strip())
        if len(out) >= max_phrases:
            return out
    for m in _PAREN_ALIAS_RE.finditer(text or ""):
        lhs = m.group(1).strip()
        rhs = m.group(2).strip()
        if 3 <= len(lhs) <= 80:
            out.append(lhs)
        if 2 <= len(rhs) <= 30:
            out.append(rhs)
        if len(out) >= max_phrases:
            return out
    for m in _REPORT_METRIC_RE.finditer(text or ""):
        out.append(m.group(1).strip())
        if len(out) >= max_phrases:
            return out
    for m in _FOR_TASK_RE.finditer(text or ""):
        out.append(m.group(1).strip())
        if len(out) >= max_phrases:
            return out
    return out[:max_phrases]


def _lexicon_mentions(text: str) -> dict[str, set[str]]:
    low = (text or "").lower()
    out: dict[str, set[str]] = {"dataset": set(), "model": set(), "metric": set(), "task": set()}
    for d in lexicon.DATASETS:
        if d in low:
            out["dataset"].add(d)
    for m in lexicon.MODELS:
        if m in low:
            out["model"].add(m)
    for met in lexicon.METRICS:
        if met in low:
            out["metric"].add(met)
    for t in lexicon.TASKS:
        if t in low:
            out["task"].add(t)
    return out


def build_entity_graph(blocks: list[Block], *, cfg: EntityExtractConfig = EntityExtractConfig()) -> EntityGraph:
    """
    Build an entity graph from blocks with provenance.

    This is NOT a full NER/RE system; it's a lightweight baseline that:
    - extracts candidates (capitalized tokens / acronyms)
    - infers coarse types using local lexical cues
    - creates a few high-value relations with evidence_block_ids
    """
    g = EntityGraph()

    # entity registry by normalized name+type
    by_key: dict[tuple[str, str], str] = {}

    def get_or_add(typ: str, name: str) -> str:
        name_n = _norm_name(name)
        key = (typ, name_n.lower())
        eid = by_key.get(key)
        if eid:
            return eid
        eid = _entity_id(typ, name_n)
        # avoid collisions: append suffix if needed
        base = eid
        n = 2
        while eid in g.entities and g.entities[eid].name.lower() != name_n.lower():
            eid = f"{base}-{n}"
            n += 1
        g.add_entity(Entity(entity_id=eid, name=name_n, type=typ))  # type: ignore[arg-type]
        by_key[key] = eid
        return eid

    for b in blocks:
        text = b.text or ""
        token_cands = _candidates_from_text(text)[: cfg.max_entities_per_block]
        phrase_cands = _phrase_candidates(text, max_phrases=cfg.max_phrases_per_block)
        lex = _lexicon_mentions(text) if cfg.enable_lexicon else {"dataset": set(), "model": set(), "metric": set(), "task": set()}

        cands = list(dict.fromkeys(phrase_cands + token_cands))[: cfg.max_entities_per_block]
        if not cands and not any(lex.values()):
            continue

        # very coarse block cues
        has_dataset = bool(_DATASET_CUES.search(text))
        has_eval = bool(_EVAL_ON_RE.search(text))
        has_metric = bool(_USES_METRIC_RE.search(text))
        has_task = bool(_TASK_RE.search(text))

        # Create entities
        dataset_ids: list[str] = []
        model_ids: list[str] = []
        metric_ids: list[str] = []
        task_ids: list[str] = []

        # Lexicon-driven entities (more reliable)
        for d in sorted(list(lex["dataset"]))[:6]:
            dataset_ids.append(get_or_add("dataset", d.upper() if d.isupper() else d))
        for m in sorted(list(lex["model"]))[:6]:
            model_ids.append(get_or_add("model", m.upper() if m.isupper() else m))
        for met in sorted(list(lex["metric"]))[:6]:
            metric_ids.append(get_or_add("metric", met))
        for t in sorted(list(lex["task"]))[:6]:
            task_ids.append(get_or_add("task", t))

        for name in cands:
            # Guess type from token shape and block cues.
            low = name.lower()
            if cfg.enable_lexicon:
                if low in lexicon.DATASETS:
                    dataset_ids.append(get_or_add("dataset", name))
                    continue
                if low in lexicon.MODELS:
                    model_ids.append(get_or_add("model", name))
                    continue
                if low in lexicon.METRICS:
                    metric_ids.append(get_or_add("metric", name))
                    continue
                if low in lexicon.TASKS:
                    task_ids.append(get_or_add("task", name))
                    continue

            if has_dataset and len(name) >= 3 and (name[0].isupper() or name.isupper()):
                dataset_ids.append(get_or_add("dataset", name))
                continue
            if has_metric and low in {"accuracy", "f1", "bleu", "rouge", "mrr", "ndcg", "anls", "exact match", "em"}:
                metric_ids.append(get_or_add("metric", name))
                continue
            if has_task and low in {"qa", "question answering", "reading comprehension", "natural language inference"}:
                task_ids.append(get_or_add("task", "Question Answering" if low == "qa" else name))
                continue

            # Default: treat as model/method if it looks like an acronym or common model token.
            if name.isupper() and len(name) <= 10:
                model_ids.append(get_or_add("model", name))
            else:
                model_ids.append(get_or_add("method", name))

        # Relations (high precision, low recall)
        evidence = (b.block_id,)

        if has_eval and dataset_ids:
            for mid in model_ids[:3]:
                for did in dataset_ids[:3]:
                    g.add_relation(Relation(src=mid, dst=did, type="evaluated_on", evidence_block_ids=evidence))

        if has_metric and metric_ids:
            for mid in model_ids[:3]:
                for met in metric_ids[:3]:
                    g.add_relation(Relation(src=mid, dst=met, type="uses_metric", evidence_block_ids=evidence))

        if has_task and task_ids:
            for mid in model_ids[:3]:
                for tid in task_ids[:3]:
                    g.add_relation(Relation(src=mid, dst=tid, type="addresses_task", evidence_block_ids=evidence))

        # Dataset -> task linkage (useful multi-hop for "what is dataset used for?")
        if dataset_ids and task_ids:
            for did in dataset_ids[:3]:
                for tid in task_ids[:3]:
                    g.add_relation(Relation(src=did, dst=tid, type="described_as", evidence_block_ids=evidence))

        # Link to a "paper section" if present
        if b.heading_path:
            sec_name = " > ".join(b.heading_path)
            sid = get_or_add("paper_section", sec_name)
            for ent_id in (dataset_ids + model_ids + metric_ids + task_ids)[:6]:
                g.add_relation(Relation(src=ent_id, dst=sid, type="defined_in", evidence_block_ids=evidence))

    g.build_indexes()
    return g

