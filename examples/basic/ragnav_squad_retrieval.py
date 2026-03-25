"""
SQuAD Hybrid Retrieval — Quick Demo
=====================================
Problem  : Shows hybrid BM25+embedding retrieval finding the right Wikipedia
           passage for open-domain questions. No API key required.
Module   : ragnav.retrieval (RAGNavIndex, RAGNavRetriever)
Dataset  : rajpurkar/squad (HuggingFace, CC BY-SA 4.0)
Install  : pip install ragnav[embeddings] datasets
Env vars : NONE
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from datasets import load_dataset

from ragnav import RAGNavIndex, RAGNavRetriever
from ragnav.llm.fake import FakeLLMClient
from ragnav.models import Block, Document


def _doc_id(context: str) -> str:
    return hashlib.sha256(context.encode("utf-8")).hexdigest()[:12]


def main() -> None:
    model_name = "all-MiniLM-L6-v2"
    print("Loading SQuAD (validation)…", flush=True)
    ds = load_dataset("rajpurkar/squad", split="validation")
    rows: list[dict[str, Any]] = list(ds)

    seen: dict[str, tuple[Document, Block]] = {}
    for row in rows:
        ctx = row["context"]
        did = _doc_id(ctx)
        if did in seen:
            continue
        seen[did] = (
            Document(doc_id=did, source="squad", metadata={}),
            Block(block_id=f"{did}:p0", doc_id=did, type="paragraph", text=ctx),
        )
        if len(seen) >= 200:
            break

    documents = [p[0] for _, p in sorted(seen.items(), key=lambda x: x[0])]
    blocks = [p[1] for _, p in sorted(seen.items(), key=lambda x: x[0])]
    print("Building hybrid index (%d passages, %s)…" % (len(blocks), model_name), flush=True)
    t0 = time.perf_counter()
    llm = FakeLLMClient()
    index = RAGNavIndex.build(
        documents=documents,
        blocks=blocks,
        llm=llm,
        use_sentence_transformers=True,
        vector_model=model_name,
        embed_batch_size=32,
    )
    t_build = time.perf_counter() - t0
    print("Index build done in %.2fs\n" % t_build, flush=True)

    retriever = RAGNavRetriever(index=index, llm=llm)

    def _tag(q: str, gold: str) -> str:
        if any(ch.isdigit() for ch in q[:24]):
            return "numeric"
        if any(m in q.lower() for m in ("when ", "what year", "in what year", "which century")):
            return "date-ish"
        if len(gold.split()) <= 3:
            return "entity"
        return "other"

    picks: list[dict[str, Any]] = []
    tags_needed = {"numeric", "date-ish", "entity", "other"}
    for row in rows:
        ans_list = row.get("answers", {}).get("text") or []
        if not ans_list:
            continue
        gold = ans_list[0]
        q = row["question"]
        tag = _tag(q, gold)
        if tag in tags_needed:
            picks.append({"question": q, "gold": gold, "row": row, "tag": tag})
            tags_needed.discard(tag)
        if not tags_needed and len(picks) >= 4:
            break
    seen_q = {p["question"] for p in picks}
    for row in rows:
        if len(picks) >= 5:
            break
        q = row["question"]
        if q in seen_q:
            continue
        ans_list = row.get("answers", {}).get("text") or []
        if not ans_list:
            continue
        seen_q.add(q)
        picks.append(
            {
                "question": q,
                "gold": ans_list[0],
                "row": row,
                "tag": _tag(q, ans_list[0]),
            }
        )

    for item in picks[:5]:
        row = item["row"]
        did = _doc_id(row["context"])
        gold_bid = "%s:p0" % did
        q = item["question"]
        gold = item["gold"]
        tq = time.perf_counter()
        res = retriever.retrieve(
            q,
            top_k=5,
            bm25_weight=0.5,
            vector_weight=0.5,
            expand_structure=False,
            expand_graph=False,
        )
        tq_ms = (time.perf_counter() - tq) * 1000.0
        top = res.blocks[0] if res.blocks else None
        snippet = (top.text[:160] + "…") if top and len(top.text) > 160 else (top.text if top else "")
        hit = top is not None and gold.lower() in top.text.lower()
        mark = "✓" if hit else "✗"
        print("[%s] %s Q: %s" % (mark, item["tag"], q))
        print("    Gold: %s" % gold[:120])
        print("    Top:  %s" % snippet.replace("\n", " "))
        print("    Query time: %.0f ms | gold block id in top-5: %s\n" % (tq_ms, gold_bid in {b.block_id for b in res.blocks[:5]}))


if __name__ == "__main__":
    main()

# --- Expected output (run from repo root: PYTHONPATH=. python examples/basic/ragnav_squad_retrieval.py) ---
# Loading SQuAD (validation)…
# Building hybrid index (200 passages, all-MiniLM-L6-v2)…
# Index build done in 9.73s
#
# [✓] entity Q: Which NFL team represented the AFC at Super Bowl 50?
#     Gold: Denver Broncos
#     Top:  Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conferenc…
#     Query time: 52 ms | gold block id in top-5: True
#
# [✗] numeric Q: Where did Super Bowl 50 take place?
#     Gold: Santa Clara, California
#     Top:  On 11 July 1934, the New York Herald Tribune published an article on Tesla, in which he recalled an event that would occasionally take place while experimenting…
#     Query time: 57 ms | gold block id in top-5: True
#
# [✓] date-ish Q: What year did the Denver Broncos secure a Super Bowl title for the third time?
#     Gold: 2015
#     Top:  Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conferenc…
#     Query time: 57 ms | gold block id in top-5: True
#
# [✓] other Q: Who was limited by Denver's defense?
#     Gold: Newton was limited by Denver's defense
#     Top:  The Broncos took an early lead in Super Bowl 50 and never trailed. Newton was limited by Denver's defense, which sacked him seven times and forced him into thre…
#     Query time: 44 ms | gold block id in top-5: True
#
# [✓] entity Q: Which NFL team represented the NFC at Super Bowl 50?
#     Gold: Carolina Panthers
#     Top:  Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conferenc…
#     Query time: 16 ms | gold block id in top-5: True
