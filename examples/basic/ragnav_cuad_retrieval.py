"""
CUAD Legal Contract Retrieval — Quick Demo
===========================================
Problem  : Retrieves specific contract clauses from a legal document corpus,
           showing how structure-graph expansion recovers section context.
Module   : ragnav.retrieval (RAGNavIndex, RAGNavRetriever)
Dataset  : theatticusproject/cuad-qa (HuggingFace, CC BY 4.0)
           Same JSON as the official CUAD `data.zip` release.
Install  : pip install ragnav[embeddings] datasets
Env vars : NONE
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import time
import zipfile
from typing import Any
from urllib.request import urlopen

from ragnav import RAGNavIndex, RAGNavRetriever
from ragnav.ingest.markdown import MarkdownIngestOptions, ingest_markdown_string_graph
from ragnav.llm.fake import FakeLLMClient

_CUAD_DATA_ZIP = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"


def _contract_doc_id(context: str) -> str:
    h = hashlib.sha256(context.encode("utf-8")).hexdigest()[:12]
    return "cuad:%s.md" % h


def _load_cuad_test_rows() -> list[dict[str, Any]]:
    with urlopen(_CUAD_DATA_ZIP, timeout=300) as resp:
        zf = zipfile.ZipFile(io.BytesIO(resp.read()))
    raw = zf.read("test.json")
    data = json.loads(raw.decode("utf-8"))
    rows: list[dict[str, Any]] = []
    for article in data["data"]:
        title = article.get("title", "") or ""
        for para in article["paragraphs"]:
            ctx = (para.get("context") or "").strip()
            for qa in para["qas"]:
                answers = qa.get("answers") or []
                texts = [str(a.get("text", "")).strip() for a in answers if a.get("text")]
                rows.append(
                    {
                        "id": qa["id"],
                        "title": title,
                        "context": ctx,
                        "question": (qa.get("question") or "").strip(),
                        "answers": {"text": texts},
                    }
                )
    return rows


def _merge_corpus(sample_rows: list[dict[str, Any]]) -> tuple[list[Any], list[Any], list[Any]]:
    opts = MarkdownIngestOptions(doc_id_prefix="cuad:")
    seen_ctx: set[str] = set()
    documents: list[Any] = []
    blocks: list[Any] = []
    edges: list[Any] = []
    for row in sample_rows:
        ctx = row["context"]
        if ctx in seen_ctx:
            continue
        seen_ctx.add(ctx)
        h = hashlib.sha256(ctx.encode("utf-8")).hexdigest()[:12]
        name = "%s.md" % h
        g = ingest_markdown_string_graph(
            ctx, name=name, metadata={"title": row.get("title", "")}, opts=opts
        )
        documents.extend(g.documents.values())
        blocks.extend(g.blocks.values())
        edges.extend(g.edges)
    return documents, blocks, edges


def _gold_block_ids_for_row(row: dict[str, Any], all_blocks: list[Any]) -> frozenset[str]:
    texts = [t for t in row["answers"].get("text") or [] if t.strip()]
    if not texts:
        return frozenset()
    doc_id = _contract_doc_id(row["context"])
    out: list[str] = []
    for b in all_blocks:
        if b.doc_id != doc_id:
            continue
        low = b.text.lower()
        if any(a.lower() in low for a in texts):
            out.append(b.block_id)
    return frozenset(out)


def main() -> None:
    model_name = "all-MiniLM-L6-v2"
    print("Loading CUAD (test split via data.zip)…", flush=True)
    all_rows = _load_cuad_test_rows()
    rng = random.Random(42)
    pool = [r for r in all_rows if r["answers"].get("text")]
    rng.shuffle(pool)
    sample_cap = 500
    sample_rows = pool[:sample_cap]
    print("Building markdown-graph index (unique contracts in sample)…", flush=True)
    t0 = time.perf_counter()
    documents, blocks, edges = _merge_corpus(sample_rows)
    llm = FakeLLMClient()
    index = RAGNavIndex.build(
        documents=documents,
        blocks=blocks,
        edges=edges,
        llm=llm,
        use_sentence_transformers=True,
        vector_model=model_name,
        embed_batch_size=16,
    )
    t_build = time.perf_counter() - t0
    print(
        "Index build done in %.2fs (%d blocks, %d edges)\n" % (t_build, len(blocks), len(edges)),
        flush=True,
    )
    retriever = RAGNavRetriever(index=index, llm=llm)

    demo: list[dict[str, Any]] = []
    for row in sample_rows:
        if len(demo) >= 5:
            break
        gold_ids = _gold_block_ids_for_row(row, blocks)
        if not gold_ids:
            continue
        demo.append(row)

    graph_types = {"next", "parent"}
    for row in demo:
        q = row["question"]
        texts = row["answers"]["text"]
        gold = texts[0] if texts else ""
        tq = time.perf_counter()
        res = retriever.retrieve(
            q,
            top_k=5,
            bm25_weight=0.5,
            vector_weight=0.5,
            expand_structure=True,
            expand_graph=True,
            graph_edge_types=graph_types,
        )
        tq_ms = (time.perf_counter() - tq) * 1000.0
        gold_ids = _gold_block_ids_for_row(row, blocks)
        top = res.blocks[0] if res.blocks else None
        snippet = (top.text[:160] + "…") if top and len(top.text) > 160 else (top.text if top else "")
        hit = top is not None and any(a.lower() in top.text.lower() for a in texts if a)
        mark = "✓" if hit else "✗"
        in_top5 = bool(gold_ids & {b.block_id for b in res.blocks[:5]})
        print("%s Q: %s" % (mark, q))
        print("    Gold: %s" % gold[:120])
        print("    Top:  %s" % snippet.replace("\n", " "))
        print("    Query time: %.0f ms | gold in top-5: %s\n" % (tq_ms, in_top5))


if __name__ == "__main__":
    main()

# --- Expected output (run from repo root: PYTHONPATH=. python examples/basic/ragnav_cuad_retrieval.py) ---
# Loading CUAD (test split via data.zip)…
# Building markdown-graph index (unique contracts in sample)…
# Index build done in 11.03s (99 blocks, 99 edges)
#
# ✓ Q: Highlight the parts (if any) of this contract related to "Document Name" that should be reviewed by a lawyer. Details: The name of the contract
#     Gold: Endorsement Agreement
#     Top:  EXHIBIT 10.1   ENDORSEMENT AGREEMENT   This Endorsement Agreement ("Agreement") made October 30, 2017, between National Football League Alumni - Northern Califo…
#     Query time: 43 ms | gold in top-5: False
#
# ✗ Q: Highlight the parts (if any) of this contract related to "Agreement Date" that should be reviewed by a lawyer. Details: The date of the contract
#     Gold: March 29, 1999
#     Top:  Exhibit 10.16  EL POLLO LOCO® FRANCHISE DEVELOPMENT AGREEMENT  Dated: ____________________  Territory: Developer:  (Disclosure Document Control No. 032619)  TAB…
#     Query time: 11 ms | gold in top-5: False
#
# ✗ Q: Highlight the parts (if any) of this contract related to "Change Of Control" that should be reviewed by a lawyer. Details: Does one party have the right to terminate or is consent or notice required of the counterparty if such party undergoes a change of control, such as a merger, stock sale, transfer of all or substantially all of its assets or business, or assignment by operation of law?
#     Gold: Any change in the control of you shall be deemed a transfer for purposes of this Agreement.
#     Top:  Exhibit 10.16  EL POLLO LOCO® FRANCHISE DEVELOPMENT AGREEMENT  Dated: ____________________  Territory: Developer:  (Disclosure Document Control No. 032619)  TAB…
#     Query time: 41 ms | gold in top-5: False
#
# ✗ Q: Highlight the parts (if any) of this contract related to "Volume Restriction" that should be reviewed by a lawyer. Details: Is there a fee increase or consent requirement, etc. if one party’s use of the product/services exceeds certain threshold?
#     Gold: During the Term (including any renewal Term, if any), in the event that MusclePharm shall determine to develop and intro
#     Top:  Exhibit 10.16  EL POLLO LOCO® FRANCHISE DEVELOPMENT AGREEMENT  Dated: ____________________  Territory: Developer:  (Disclosure Document Control No. 032619)  TAB…
#     Query time: 46 ms | gold in top-5: False
#
# ✗ Q: Highlight the parts (if any) of this contract related to "Non-Transferable License" that should be reviewed by a lawyer. Details: Does the contract limit the ability of a party to transfer the license being granted to a third party?
#     Gold: Company will not sublicense pass-through or otherwise grant to any third parties the rights granted to Company hereunder
#     Top:  Exhibit 10.1 NON-COMPETITION AGREEMENT AMENDMENT NO. 1   This NON-COMPETITION AGREEMENT AMENDMENT NO. 1 (this "Amendment") is entered into as of August 16, 2017…
#     Query time: 45 ms | gold in top-5: False
