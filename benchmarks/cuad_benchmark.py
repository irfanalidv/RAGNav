"""
CUAD Legal Contract Retrieval Benchmark
========================================
Problem  : Proves that RAGNav structure-graph expansion improves retrieval on
           long, nested legal contracts where the relevant clause is embedded
           deep under a section heading.
Module   : ragnav.retrieval (RAGNavIndex, RAGNavRetriever)
           ragnav.ingest.legal (ingest_legal) — numbered contract structure
           ragnav.graph (BlockGraph) — structural expansion
           ragnav.reranking (CrossEncoderReranker) — optional rerank row
Dataset  : theatticusproject/cuad-qa (HuggingFace, CC BY 4.0)
           Loaded from official CUAD release zip (same source as HF dataset script):
           https://github.com/TheAtticusProject/cuad/raw/main/data.zip
Install  : pip install ragnav[embeddings] datasets
Env vars : NONE
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.error import URLError
from urllib.request import urlopen

_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from _runner import print_table, score_any_gold  # noqa: E402

_CUAD_DATA_ZIP = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"


def _contract_doc_id(context: str) -> str:
    h = hashlib.sha256(context.encode("utf-8")).hexdigest()[:12]
    return "cuad:%s.md" % h


def _load_cuad_test_rows() -> list[dict[str, Any]]:
    try:
        with urlopen(_CUAD_DATA_ZIP, timeout=300) as resp:
            zf = zipfile.ZipFile(io.BytesIO(resp.read()))
    except URLError as e:
        raise SystemExit(
            "Failed to download CUAD data.zip. Check network access.\n"
            "URL: %s\nError: %s" % (_CUAD_DATA_ZIP, e)
        ) from e
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


def _merge_corpus(sample_rows: list[dict[str, Any]]):
    from ragnav.ingest.legal import ingest_legal

    seen_ctx: set[str] = set()
    documents: list[Any] = []
    blocks: list[Any] = []
    edges: list[Any] = []
    for row in sample_rows:
        ctx = row["context"]
        if ctx in seen_ctx:
            continue
        seen_ctx.add(ctx)
        did = _contract_doc_id(ctx)
        _doc, blks, g = ingest_legal(ctx, doc_id=did, metadata={"title": row.get("title", "")})
        documents.append(_doc)
        blocks.extend(blks)
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


def _evaluate(
    retriever,
    eval_rows: list[dict[str, Any]],
    gold_by_qid: dict[str, frozenset[str]],
    *,
    bm25_w: float,
    vec_w: float,
    expand_structure: bool,
    expand_graph: bool,
    graph_edge_types: Optional[set[str]],
    fusion: str = "rrf",
) -> list[tuple[list[str], frozenset[str]]]:
    results: list[tuple[list[str], frozenset[str]]] = []
    for row in eval_rows:
        qid = row["id"]
        gold = gold_by_qid.get(qid)
        if not gold:
            continue
        kw: dict[str, Any] = dict(
            top_k=10,
            bm25_weight=bm25_w,
            vector_weight=vec_w,
            expand_structure=expand_structure,
            expand_graph=expand_graph,
            use_vectors=True,
            fusion=fusion,
        )
        if expand_graph and graph_edge_types is not None:
            kw["graph_edge_types"] = graph_edge_types
        res = retriever.retrieve(row["question"], **kw)
        trace = res.trace or {}
        ranked = list(trace.get("seed_block_ids") or [])[:10]
        if not ranked:
            ranked = [b.block_id for b in res.blocks[:10]]
        results.append((ranked, gold))
    return results


def _system_block(model_name: str) -> str:
    import platform

    lines = [
        "Python: %s" % platform.python_version(),
        "Platform: %s" % platform.platform(),
    ]
    try:
        import sentence_transformers as st

        lines.append("sentence-transformers: %s" % st.__version__)
    except ImportError:
        lines.append("sentence-transformers: (import failed)")
    lines.append("Embedding model: %s" % model_name)
    return "\n".join(lines)


def main() -> int:
    from ragnav import RAGNavIndex, __version__ as ragnav_version
    from ragnav.llm.fake import FakeLLMClient
    from ragnav.retrieval import RAGNavRetriever

    parser = argparse.ArgumentParser(description="CUAD legal retrieval benchmark for RAGNav")
    parser.add_argument(
        "--sample",
        "--n-questions",
        type=int,
        default=300,
        metavar="N",
        dest="sample",
        help="Number of questions after shuffle (default: 300)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed (default: 42)")
    args = parser.parse_args()

    print("Loading CUAD test split (Atticus data.zip)…", flush=True)
    all_rows = _load_cuad_test_rows()
    rng = random.Random(args.seed)
    pool = [r for r in all_rows if r["answers"].get("text")]
    rng.shuffle(pool)
    sample_rows = pool[: args.sample]

    print("Building corpus (legal ingest graph, unique contracts in sample)…", flush=True)
    documents, blocks, edges = _merge_corpus(sample_rows)
    n_contracts = len(documents)
    print(
        "Indexing %d blocks, %d edges, %d contracts…" % (len(blocks), len(edges), n_contracts),
        flush=True,
    )

    model_name = "all-MiniLM-L6-v2"
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
    from ragnav.reranking import CrossEncoderReranker

    retriever_base = RAGNavRetriever(index=index, llm=llm)
    retriever_rerank = RAGNavRetriever(index=index, llm=llm, reranker=CrossEncoderReranker())

    gold_by_qid: dict[str, frozenset[str]] = {}
    eval_rows: list[dict[str, Any]] = []
    for row in sample_rows:
        texts = [t for t in row["answers"].get("text") or [] if t.strip()]
        if not texts:
            continue
        gset = _gold_block_ids_for_row(row, blocks)
        if not gset:
            continue
        gold_by_qid[row["id"]] = gset
        eval_rows.append(row)

    graph_types = {"next", "parent"}
    configs = [
        ("BM25-only", 1.0, 0.0, False, False, None, False, "rrf"),
        ("Embedding-only", 0.0, 1.0, False, False, None, False, "rrf"),
        ("Hybrid (0.5/0.5)", 0.5, 0.5, False, False, None, False, "rrf"),
        ("Hybrid + struct expansion", 0.5, 0.5, True, False, None, False, "rrf"),
        ("Hybrid + graph expansion", 0.5, 0.5, True, True, graph_types, False, "rrf"),
        ("Hybrid RRF + Legal Ingest + Reranker", 0.5, 0.5, False, False, None, True, "rrf"),
    ]
    scores = []
    for label, bw, vw, ex_s, ex_g, gt, use_rr, fus in configs:
        print("Evaluating: %s…" % label, flush=True)
        r = retriever_rerank if use_rr else retriever_base
        raw = _evaluate(
            r,
            eval_rows,
            gold_by_qid,
            bm25_w=bw,
            vec_w=vw,
            expand_structure=ex_s,
            expand_graph=ex_g,
            graph_edge_types=gt,
            fusion=fus,
        )
        scores.append(score_any_gold(label, raw))

    print()
    print_table(scores)

    results_dir = _BENCH_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "cuad_results.txt"
    header = f"{'Method':<30} {'R@1':>5}  {'R@3':>5}  {'R@5':>5}  {'MRR@10':>7}"
    lines_out = [
        "CUAD retrieval benchmark — RAGNav",
        "Timestamp (UTC): %s" % datetime.now(timezone.utc).isoformat(),
        "ragnav version: %s" % ragnav_version,
        "n_questions (requested): %d" % args.sample,
        "n_questions (evaluated, with gold in index): %d" % len(eval_rows),
        "seed: %d" % args.seed,
        "n_unique_contracts in index: %d" % n_contracts,
        "",
        header,
        "─" * len(header),
    ]
    for s in scores:
        lines_out.append(s.row())
    lines_out.extend(["", "--- System ---", _system_block(model_name), ""])
    out_path.write_text("\n".join(lines_out), encoding="utf-8")
    print("\nWrote %s" % out_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
