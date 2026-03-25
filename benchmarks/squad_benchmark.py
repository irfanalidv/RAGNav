"""
SQuAD Retrieval Benchmark
=========================
Problem  : Proves RAGNav hybrid BM25+embedding retrieval (RRF fusion by default,
           optional cross-encoder reranker) against BM25-only and embedding-only on
           factual Wikipedia passage retrieval.
Module   : ragnav.retrieval (RAGNavIndex, RAGNavRetriever), ragnav.reranking
Dataset  : rajpurkar/squad (HuggingFace, CC BY-SA 4.0)
Install  : pip install ragnav[embeddings] datasets
Env vars : NONE
"""

from __future__ import annotations

import argparse
import hashlib
import io
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Allow `python benchmarks/squad_benchmark.py` from repo root
_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from _runner import print_table, score as retrieval_score  # noqa: E402


def _doc_id_for_context(context: str) -> str:
    return hashlib.sha256(context.encode("utf-8")).hexdigest()[:12]


def _build_index_and_retriever():
    from ragnav import RAGNavIndex, __version__ as ragnav_version
    from ragnav.llm.fake import FakeLLMClient
    from ragnav.retrieval import RAGNavRetriever

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "The `datasets` package is required. Install with: pip install datasets"
        ) from e

    return load_dataset, RAGNavIndex, RAGNavRetriever, FakeLLMClient, ragnav_version


def _collect_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    return shuffled[:n]


def _unique_passages(sample_rows: list[dict]) -> tuple[list[Any], list[Any]]:
    from ragnav.models import Block, Document

    seen: dict[str, tuple[Document, Block]] = {}
    for row in sample_rows:
        ctx = row["context"]
        did = _doc_id_for_context(ctx)
        if did in seen:
            continue
        doc = Document(doc_id=did, source="squad", metadata={})
        blk = Block(block_id=f"{did}:p0", doc_id=did, type="paragraph", text=ctx)
        seen[did] = (doc, blk)
    ordered = sorted(seen.items(), key=lambda x: x[0])
    documents = [p[0] for _, p in ordered]
    blocks = [p[1] for _, p in ordered]
    return documents, blocks


def _gold_block_id(row: dict) -> Optional[str]:
    ctx = row["context"]
    texts = row.get("answers", {}).get("text") or []
    if not texts:
        return None
    gold = texts[0]
    if gold.lower() not in ctx.lower():
        return None
    return f"{_doc_id_for_context(ctx)}:p0"


def _evaluate(
    retriever,
    sample_rows: list[dict],
    *,
    bm25_w: float,
    vec_w: float,
    fusion: str = "rrf",
) -> list[tuple[list[str], str]]:
    results: list[tuple[list[str], str]] = []
    for row in sample_rows:
        gid = _gold_block_id(row)
        if gid is None:
            continue
        res = retriever.retrieve(
            row["question"],
            top_k=10,
            bm25_weight=bm25_w,
            vector_weight=vec_w,
            expand_structure=False,
            expand_graph=False,
            use_vectors=True,
            fusion=fusion,
        )
        ranked = [b.block_id for b in res.blocks[:10]]
        results.append((ranked, gid))
    return results


def _system_block(model_name: str) -> str:
    import platform

    lines = [
        f"Python: {platform.python_version()}",
        f"Platform: {platform.platform()}",
    ]
    try:
        import sentence_transformers as st

        lines.append("sentence-transformers: %s" % st.__version__)
    except ImportError:
        lines.append("sentence-transformers: (import failed)")
    lines.append(f"Embedding model: {model_name}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="SQuAD retrieval benchmark for RAGNav")
    parser.add_argument(
        "--sample",
        "--n-questions",
        type=int,
        default=500,
        metavar="N",
        dest="sample",
        help="Number of questions to evaluate after shuffle (default: 500)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for shuffling (default: 42)")
    args = parser.parse_args()

    load_dataset, RAGNavIndex, RAGNavRetriever, FakeLLMClient, ragnav_version = _build_index_and_retriever()

    model_name = "all-MiniLM-L6-v2"
    print("Loading SQuAD validation split…", flush=True)
    ds = load_dataset("rajpurkar/squad", split="validation")
    rows = list(ds)
    sample_rows = _collect_sample(rows, args.sample, args.seed)
    documents, blocks = _unique_passages(sample_rows)
    print(
        "Building index: %d unique passages, %d questions in sample…" % (len(blocks), len(sample_rows)),
        flush=True,
    )
    llm = FakeLLMClient()
    index = RAGNavIndex.build(
        documents=documents,
        blocks=blocks,
        llm=llm,
        use_sentence_transformers=True,
        vector_model=model_name,
        embed_batch_size=32,
    )
    from ragnav.reranking import CrossEncoderReranker

    retriever_plain = RAGNavRetriever(index=index, llm=llm)
    retriever_rerank = RAGNavRetriever(index=index, llm=llm, reranker=CrossEncoderReranker())

    configs = [
        ("BM25-only", 1.0, 0.0, retriever_plain, "rrf"),
        ("Embedding-only", 0.0, 1.0, retriever_plain, "rrf"),
        ("Hybrid RAGNav (0.5/0.5)", 0.5, 0.5, retriever_plain, "rrf"),
        ("Hybrid RAGNav (0.3/0.7)", 0.3, 0.7, retriever_plain, "rrf"),
        ("Hybrid RRF + Reranker", 0.5, 0.5, retriever_rerank, "rrf"),
    ]
    scores = []
    for label, bw, vw, retriever, fusion in configs:
        print("Evaluating: %s…" % label, flush=True)
        raw = _evaluate(retriever, sample_rows, bm25_w=bw, vec_w=vw, fusion=fusion)
        scores.append(retrieval_score(label, raw))

    print()
    print_table(scores)

    results_dir = _BENCH_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "squad_results.txt"
    buf = io.StringIO()
    buf.write("SQuAD retrieval benchmark — RAGNav\n")
    buf.write("Timestamp (UTC): %s\n" % datetime.now(timezone.utc).isoformat())
    buf.write("ragnav version: %s\n" % ragnav_version)
    buf.write("n_questions (requested): %d\n" % args.sample)
    buf.write("seed: %d\n" % args.seed)
    buf.write("unique passages indexed: %d\n" % len(blocks))
    buf.write("\n")
    header = f"{'Method':<30} {'R@1':>5}  {'R@3':>5}  {'R@5':>5}  {'MRR@10':>7}"
    buf.write(header + "\n")
    buf.write("─" * len(header) + "\n")
    for s in scores:
        buf.write(s.row() + "\n")
    buf.write("\n--- System ---\n")
    buf.write(_system_block(model_name))
    buf.write("\n")
    text = buf.getvalue()
    out_path.write_text(text, encoding="utf-8")
    print("\nWrote %s" % out_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
