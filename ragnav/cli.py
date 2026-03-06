from __future__ import annotations

import argparse
import json
from typing import Optional

from .env import load_env


def _cmd_vectorless_pdf(args: argparse.Namespace) -> int:
    load_env()
    from .llm.mistral import MistralClient
    from .cache.sqlite_cache import RetrievalCache, SqliteCacheConfig, SqliteKV
    from .pipelines.vectorless import VectorlessRagConfig, vectorless_answer, vectorless_rag_pdf_url

    llm = MistralClient()
    kv = SqliteKV(SqliteCacheConfig(path=args.cache_db)) if args.cache_db else None
    rcache = RetrievalCache(kv) if kv else None
    retrieve_fn, _doc_id = vectorless_rag_pdf_url(
        pdf_url=args.pdf_url,
        llm=llm,
        cfg=VectorlessRagConfig(max_pages=args.max_pages, max_blocks=args.max_blocks, k_final=args.k_final),
        cache_path=args.cache_path,
    )
    if args.print_hits:
        hits = retrieve_fn(args.query, retrieval_cache=rcache)
        print(json.dumps(hits, indent=2))
    ans = vectorless_answer(query=args.query, retrieve_fn=lambda q: retrieve_fn(q, retrieval_cache=rcache), llm=llm)
    print(ans)
    return 0


def _cmd_hybrid_pdf(args: argparse.Namespace) -> int:
    load_env()
    from .llm.mistral import MistralClient
    from .cache.sqlite_cache import EmbeddingCache, RetrievalCache, SqliteCacheConfig, SqliteKV
    from .pipelines.hybrid import HybridRagConfig, hybrid_answer, hybrid_rag_pdf_url

    llm = MistralClient()
    kv = SqliteKV(SqliteCacheConfig(path=args.cache_db)) if args.cache_db else None
    ecache = EmbeddingCache(kv) if kv else None
    rcache = RetrievalCache(kv) if kv else None
    cfg = HybridRagConfig(
        max_pages=args.max_pages,
        max_blocks=args.max_blocks,
        k_final=args.k_final,
        k_bm25=args.k_bm25,
        k_vec=args.k_vec,
        w_bm25=args.w_bm25,
        w_vec=args.w_vec,
        embed_batch_size=args.embed_batch_size,
    )
    retriever, _doc_id = hybrid_rag_pdf_url(
        pdf_url=args.pdf_url,
        llm=llm,
        cfg=cfg,
        cache_path=args.cache_path,
        embedding_cache=ecache,
        retrieval_cache=rcache,
    )
    if args.print_hits:
        hits = retriever.retrieve_raw(
            args.query,
            max_blocks=cfg.max_blocks,
            k_final=cfg.k_final,
            k_bm25=cfg.k_bm25,
            k_vec=cfg.k_vec,
            w_bm25=cfg.w_bm25,
            w_vec=cfg.w_vec,
            use_vectors=True,
        )
        print(json.dumps(hits, indent=2))
    ans = hybrid_answer(query=args.query, retriever=retriever, llm=llm, cfg=cfg, retrieval_cache=rcache)
    print(ans)
    return 0


def _cmd_agentic_pdf(args: argparse.Namespace) -> int:
    load_env()
    from .llm.mistral import MistralClient
    from .cache.sqlite_cache import EmbeddingCache, RetrievalCache, SqliteCacheConfig, SqliteKV
    from .pipelines.agentic import AgenticConfig
    from .pipelines.agentic_pdf import AgenticPdfConfig, agentic_pdf_url
    from .pipelines.hybrid import HybridRagConfig

    llm = MistralClient()
    kv = SqliteKV(SqliteCacheConfig(path=args.cache_db)) if args.cache_db else None
    ecache = EmbeddingCache(kv) if kv else None
    rcache = RetrievalCache(kv) if kv else None
    retrieval = HybridRagConfig(
        max_pages=args.max_pages,
        max_blocks=args.max_blocks,
        k_final=args.k_final,
        k_bm25=args.k_bm25,
        k_vec=args.k_vec,
        w_bm25=args.w_bm25,
        w_vec=args.w_vec,
        embed_batch_size=args.embed_batch_size,
    )
    agent = AgenticConfig(max_steps=args.max_steps, max_tool_blocks=args.max_tool_blocks)
    cfg = AgenticPdfConfig(retrieval=retrieval, agent=agent, max_tool_blocks=args.max_tool_blocks)

    answer, _doc_id = agentic_pdf_url(
        pdf_url=args.pdf_url,
        query=args.query,
        llm=llm,
        cfg=cfg,
        cache_path=args.cache_path,
        embedding_cache=ecache,
        retrieval_cache=rcache,
    )
    print(answer)
    return 0


def _cmd_paper_pdf(args: argparse.Namespace) -> int:
    load_env()
    from .cache.sqlite_cache import EmbeddingCache, RetrievalCache, SqliteCacheConfig, SqliteKV
    from .llm.mistral import MistralClient
    from .papers import PaperRAG, PaperRAGConfig

    llm = MistralClient()
    kv = SqliteKV(SqliteCacheConfig(path=args.cache_db)) if args.cache_db else None
    ecache = EmbeddingCache(kv) if kv else None
    rcache = RetrievalCache(kv) if kv else None

    cfg = PaperRAGConfig(
        max_pages=args.max_pages,
        build_vectors=not args.no_vectors,
        embed_batch_size=args.embed_batch_size,
        top_pages=args.top_pages,
        follow_refs=not args.no_follow_refs,
        include_next=not args.no_include_next,
        graph_hops=args.graph_hops,
        k_final=args.k_final,
        max_answer_blocks=args.max_answer_blocks,
    )

    paper = PaperRAG.from_pdf_url(
        args.pdf_url,
        llm=llm,
        pdf_name=args.pdf_name,
        cfg=cfg,
        cache_path=args.cache_path,
        embedding_cache=ecache,
    )

    if args.print_hits:
        res = paper.retrieve(args.query, cfg=cfg, retrieval_cache=rcache)
        hits = {
            "routed_pages": res.trace.get("routed_pages", []) if res.trace else [],
            "blocks": [
                {
                    "block_id": b.block_id,
                    "doc_id": b.doc_id,
                    "page": b.anchors.get("page"),
                    "title": " > ".join(b.heading_path) if b.heading_path else None,
                }
                for b in res.blocks[: args.print_hit_blocks]
            ],
        }
        print(json.dumps(hits, indent=2))

    if args.cited:
        ans = paper.answer_cited(args.query, cfg=cfg, retrieval_cache=rcache)
    else:
        ans = paper.answer(args.query, cfg=cfg, retrieval_cache=rcache)
    print(ans)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="ragnav", description="RAGNav CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_vec = sub.add_parser("vectorless-pdf", help="Vectorless RAG over a PDF URL (BM25-only).")
    p_vec.add_argument("--pdf-url", required=True, help="PDF URL (e.g. arXiv).")
    p_vec.add_argument("--query", required=True, help="User question.")
    p_vec.add_argument("--max-pages", type=int, default=25)
    p_vec.add_argument("--max-blocks", type=int, default=6)
    p_vec.add_argument("--k-final", type=int, default=6)
    p_vec.add_argument("--cache-path", default="data/document.pdf")
    p_vec.add_argument("--cache-db", default=None, help="SQLite cache DB path (enables retrieval caching).")
    p_vec.add_argument("--print-hits", action="store_true", help="Print retrieved evidence JSON.")
    p_vec.set_defaults(func=_cmd_vectorless_pdf)

    p_hyb = sub.add_parser("hybrid-pdf", help="Hybrid RAG over a PDF URL (BM25 + embeddings).")
    p_hyb.add_argument("--pdf-url", required=True, help="PDF URL (e.g. arXiv).")
    p_hyb.add_argument("--query", required=True, help="User question.")
    p_hyb.add_argument("--max-pages", type=int, default=25)
    p_hyb.add_argument("--max-blocks", type=int, default=6)
    p_hyb.add_argument("--k-final", type=int, default=6)
    p_hyb.add_argument("--k-bm25", type=int, default=30)
    p_hyb.add_argument("--k-vec", type=int, default=30)
    p_hyb.add_argument("--w-bm25", type=float, default=0.45)
    p_hyb.add_argument("--w-vec", type=float, default=0.55)
    p_hyb.add_argument("--embed-batch-size", type=int, default=64)
    p_hyb.add_argument("--cache-path", default="data/document.pdf")
    p_hyb.add_argument("--cache-db", default=None, help="SQLite cache DB path (enables embedding+retrieval caching).")
    p_hyb.add_argument("--print-hits", action="store_true", help="Print retrieved evidence JSON.")
    p_hyb.set_defaults(func=_cmd_hybrid_pdf)

    p_ag = sub.add_parser("agentic-pdf", help="Agentic retrieval over a PDF URL (hybrid retrieval tool).")
    p_ag.add_argument("--pdf-url", required=True, help="PDF URL (e.g. arXiv).")
    p_ag.add_argument("--query", required=True, help="User question.")
    p_ag.add_argument("--max-pages", type=int, default=25)
    p_ag.add_argument("--max-blocks", type=int, default=8)
    p_ag.add_argument("--k-final", type=int, default=8)
    p_ag.add_argument("--k-bm25", type=int, default=30)
    p_ag.add_argument("--k-vec", type=int, default=30)
    p_ag.add_argument("--w-bm25", type=float, default=0.45)
    p_ag.add_argument("--w-vec", type=float, default=0.55)
    p_ag.add_argument("--embed-batch-size", type=int, default=64)
    p_ag.add_argument("--max-steps", type=int, default=3)
    p_ag.add_argument("--max-tool-blocks", type=int, default=8)
    p_ag.add_argument("--cache-path", default="data/document.pdf")
    p_ag.add_argument("--cache-db", default=None, help="SQLite cache DB path (enables embedding+retrieval caching).")
    p_ag.set_defaults(func=_cmd_agentic_pdf)

    p_paper = sub.add_parser("paper-pdf", help="Paper-mode RAG over a PDF URL (page routing + cross-refs).")
    p_paper.add_argument("--pdf-url", required=True, help="PDF URL (e.g. arXiv).")
    p_paper.add_argument("--pdf-name", default="paper.pdf", help="PDF filename used for doc_id.")
    p_paper.add_argument("--query", required=True, help="User question.")
    p_paper.add_argument("--max-pages", type=int, default=25)
    p_paper.add_argument("--top-pages", type=int, default=4)
    p_paper.add_argument("--k-final", type=int, default=10)
    p_paper.add_argument("--graph-hops", type=int, default=1)
    p_paper.add_argument("--embed-batch-size", type=int, default=64)
    p_paper.add_argument("--max-answer-blocks", type=int, default=8)
    p_paper.add_argument("--no-vectors", action="store_true", help="Disable embeddings (BM25-only).")
    p_paper.add_argument("--no-follow-refs", action="store_true", help="Disable cross-reference following.")
    p_paper.add_argument("--no-include-next", action="store_true", help="Disable including 'next' blocks.")
    p_paper.add_argument("--cited", action="store_true", help="Answer with inline citations per sentence.")
    p_paper.add_argument("--cache-path", default="data/paper.pdf")
    p_paper.add_argument("--cache-db", default=None, help="SQLite cache DB path (enables embedding+retrieval caching).")
    p_paper.add_argument("--print-hits", action="store_true", help="Print routed pages + evidence JSON.")
    p_paper.add_argument("--print-hit-blocks", type=int, default=10)
    p_paper.set_defaults(func=_cmd_paper_pdf)

    args = parser.parse_args(argv)
    return int(args.func(args))

