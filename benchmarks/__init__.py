"""
Benchmarks and no-key regressions.

This package exists so you can run:
- `python -m benchmarks.offline_smoke`
- `python -m benchmarks.regression`
- `python -m benchmarks.smoke_markdown`
- `python -m benchmarks.paper_eval`
- `python -m benchmarks.entity_eval`
- `python -m benchmarks.entity_eval_excerpt`
- `python -m benchmarks.security_eval`
- `python -m benchmarks.scorecard`
- `python -m benchmarks.paper_pdf_suite`
- `python -m benchmarks.citation_eval`

It also provides small helpers for comparing:
- **RAGNav hybrid retrieval** (default engine)
- **PageIndex-style baselines** (LLM selects node_ids from a structure index)
"""

def offline_smoke_main() -> None:
    # Lazy import to avoid runpy warnings when executing benchmark modules.
    from .offline_smoke import main

    main()


def regression_main() -> None:
    # Lazy import to avoid runpy warnings when executing benchmark modules.
    from .regression import main

    main()


def smoke_markdown_main() -> None:
    # Lazy import to avoid runpy warnings when executing benchmark modules.
    from .smoke_markdown import main

    main()


def paper_eval_main() -> None:
    # Lazy import to avoid runpy warnings when executing benchmark modules.
    from .paper_eval import main

    main()


def entity_eval_main() -> None:
    # Lazy import to avoid runpy warnings when executing benchmark modules.
    from .entity_eval import main

    main()


def entity_eval_excerpt_main() -> None:
    # Lazy import to avoid runpy warnings when executing benchmark modules.
    from .entity_eval_excerpt import main

    main()


def security_eval_main() -> None:
    # Lazy import to avoid runpy warnings when executing benchmark modules.
    from .security_eval import main

    main()


def scorecard_main() -> None:
    # Lazy import to avoid runpy warnings when executing benchmark modules.
    from .scorecard import main

    main()


def paper_pdf_suite_main() -> None:
    # Lazy import to avoid runpy warnings when executing benchmark modules.
    from .paper_pdf_suite import main

    main()


def citation_eval_main() -> None:
    # Lazy import to avoid runpy warnings when executing benchmark modules.
    from .citation_eval import main

    main()

__all__ = [
    "offline_smoke_main",
    "regression_main",
    "smoke_markdown_main",
    "paper_eval_main",
    "entity_eval_main",
    "entity_eval_excerpt_main",
    "security_eval_main",
    "scorecard_main",
    "paper_pdf_suite_main",
    "citation_eval_main",
]

