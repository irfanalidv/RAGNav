# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.1] - 2026-10-31

### Fixed
- A retrieval-cache hit now returns the confidence and top score recorded when the result was
  computed. It used to report `LOW` / `0.0` on every hit, which made `QueryFallback` re-run
  repeated questions. Cache entries written by older versions are treated as a miss and
  recomputed once.
- Chat ingestion: `same_thread` edges pointed at a `#thread:<ts>` node that was never created, so
  graph expansion could not follow them. They now link every message of a thread (root
  included) to the others.
- Legal ingestion: a short all-caps document title was emitted as the last block, under the last
  heading. It now stays first, in document order.
- `ragnav hybrid-pdf --print-hits` now uses the retrieval cache from `--cache-db`.

### Deprecated
- `hybrid_rag_pdf_url(retrieval_cache=...)` was always ignored; passing it now emits a
  `DeprecationWarning`. Pass the cache to `hybrid_answer()` or `RAGNavRetriever.retrieve()`.

### Added
- `RetrievalCache.get_entry()` and the `RetrievalCacheEntry` dataclass;
  `RetrievalCache.set()` accepts `confidence=` and `top_score=`.

## [0.6.0] - 2026-10-31

### Fixed

- `QueryFallback` could never adopt a rephrased query against a real index: it compared
  `top_score`, which is min-max normalized per result list and is 1.0 for almost every
  result. Results are now compared by confidence level, then `top_score`.
- With `fusion="rrf"` and a single contributing list (BM25-only or vector-only), RRF scores
  depend only on rank, so `ConfidenceLevel` reflected the candidate count rather than match
  quality. A single list now keeps its own scores (same ranking; SQuAD numbers unchanged).
- `expand_structure` re-sorted results by heading depth and block id, so `result.blocks[0]`
  was rarely the best hit and truncating the list could drop the best evidence. Expansion
  now keeps relevance order. `expand_graph` returned blocks in set-iteration order, which
  varied between processes; it is now seed-first and deterministic.
- The retrieval cache key ignored `content_policy`, so a result cached without a policy was
  returned to a later call that passed `SimpleInjectionPolicy`, including the dropped
  injection blocks.
- `RAGNavIndex.build(llm=None)` silently fell back to hash-based fake embeddings when
  sentence-transformers was missing; it now builds a BM25-only index and logs a warning.
  `build_vectors=False` no longer tries to construct a Mistral client.
- `agentic_retrieve_then_answer` crashed when the model's forced final answer was plain text
  instead of JSON; the text is now returned.
- PyMuPDF documents and `SqliteKV` connections were never closed (`SqliteKV` now has
  `close()` and context-manager support).
- `MistralClient.chat` crashed on chunked (list) or empty message content from mistralai 2.x.
- `ingest_pdf_file*` derived default names with `path.split("/")` (wrong on Windows);
  `vectorless_rag_pdf_url(cache_path=None)` named the document `"None"`.
- Over-broad `except Exception` around JSON parsing and optional imports narrowed so real
  errors surface.

### Changed

- Strict typing: mypy with `check_untyped_defs` and `disallow_untyped_defs` passes and is a
  blocking CI step. Ruff rule set broadened (adds SIM, C4, RUF, PTH, RET, PIE) with no blanket
  ignores. CI uses uv, cancels superseded runs, tests the `pdf` / `messy` extras, and enforces
  85% coverage. CodeQL scanning added. `black` removed from the dev extra.
- README rewritten: architecture diagram, feature / configuration / API tables, verified
  snippets, references; unverifiable third-party comparisons removed.

### Added

- `examples/offline/`: four recipes that run with the base install and no API key, exercised by
  the test suite.
- Tests for PDF ingest (PDFs generated in-process), PaperRAG, PDF pipelines, every CLI
  subcommand, email threading. Coverage (line + branch) 76% -> 88%.

## [0.5.1] - 2026-10-31

### Fixed

- Inline citations: LLMs frequently echo the context label and write `[[BLOCK_ID=<id>]]`, which
  failed validation and made `answer_with_inline_citations` raise (callers such as nepal-gov-agent
  then silently fell back to un-synthesized answers). Citations are now normalized to `[[<id>]]`
  before validation, and the prompt tells the model to copy the bare id. Verified live with
  `gpt-4o-mini`. New public helper: `ragnav.answering.normalize_citations`.
- PDF ingest imports `pymupdf` instead of the deprecated `fitz` alias (falls back to `fitz` on
  older PyMuPDF), removing the PyMuPDF `FutureWarning`.

## [0.5.0] - 2026-10-31

### Added

- `ragnav --version` CLI flag.
- Unit tests for `MistralClient` (SDK mocked, no network) and for version metadata.
- `CITATION.cff`, issue template config, `MANIFEST.in` (ships CHANGELOG/CITATION in the sdist).
- Project URLs for Documentation, Issues, and Changelog; `Typing :: Typed` and Python 3.13 classifiers.

### Changed

- **Python 3.10+ is now required** (3.9 dropped; CI matrix is 3.10–3.13).
- `__version__` is read from installed package metadata (`importlib.metadata`).
- Build backend requirement raised to `setuptools>=77` (PEP 639 license metadata).
- Ruff rule set pinned explicitly (`E, F, W, I, UP, B`, target py310) so newer Ruff releases do not change CI results; codebase modernized to PEP 604 annotations.
- Release workflow: `skip-existing: true` on the PyPI publish step.

### Fixed

- `MistralClient` works with `mistralai` 2.x+, which moved `Mistral` to `mistralai.client`
  (previously raised "Missing optional dependency `mistralai`" even when installed). 1.x still works.

## [0.4.0] - 2026-06-14

### Added

- GitHub Actions CI: ruff lint + ruff format check + mypy (non-blocking) + pytest with coverage, across Python 3.9–3.12.
- Automated PyPI releases via OIDC Trusted Publishing (tokenless, tag-triggered).
- Coverage reporting via Codecov + coverage badge.
- SECURITY.md, CODE_OF_CONDUCT.md (Contributor Covenant), issue templates, PR template.
- pre-commit hooks (ruff + ruff-format + hygiene hooks).
- Dependabot for pip and github-actions.
- CODEOWNERS and FUNDING placeholder.
- Expanded unit test coverage to ~72% across retrieval, graphrag, pipelines, indexing, and answering (offline-first; `fail_under = 65`).

### Changed

- README: benchmark section restructured to lead with the reproducible SQuAD R@3 win; CUAD moved into an explicit "Limitations" subsection.

### Fixed

- Security: block-level and document-level ACLs are now AND-ed (intersection); previously they were unioned, which let a document ACL widen a block-level restriction.

## [0.3.0]

### Added

- Public release: hybrid BM25 + dense embeddings + structure-aware document-graph expansion, fully offline, no API key required.
