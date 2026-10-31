# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
