# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
