# Contributing to RAGNav

## Development setup

Clone the repository and install the package in editable mode with optional extras:

```bash
pip install -e ".[dev,pdf,messy]"
```

Use `".[mistral]"` as well when exercising Mistral-backed examples.

## Running tests

Unit tests (no external APIs):

```bash
pytest tests/unit/ -v
```

Integration tests (network or optional heavy dependencies). Mark tests with `@pytest.mark.integration`:

```bash
pytest tests/integration/ -m integration -v
```

Quick quiet run:

```bash
pytest tests/unit/ -q
```

## Code style

- Format with **Black** and lint with **Ruff**, line length **100** (see `pyproject.toml`).
- Docstrings: explain **why** something exists or behaves a certain way, not a restatement of the signature or obvious control flow.

## Release checklist

1. Update the version in `pyproject.toml` and `ragnav/__init__.py` (`__version__`) so they match.
2. Run `pytest tests/unit/ -q` and fix failures.
3. Run `ruff check ragnav` and `black --check ragnav`.
4. Update `README.md` if public APIs or install instructions changed.
5. Build and smoke-test the sdist/wheel: `python -m build` and install the wheel in a clean virtualenv.
6. Tag the release and publish to PyPI (e.g. `twine upload dist/*`).

## Pull requests

Keep changes focused: one concern per PR, minimal diffs, and tests for new behavior where practical.
