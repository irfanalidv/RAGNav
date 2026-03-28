# Contributing to RAGNav

## Architecture diagram

Regenerate `assets/ragnav-architecture.png` after editing `scripts/gen_architecture.py`:

```bash
pip install pillow
python3 scripts/gen_architecture.py
```

If GitHub still shows a stale image in the README, bump the `?v=` query on the architecture `raw.githubusercontent.com` URL in `README.md` (CDN cache).

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

Run every step before `twine upload`:

```bash
# 1. Version consistent
grep "version" pyproject.toml ragnav/__init__.py

# 2. Tests green
pytest tests/unit/ -q

# 3. No AI smell
grep -rn "Initialize \|This class\|This module" ragnav/ --include="*.py"
grep -rn 'logger\..*f"' ragnav/ --include="*.py"
grep -rn "except:" ragnav/ --include="*.py"

# 4. No secrets
grep -rn "OPENAI_API_KEY\|sk-\|password\s*=" ragnav/ --include="*.py"

# 5. Ruff clean
ruff check ragnav/

# 6. Build
rm -rf dist/ build/
python -m build
twine check dist/*

# 7. TestPyPI
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ ragnav==0.3.0
python -c "import ragnav; print(ragnav.__version__)"

# 8. Publish
twine upload dist/*
git tag -a v0.3.0 -m "v0.3.0: confidence, cost tracking, fallback, reranker"
git push origin main --tags
```

Adjust the version in step 7–8 when releasing a new version.

## Pull requests

Keep changes focused: one concern per PR, minimal diffs, and tests for new behavior where practical.
