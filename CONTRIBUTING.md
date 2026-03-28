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

# 7. TestPyPI (optional — separate account at test.pypi.org)
# Put TWINE_USERNAME=__token__ and TWINE_PASSWORD=pypi-... in local .env (gitignored).
# Prefer bash `source` over `export $(grep ... | xargs)` — xargs breaks on quotes/spaces in values.
set -a && [ -f .env ] && . ./.env && set +a
twine upload --repository testpypi dist/* --non-interactive
# If env load fails, pass credentials explicitly (still from your shell, not committed):
# twine upload --repository testpypi dist/* --username "$TWINE_USERNAME" --password "$TWINE_PASSWORD"
# Install test (TestPyPI has no deps mirror; add PyPI as extra index if needed):
pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "ragnav[embeddings]==0.3.0"
python -c "import ragnav; print(ragnav.__version__)"

# 8. Publish (production PyPI — token from pypi.org, not TestPyPI)
set -a && [ -f .env ] && . ./.env && set +a
twine upload dist/* --non-interactive

# 9. Verify install from real PyPI (fresh venv recommended)
pip install ragnav[embeddings]
python -c "
from ragnav import RAGNavIndex, RAGNavRetriever
from ragnav.ingest.markdown import ingest_markdown_string
doc, blocks = ingest_markdown_string('Paris is the capital of France.', name='demo.md')
index = RAGNavIndex.build(documents=[doc], blocks=blocks, use_sentence_transformers=True, vector_model='all-MiniLM-L6-v2', embed_batch_size=32)
retriever = RAGNavRetriever(index=index)
result = retriever.retrieve('What is the capital of France?', top_k=3, expand_structure=False, expand_graph=False)
print(result.blocks[0].text)
print(result.confidence)
"

# 10. Tag
git tag -a v0.3.0 -m "v0.3.0: hybrid retrieval, confidence scoring, cost tracking, query fallback, legal ingest"
git push origin main --tags
```

Adjust versions in steps 7–10 when releasing a new version. **Mistral** keys (`MISTRAL_API_KEY`) are unrelated to **PyPI** upload credentials.

## Pull requests

Keep changes focused: one concern per PR, minimal diffs, and tests for new behavior where practical.
